"""G2 with the NDT pose update on: lidar scan matching as a second absolute fix.

Same loop as test_g2_gnss_suspect (EKF -> centerline planner -> StanleyPID ->
command gate -> bicycle, supervisor at 50 Hz, loc HOLD escalation wired),
plus the opt-in SimLidar2D on the bicycle backend raycasting
maps/figure8_v0/landmarks.csv and the Python v0 Ndt2D matcher feeding
EkfLoc at 10 Hz. G1 and the other G2 tests run without the lidar.

Bounds stated before measuring:
- GNSS-only outage (inject_gnss_outage) of OUTAGE_S = 20 s with NDT on, per
  seed: never HOLD and hold_required never set, only RUNNING or DEGRADED
  after start-up, RUNNING at the end, no GNSS fix accepted in the outage,
  NDT accepted at least once, max lateral excursion (rear axle to
  centerline) under LAT_BAR_M = 0.50 m (the G1 catalog bar) and max pose
  error under POS_BAR_M = 0.50 m from the outage start to the end.
- The same outage with NDT off reaches HOLD (control: the outage is long
  enough to matter).
- NEES on clean NDT-on runs. For each accepted NDT match, the measurement
  NEES e' inv(R) e with e = match pose minus true pose (yaw wrapped) and R
  the matcher covariance. A consistent R gives chi-square with 3 dof, mean
  3. Bound: the per-seed mean is at most 3 + 2.33 sqrt(6 / N), the one-sided
  99 % upper limit for the mean of N independent samples. A mean over that
  says R is overconfident. A mean far under 3 says R is conservative; that
  is printed, not asserted. The filter NEES on (x, y, yaw) is printed only,
  because its samples are strongly correlated in time.
- Corrupted scan: during a GNSS-only outage, every scan in a CORRUPT_S = 2 s
  window is shifted CORRUPT_M = 1.0 m sideways in the vehicle frame. No NDT
  update is accepted in the window (each is gate rejected or not converged)
  and the pose error stays under POS_BAR_M.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from autocar_command_gate.gate import Cmd, CommandGate, GateParams
from autocar_ctrl.bicycle import L_TWIN, BicycleParams
from autocar_ctrl.stanley import StanleyPID
from autocar_loc.ekf import CHI2_GATE_3DOF, SX, SY, STH, EkfLoc
from autocar_loc.gt_ins import Localization
from autocar_loc.ndt2d import Ndt2D
from autocar_perc.source import Perception
from autocar_plan.simple_planner import CenterlinePlanner, closest_frenet
from autocar_supervisor.sm import Brain, St, Supervisor
from autocar_vehicle_interface.bicycle import BicycleBackend
from autocar_vehicle_interface.lidar2d import LandmarkMap, SimLidar2D
from test_g1_closed_loop import DT_CTRL, LANELET_ID, PLAN_EVERY, SEEDS, V_CRUISE, _lane, _LatErr, _roadside_objects, _start
from test_g2_loc_drop import _health

ROOT = Path(__file__).resolve().parents[4]
LANDMARKS = ROOT / "maps" / "figure8_v0" / "landmarks.csv"
PARAMS = [(n, f, o) for n, (f, o) in enumerate(SEEDS)]

T_OUT = 10.0
OUTAGE_S = 20.0
T_END_OUT = 32.0
LAT_BAR_M = 0.50
POS_BAR_M = 0.50
T_CLEAN = 30.0
NEES_Z99 = 2.33
T_CORRUPT = 14.0
CORRUPT_S = 2.0
CORRUPT_M = 1.0

_LMAP = LandmarkMap.from_csv(LANDMARKS)
_NDT = Ndt2D(_LMAP.sample_points())


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _run(frac: float, off: float, seed: int, t_end: float, ndt_on: bool = True,
         outage: tuple[float, float] | None = None, corrupt: tuple[float, float] | None = None) -> dict:
    lane = _lane()
    i0, st0 = _start(lane, frac, off)
    backend = BicycleBackend(st0, BicycleParams())
    clock = {"t": 0.0}
    meas: list[dict] = []

    def gt() -> tuple[float, float, float, float]:
        s = backend.state
        return (s.x, s.y, s.yaw, s.v)

    def scan() -> np.ndarray:
        s = backend.state
        pts = backend.lidar.points(backend.lidar_scan())
        bad = corrupt is not None and corrupt[0] - 1e-9 <= clock["t"] < corrupt[0] + corrupt[1] - 1e-9
        if bad:
            pts = pts + np.array([0.0, CORRUPT_M])
        meas.append({"t": clock["t"], "truth": (s.x, s.y, s.yaw), "corrupt": bad})
        return pts

    if ndt_on:
        backend.attach_lidar(SimLidar2D(_LMAP, seed=1000 + seed))
        ekf = EkfLoc(gt, seed=seed, ndt=_NDT, scan=scan)
    else:
        ekf = EkfLoc(gt, seed=seed)
    loc = Localization(ekf)
    objects = _roadside_objects(lane)
    perc = Perception("gt", lambda: objects)
    ctrl = StanleyPID(L_TWIN)
    planner = CenterlinePlanner(lane, LANELET_ID, V_CRUISE)
    lat = _LatErr(lane, i0)
    sup = Supervisor()
    assert sup.request_arm(Brain.A)
    gate = CommandGate(GateParams(dt=DT_CTRL))
    no_occ = np.zeros((0, 2), np.float32)
    lights = {"lights": []}
    traj = None
    keys = ("t", "state", "gacc", "nacc", "nrej", "nnc", "nslow", "hold_req", "abs_age", "pos_err", "lat",
            "fnees")
    rec = {k: [] for k in keys}
    injected = False
    for k in range(int(round(t_end / DT_CTRL))):
        t = k * DT_CTRL
        clock["t"] = t
        stamp = int(round(t * 1e9))
        if outage is not None and not injected and t >= outage[0] - 1e-9:
            assert ekf.inject_gnss_outage(outage[1], stamp)["ok"]
            injected = True
        st = backend.state
        n_before = len(meas)
        pose = loc.step(stamp)
        if len(meas) > n_before and ekf.last_ndt is not None:
            m = meas[-1]
            res = ekf.last_ndt
            m["converged"] = res.converged
            if res.converged:
                tx, ty, tyaw = m["truth"]
                e = np.array([res.x - tx, res.y - ty, _wrap(res.yaw - tyaw)])
                m["nees"] = float(e @ np.linalg.solve(res.cov, e))
        if loc.hold_required and sup.state in (St.RUNNING, St.DEGRADED):
            assert sup.request_hold()
        state = sup.step(_health(loc.healthy))
        if k % PLAN_EVERY == 0:
            perc.step(stamp)
            traj = planner.step(pose.xyyaw(), perc.msg(), lights)
        fx, fy = pose.x + L_TWIN * math.cos(pose.theta), pose.y + L_TWIN * math.sin(pose.theta)
        e, th, vref, _ = closest_frenet((fx, fy, pose.theta), traj)
        raw = Cmd(t, ctrl.lateral(e, th, st.v), ctrl.longitudinal(st.v, vref, DT_CTRL), False, k, "A")
        out = gate.tick(t, raw, state.name, None, no_occ, pose.xyyaw(), st.v, False)
        backend.set_command(out.cmd.steer, out.cmd.accel)
        ef = np.array([ekf.s[SX] - st.x, ekf.s[SY] - st.y, _wrap(ekf.s[STH] - st.yaw)])
        pp = ekf.P[np.ix_([SX, SY, STH], [SX, SY, STH])]
        for key, val in (("t", t), ("state", state), ("gacc", ekf.n_fix_accepted), ("nacc", ekf.n_ndt_accepted),
                         ("nrej", ekf.n_ndt_rejected), ("nnc", ekf.n_ndt_not_converged), ("nslow", ekf.n_ndt_slow),
                         ("hold_req", loc.hold_required), ("abs_age", pose.abs_age_s),
                         ("pos_err", math.hypot(pose.x - st.x, pose.y - st.y)), ("lat", lat(st.x, st.y)),
                         ("fnees", float(ef @ np.linalg.solve(pp, ef)))):
            rec[key].append(val)
        backend.step(DT_CTRL)
    r = {k: (v if k == "state" else np.array(v)) for k, v in rec.items()}
    r["ekf"], r["meas"] = ekf, meas
    return r


def _delta(r: dict, key: str, t0: float, t1: float) -> int:
    t = r["t"]
    w = np.where((t >= t0 - 1e-9) & (t < t1 - 1e-9))[0]
    before = r[key][w[0] - 1] if w[0] > 0 else 0
    return int(r[key][w[-1]] - before)


def _timeline(r: dict) -> str:
    out, prev = [], None
    for s, t in zip(r["state"], r["t"]):
        if s is not prev:
            out.append(f"{s.name}@{t:.2f}")
            prev = s
    return " ".join(out)


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_ndt_gnss_outage_keeps_running(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, T_END_OUT, outage=(T_OUT, OUTAGE_S))
    t, states = r["t"], r["state"]
    win = t >= T_OUT - 1e-9
    t1 = T_OUT + OUTAGE_S
    g_out = _delta(r, "gacc", T_OUT, t1)
    na, nr, nn, ns = (_delta(r, k, T_OUT, t1) for k in ("nacc", "nrej", "nnc", "nslow"))
    max_lat = float(r["lat"][win].max())
    max_pos = float(r["pos_err"][win].max())
    max_age = float(r["abs_age"][(t >= T_OUT - 1e-9) & (t < t1 - 1e-9)].max())
    print(f"\nG2 ndt gnss_outage seed={seed} {OUTAGE_S:.0f} s at {T_OUT:.0f} s  timeline: {_timeline(r)}  "
          f"hold={'HOLD' in _timeline(r)}  gnss_accepted_in_outage={g_out}  ndt in outage: accepted={na} "
          f"rejected={nr} not_converged={nn} slow={ns}  max_abs_age_in_outage={max_age:.2f} s  "
          f"max_lat={max_lat:.4f} m  max_pos_err={max_pos:.4f} m")
    assert {s for s, ti in zip(states, t) if ti >= 1.0} <= {St.RUNNING, St.DEGRADED}
    assert not r["hold_req"].any()
    assert states[-1] is St.RUNNING
    assert g_out == 0 and na > 0
    assert max_lat < LAT_BAR_M
    assert max_pos < POS_BAR_M


def test_g2_gnss_outage_without_ndt_holds() -> None:
    frac, off = SEEDS[0]
    r = _run(frac, off, 0, T_END_OUT, ndt_on=False, outage=(T_OUT, OUTAGE_S))
    print(f"\nG2 gnss_outage ndt off seed=0 timeline: {_timeline(r)}")
    assert St.HOLD in r["state"]


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_ndt_clean_nees(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, T_CLEAN)
    ekf = r["ekf"]
    nees = np.array([m["nees"] for m in r["meas"] if m.get("converged") and "nees" in m])
    n = len(nees)
    bound = 3.0 + NEES_Z99 * math.sqrt(6.0 / n)
    fn = r["fnees"][r["t"] >= 1.0]
    print(f"\nG2 ndt clean seed={seed} scans={len(r['meas'])} ndt accepted={ekf.n_ndt_accepted} "
          f"rejected={ekf.n_ndt_rejected} not_converged={ekf.n_ndt_not_converged} slow={ekf.n_ndt_slow}  "
          f"gnss accepted={ekf.n_fix_accepted} rejected={ekf.n_fix_rejected}  "
          f"meas NEES n={n} mean={nees.mean():.3f} median={np.median(nees):.3f} max={nees.max():.2f} "
          f"bound={bound:.3f} (consistent mean 3)  filter NEES mean={fn.mean():.3f} (printed only)  "
          f"timeline: {_timeline(r)}")
    assert {s for s, ti in zip(r["state"], r["t"]) if ti >= 1.0} == {St.RUNNING}
    assert nees.mean() <= bound


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_ndt_corrupted_scan_rejected(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, 20.0, outage=(T_OUT, 8.0), corrupt=(T_CORRUPT, CORRUPT_S))
    t1 = T_CORRUPT + CORRUPT_S
    na, nr, nn = (_delta(r, k, T_CORRUPT, t1) for k in ("nacc", "nrej", "nnc"))
    n_bad = sum(m["corrupt"] for m in r["meas"])
    w = (r["t"] >= T_CORRUPT - 1e-9) & (r["t"] < t1 + 0.5)
    max_pos = float(r["pos_err"][w].max())
    nis = r["ekf"].last_ndt_nis
    print(f"\nG2 ndt corrupted scan seed={seed} {CORRUPT_M:.1f} m shift for {CORRUPT_S:.0f} s: corrupted scans={n_bad} "
          f"accepted={na} gate_rejected={nr} not_converged={nn}  max_pos_err={max_pos:.4f} m  "
          f"gate={CHI2_GATE_3DOF}  timeline: {_timeline(r)}")
    assert n_bad == int(round(CORRUPT_S * 10.0))
    assert na == 0 and nr + nn == n_bad
    assert max_pos < POS_BAR_M
