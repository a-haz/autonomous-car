"""G2 GNSS-suspect latch and supervisor HOLD escalation, EKF backend.

Same loop as test_g2_gnss_jump, plus the escalation wiring: when
Localization.hold_required is set (HealthMonitor: autocar_loc.ekf
gnss_suspect latched, or sigma over HOLD_SIGMA_M with a stale absolute fix),
the loop requests the existing supervisor HOLD, and the command gate brakes
the car to a stop at -a_hold as for any supervisor HOLD.

Bounds, all stated before measuring:
- Never-ending 5 m offset over 120 s: zero offset fixes accepted. The first
  rejected fix is at 10.00 s, the rejection run first exceeds T_SUSPECT_S at
  the 13.10 s fix, so the latch trips at LATCH_AT_S = 3.10 s into the jump,
  and HOLD is requested on the same tick. The car is under V_STOP_MPS by
  STOP_BY_S = 6.0 s into the jump and stays stopped in HOLD to the end. Max
  pose error and max true lateral excursion before the stop stay under the
  1.0 m G2 bar. RUNNING never returns after DEGRADED.
- The 2 s jump keeps its numbers (DEGRADED at 1.44 s, RUNNING T_recover after
  the first fix after the jump, no rejection after it) and never sets the
  latch or HOLD.
- Clean 120 s runs never set the latch or hold_required.
- Pure 120 s loc_drop: no fix is rejected during the drop and the latch never
  trips. The DEGRADED dwell limit requests HOLD T_DEGRADED_MAX_S after
  DEGRADED, and the car is under V_STOP_MPS within OUTAGE_STOP_AFTER_HOLD_S of
  HOLD (v_degrade / a_hold plus the jerk ramp is under 2.0 s). With the steer
  tracking the dead-reckoned pose in DEGRADED, max pose error and max true
  lateral excursion before the stop stay under the 1.0 m G2 bar. These bounds
  were set before measuring, after the earlier sigma-only HOLD failed them.
- 2 s loc_drop: DEGRADED and back to RUNNING, hold_required never set.
- /reset after a 6 s jump has ended: while latched, clean fixes are still
  rejected; after reset the next fix is accepted and loc is healthy again.
  The supervisor stays in HOLD until an operator resume.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from autocar_command_gate.gate import Cmd, CommandGate, GateParams
from autocar_ctrl.bicycle import L_TWIN, BicycleParams
from autocar_ctrl.stanley import StanleyPID
from autocar_loc.ekf import GNSS_JUMP_M, T_SUSPECT_S, EkfLoc
from autocar_loc.gt_ins import Localization
from autocar_loc.pose import HOLD_SIGMA_M, T_ABS_STALE_S, T_DEGRADED_MAX_S
from autocar_perc.source import Perception
from autocar_plan.simple_planner import CenterlinePlanner, closest_frenet
from autocar_supervisor.sm import DT, T_RECOVER_S, Brain, St, Supervisor
from autocar_vehicle_interface.bicycle import BicycleBackend
from test_g1_closed_loop import DT_CTRL, LANELET_ID, PLAN_EVERY, SEEDS, V_CRUISE, _lane, _LatErr, _roadside_objects, _start
from test_g2_loc_drop import _health

T_JUMP = 10.0
T_LONG = 120.0
FOREVER_S = 1000.0  # longer than any horizon here
LATCH_AT_S = 3.10
STOP_BY_S = 6.0
V_STOP_MPS = 0.05
EXCURSION_BAR_M = 1.0
OLD_READMIT_S = 37.0  # earliest readmission of the 5 m spoof by the gate alone
DEGRADED_AT_S = 1.44
RESET_JUMP_S = 6.0
T_RESET = 20.0
OUTAGE_STOP_AFTER_HOLD_S = 2.0

PARAMS = [(n, f, o) for n, (f, o) in enumerate(SEEDS)]


def _run(frac: float, off: float, seed: int, t_end: float, jump_s: float | None = None,
         drop_s: float | None = None, reset_at: float | None = None) -> dict:
    lane = _lane()
    i0, st0 = _start(lane, frac, off)
    backend = BicycleBackend(st0, BicycleParams())

    def gt() -> tuple[float, float, float, float]:
        s = backend.state
        return (s.x, s.y, s.yaw, s.v)

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
    keys = ("t", "state", "acc", "rej", "jump", "suspect", "hold_req", "healthy", "abs_age", "sigma",
            "pos_err", "lat", "v")
    rec = {k: [] for k in keys}
    injected = was_reset = False
    for k in range(int(round(t_end / DT_CTRL))):
        t = k * DT_CTRL
        stamp = int(round(t * 1e9))
        if not injected and t >= T_JUMP - 1e-9:
            if jump_s is not None:
                assert ekf.inject_gnss_jump(jump_s, stamp)["ok"]
            if drop_s is not None:
                assert ekf.inject_loc_drop(drop_s, stamp)["ok"]
            injected = True
        if reset_at is not None and not was_reset and t >= reset_at - 1e-9:
            assert ekf.reset()["ok"]
            was_reset = True
        st = backend.state
        pose = loc.step(stamp)
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
        for key, val in (("t", t), ("state", state), ("acc", ekf.n_fix_accepted), ("rej", ekf.n_fix_rejected),
                         ("jump", ekf.gnss_jump_active), ("suspect", ekf.gnss_suspect),
                         ("hold_req", loc.hold_required), ("healthy", loc.healthy), ("abs_age", pose.abs_age_s),
                         ("sigma", pose.sigma_pos), ("pos_err", math.hypot(pose.x - st.x, pose.y - st.y)),
                         ("lat", lat(st.x, st.y)), ("v", st.v)):
            rec[key].append(val)
        backend.step(DT_CTRL)
    r = {k: (v if k == "state" else np.array(v)) for k, v in rec.items()}
    r["ekf"], r["sup"] = ekf, sup
    return r


def _first(t: np.ndarray, mask, start: float) -> float | None:
    idx = np.where((t >= start - 1e-9) & np.asarray(mask))[0]
    return None if idx.size == 0 else float(t[idx[0]])


def _counts(r: dict, t0: float, t1: float) -> tuple[int, int]:
    """Accepted and rejected fixes on estimates with t0 <= t < t1."""
    t = r["t"]
    w = np.where((t >= t0 - 1e-9) & (t < t1 - 1e-9))[0]
    a0 = r["acc"][w[0] - 1] if w[0] > 0 else 0
    j0 = r["rej"][w[0] - 1] if w[0] > 0 else 0
    return int(r["acc"][w[-1]] - a0), int(r["rej"][w[-1]] - j0)


def _is(states, s: St) -> np.ndarray:
    return np.array([x is s for x in states])


def _stop_view(r: dict, t_hold: float) -> dict:
    t = r["t"]
    t_stop = _first(t, r["v"] < V_STOP_MPS, t_hold)
    before = (t >= T_JUMP - 1e-9) & (t < (t_stop if t_stop is not None else t[-1] + 1.0) - 1e-9)
    return {"t_stop": t_stop, "max_pos": float(r["pos_err"][before].max()),
            "max_lat": float(r["lat"][before].max()), "v_hold": float(r["v"][t >= t_hold - 1e-9][0])}


def _rel(x: float | None) -> str:
    return "none" if x is None else f"{x - T_JUMP:.2f}"


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_never_ending_offset_latches_and_holds(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, T_LONG, jump_s=FOREVER_S)
    t, states = r["t"], r["state"]
    ekf = r["ekf"]
    acc_j, rej_j = _counts(r, T_JUMP, T_LONG)
    t_latch = None if ekf.t_suspect_ns is None else ekf.t_suspect_ns * 1e-9
    t_deg = _first(t, _is(states, St.DEGRADED), T_JUMP)
    t_hold = _first(t, _is(states, St.HOLD), T_JUMP)
    s = _stop_view(r, t_hold) if t_hold is not None else {}
    print(f"G2 suspect never-ending ekf seed={seed} {GNSS_JUMP_M:.1f} m  T_SUSPECT={T_SUSPECT_S:.1f} s  "
          f"accepted_spoofed={acc_j} rejected={rej_j}  degraded_at={_rel(t_deg)} latch_at={_rel(t_latch)} "
          f"hold_at={_rel(t_hold)} stop_at={_rel(s.get('t_stop'))} (s after jump start, v<{V_STOP_MPS} m/s)  "
          f"v_at_hold={s.get('v_hold', float('nan')):.3f} m/s  max_pos_err_before_stop={s.get('max_pos', float('nan')):.4f} m  "
          f"max_lat_before_stop={s.get('max_lat', float('nan')):.4f} m  final_v={r['v'][-1]:.4f} m/s  "
          f"final_state={states[-1].name} reason={r['sup'].reason}")
    assert {x for x, ti in zip(states, t) if 1.0 <= ti < T_JUMP} == {St.RUNNING}
    assert acc_j == 0  # no spoofed fix ever accepted
    assert t_latch is not None and t_latch - T_JUMP == pytest.approx(LATCH_AT_S, abs=DT_CTRL / 2)
    assert t_deg - T_JUMP == pytest.approx(DEGRADED_AT_S, abs=DT_CTRL / 2)
    assert t_hold == pytest.approx(t_latch, abs=DT_CTRL / 2)
    assert not _is(states, St.RUNNING)[t >= t_deg - 1e-9].any()
    assert s["t_stop"] is not None and s["t_stop"] - T_JUMP <= STOP_BY_S < OLD_READMIT_S
    after_stop = t >= s["t_stop"] - 1e-9
    assert {x for x, a in zip(states, after_stop) if a} == {St.HOLD}
    assert r["v"][after_stop].max() < V_STOP_MPS
    assert s["max_pos"] < EXCURSION_BAR_M
    assert s["max_lat"] < EXCURSION_BAR_M
    assert r["suspect"][t >= t_latch - 1e-9].all()


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_2s_jump_unchanged_with_latch(seed: int, frac: float, off: float) -> None:
    t_end = 16.0
    r = _run(frac, off, seed, t_end, jump_s=2.0)
    t, states = r["t"], r["state"]
    acc_j, rej_j = _counts(r, T_JUMP, T_JUMP + 2.0)
    acc_a, rej_a = _counts(r, T_JUMP + 2.0, t_end)
    t_stale = _first(t, r["abs_age"] > T_ABS_STALE_S, T_JUMP)
    t_deg = _first(t, _is(states, St.DEGRADED), T_JUMP)
    t_fresh = _first(t, r["abs_age"] <= T_ABS_STALE_S, t_stale)
    t_back = _first(t, _is(states, St.RUNNING), t_deg)
    print(f"G2 suspect 2s-jump ekf seed={seed}  accepted_in_jump={acc_j} rejected_in_jump={rej_j}  "
          f"degraded_at={_rel(t_deg)} fresh_at={_rel(t_fresh)} running_again_at={_rel(t_back)}  "
          f"accepted_after={acc_a} rejected_after={rej_a}  latch_ever={bool(r['suspect'].any())} "
          f"hold_required_ever={bool(r['hold_req'].any())} states={sorted({x.name for x in states})}")
    assert acc_j == 0 and rej_j == 20
    assert t_deg - T_JUMP == pytest.approx(DEGRADED_AT_S, abs=DT_CTRL / 2)
    assert t_fresh == pytest.approx(T_JUMP + 2.0, abs=DT_CTRL / 2)
    assert t_back - t_fresh == pytest.approx(T_RECOVER_S - DT, abs=DT / 2)
    assert acc_a > 0 and rej_a == 0
    assert not r["suspect"].any() and not r["hold_req"].any()
    assert St.HOLD not in set(states) and states[-1] is St.RUNNING


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_clean_run_never_latches(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, T_LONG)
    ekf = r["ekf"]
    n = ekf.n_fix_accepted + ekf.n_fix_rejected
    print(f"G2 suspect clean ekf seed={seed} {T_LONG:.0f} s  accepted={ekf.n_fix_accepted} rejected={ekf.n_fix_rejected} "
          f"({ekf.n_fix_rejected / n * 100:.2f} %)  latch_ticks={int(r['suspect'].sum())} "
          f"hold_required_ticks={int(r['hold_req'].sum())} max_sigma={float(r['sigma'].max()):.4f} m")
    assert not r["suspect"].any() and not r["hold_req"].any()
    assert {x for x, ti in zip(r["state"], r["t"]) if ti >= 1.0} == {St.RUNNING}


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_pure_loc_drop_never_latches(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, T_LONG, drop_s=FOREVER_S)
    t, states = r["t"], r["state"]
    acc_d, rej_d = _counts(r, T_JUMP, T_LONG)
    t_deg = _first(t, _is(states, St.DEGRADED), T_JUMP)
    t_sig = _first(t, r["sigma"] > HOLD_SIGMA_M, T_JUMP)
    t_hold = _first(t, _is(states, St.HOLD), T_JUMP)
    s = _stop_view(r, t_hold) if t_hold is not None else {}
    print(f"G2 suspect loc_drop ekf seed={seed} {T_LONG - T_JUMP:.0f} s drop  accepted={acc_d} rejected={rej_d}  "
          f"latch_ever={bool(r['suspect'].any())}  degraded_at={_rel(t_deg)} sigma_over_{HOLD_SIGMA_M}_at={_rel(t_sig)} "
          f"hold_at={_rel(t_hold)} stop_at={_rel(s.get('t_stop'))}  v_at_hold={s.get('v_hold', float('nan')):.3f} m/s  "
          f"max_pos_err_before_stop={s.get('max_pos', float('nan')):.4f} m  "
          f"max_lat_before_stop={s.get('max_lat', float('nan')):.4f} m  final_state={states[-1].name}")
    assert acc_d == 0 and rej_d == 0
    assert not r["suspect"].any()
    assert t_deg - T_JUMP == pytest.approx(DEGRADED_AT_S, abs=DT_CTRL / 2)
    assert not _is(states, St.RUNNING)[t >= t_deg - 1e-9].any()
    assert t_hold is not None and t_hold - t_deg == pytest.approx(T_DEGRADED_MAX_S, abs=DT_CTRL / 2)
    assert t_sig is None or t_sig >= t_hold - 1e-9  # the dwell limit acts first
    assert s["t_stop"] is not None and s["t_stop"] - t_hold <= OUTAGE_STOP_AFTER_HOLD_S
    after_stop = t >= s["t_stop"] - 1e-9
    assert {x for x, a in zip(states, after_stop) if a} == {St.HOLD}
    assert r["v"][after_stop].max() < V_STOP_MPS
    assert s["max_pos"] < EXCURSION_BAR_M
    assert s["max_lat"] < EXCURSION_BAR_M


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_2s_loc_drop_recovers_without_hold(seed: int, frac: float, off: float) -> None:
    t_end = 16.0
    r = _run(frac, off, seed, t_end, drop_s=2.0)
    t, states = r["t"], r["state"]
    t_deg = _first(t, _is(states, St.DEGRADED), T_JUMP)
    t_back = _first(t, _is(states, St.RUNNING), t_deg)
    print(f"G2 suspect 2s-loc_drop ekf seed={seed}  degraded_at={_rel(t_deg)} running_again_at={_rel(t_back)}  "
          f"hold_required_ever={bool(r['hold_req'].any())} max_lat={float(r['lat'][t >= T_JUMP - 1e-9].max()):.4f} m")
    assert t_deg - T_JUMP == pytest.approx(DEGRADED_AT_S, abs=DT_CTRL / 2)
    assert t_back is not None and not r["hold_req"].any()
    assert St.HOLD not in set(states) and states[-1] is St.RUNNING


@pytest.mark.parametrize("seed,frac,off", PARAMS)
def test_g2_reset_clears_latch(seed: int, frac: float, off: float) -> None:
    t_end = 26.0
    r = _run(frac, off, seed, t_end, jump_s=RESET_JUMP_S, reset_at=T_RESET)
    t, states = r["t"], r["state"]
    acc_l, rej_l = _counts(r, T_JUMP + RESET_JUMP_S, T_RESET)
    acc_r, rej_r = _counts(r, T_RESET, t_end)
    t_first_acc = _first(t, np.diff(r["acc"], prepend=r["acc"][0]) > 0, T_RESET)
    t_healthy = _first(t, r["healthy"], T_RESET)
    print(f"G2 suspect reset ekf seed={seed}  latched clean fixes {T_JUMP + RESET_JUMP_S:.0f}-{T_RESET:.0f} s: "
          f"accepted={acc_l} rejected={rej_l}  after reset: accepted={acc_r} rejected={rej_r} "
          f"first_accepted_at={t_first_acc} healthy_at={t_healthy}  latch_after_reset={bool(r['suspect'][t >= T_RESET - 1e-9].any())} "
          f"final_state={states[-1].name}")
    assert r["suspect"][(t >= T_JUMP + RESET_JUMP_S - 1e-9) & (t < T_RESET - 1e-9)].all()
    assert acc_l == 0 and rej_l == int(round((T_RESET - T_JUMP - RESET_JUMP_S) * 10.0))
    assert not r["suspect"][t >= T_RESET - 1e-9].any()
    assert t_first_acc == pytest.approx(T_RESET, abs=DT_CTRL / 2)
    assert acc_r > 0
    assert t_healthy is not None and r["healthy"][t >= t_healthy - 1e-9].all()
    assert {x for x, ti in zip(states, t) if ti >= T_RESET} == {St.HOLD}  # no auto-resume
