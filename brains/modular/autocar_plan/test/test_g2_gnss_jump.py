"""G2 'GNSS jump is gated out' on the bicycle loop with the EKF backend.

Same loop as test_g2_loc_drop (EKF -> centerline planner -> StanleyPID ->
command gate -> bicycle, supervisor at 50 Hz). At T_JUMP,
/scenario/inject/gnss_jump adds a fixed 5 m lateral offset to every GNSS fix
for 2 s. The EKF chi-square gate (autocar_loc.ekf.CHI2_GATE_2DOF) must reject
every offset fix, so the pose keeps dead reckoning instead of following the
bad GNSS, the absolute fix goes stale and the supervisor goes DEGRADED. Once
the offset ends, fixes are accepted again and RUNNING returns after T_recover.

Bound stated before measuring: on a clean run (no injection), the gate
rejects at most FALSE_REJECT_BOUND = 1 % of the gated fixes per seed. A
consistent filter rejects 0.27 % at the 99.7 % gate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from autocar_command_gate.gate import Cmd, CommandGate, GateParams
from autocar_ctrl.bicycle import L_TWIN, BicycleParams
from autocar_ctrl.stanley import StanleyPID
from autocar_loc.ekf import CHI2_GATE_2DOF, GNSS_JUMP_M, EkfLoc
from autocar_loc.gt_ins import Localization
from autocar_loc.pose import T_ABS_STALE_S, UNHEALTHY_HOLD_S
from autocar_perc.source import Perception
from autocar_plan.simple_planner import CenterlinePlanner, closest_frenet
from autocar_supervisor.sm import DT, T_RECOVER_S, Brain, St, Supervisor
from autocar_vehicle_interface.bicycle import BicycleBackend
from test_g1_closed_loop import DT_CTRL, LANELET_ID, PLAN_EVERY, SEEDS, V_CRUISE, _lane, _LatErr, _roadside_objects, _start
from test_g2_loc_drop import _health

T_JUMP = 10.0
T_END = 16.0
JUMP_S = 2.0
FALSE_REJECT_BOUND = 0.01
EXCURSION_BAR_M = 1.0
# Same freshness arithmetic as the EKF loc_drop case: the last accepted fix
# is at 9.90 s, every fix from 10.00 s is rejected, its age first exceeds
# T_ABS_STALE_S at 0.42 s into the jump and DEGRADED follows after the 1.0 s
# hold, at 1.44 s.
STALE_AT_S = 0.42
DEGRADED_AT_S = 1.44


def _run(frac: float, off: float, seed: int, jump_s: float | None, t_end: float = T_END) -> dict:
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
    rec = {k: [] for k in ("t", "state", "brain", "acc", "rej", "jump", "abs_age", "pos_err", "lat")}
    injected = False
    for k in range(int(round(t_end / DT_CTRL))):
        t = k * DT_CTRL
        stamp = int(round(t * 1e9))
        if jump_s is not None and not injected and t >= T_JUMP - 1e-9:
            assert ekf.inject_gnss_jump(jump_s, stamp)["ok"]
            injected = True
        st = backend.state
        pose = loc.step(stamp)
        state = sup.step(_health(loc.healthy))
        if k % PLAN_EVERY == 0:
            perc.step(stamp)
            traj = planner.step(pose.xyyaw(), perc.msg(), lights)
        fx, fy = pose.x + L_TWIN * math.cos(pose.theta), pose.y + L_TWIN * math.sin(pose.theta)
        e, th, vref, _ = closest_frenet((fx, fy, pose.theta), traj)
        raw = Cmd(t, ctrl.lateral(e, th, st.v), ctrl.longitudinal(st.v, vref, DT_CTRL), False, k, "A")
        out = gate.tick(t, raw, state.name, None, no_occ, pose.xyyaw(), st.v, False)
        backend.set_command(out.cmd.steer, out.cmd.accel)
        rec["t"].append(t)
        rec["state"].append(state)
        rec["brain"].append(sup.brain)
        rec["acc"].append(ekf.n_fix_accepted)
        rec["rej"].append(ekf.n_fix_rejected)
        rec["jump"].append(ekf.gnss_jump_active)
        rec["abs_age"].append(pose.abs_age_s)
        rec["pos_err"].append(math.hypot(pose.x - st.x, pose.y - st.y))
        rec["lat"].append(lat(st.x, st.y))
        backend.step(DT_CTRL)
    r = {k: np.array(v) if k not in ("state", "brain") else v for k, v in rec.items()}
    r["ekf"] = ekf
    return r


def _first(t: np.ndarray, mask, start: float) -> float | None:
    idx = np.where((t >= start - 1e-9) & mask)[0]
    return None if idx.size == 0 else float(t[idx[0]])


def _window_counts(r: dict, t0: float, t1: float) -> tuple[int, int]:
    """Accepted and rejected fixes on estimates with t0 <= t < t1."""
    t = r["t"]
    w = np.where((t >= t0 - 1e-9) & (t < t1 - 1e-9))[0]
    a0 = r["acc"][w[0] - 1] if w[0] > 0 else 0
    j0 = r["rej"][w[0] - 1] if w[0] > 0 else 0
    return int(r["acc"][w[-1]] - a0), int(r["rej"][w[-1]] - j0)


@pytest.mark.parametrize("seed,frac,off", [(n, f, o) for n, (f, o) in enumerate(SEEDS)])
def test_g2_gnss_jump_2s_rejected_and_degraded(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, JUMP_S)
    t, states = r["t"], r["state"]
    in_jump = (t >= T_JUMP - 1e-9) & (t < T_JUMP + JUMP_S - 1e-9)
    assert np.array_equal(r["jump"], in_jump)
    acc_j, rej_j = _window_counts(r, T_JUMP, T_JUMP + JUMP_S)
    deg = np.array([s is St.DEGRADED for s in states])
    run = np.array([s is St.RUNNING for s in states])
    t_stale = _first(t, r["abs_age"] > T_ABS_STALE_S, T_JUMP)
    t_deg = _first(t, deg, T_JUMP)
    t_fresh = _first(t, r["abs_age"] <= T_ABS_STALE_S, t_stale) if t_stale is not None else None
    t_back = _first(t, run, t_deg) if t_deg is not None else None
    acc_after, rej_after = _window_counts(r, T_JUMP + JUMP_S, T_END)
    win = t >= T_JUMP - 1e-9
    max_lat = float(r["lat"][win].max())
    max_pos = float(r["pos_err"][win].max())
    print(f"G2 gnss_jump ekf seed={seed} {GNSS_JUMP_M:.1f} m {JUMP_S:.2f} s  accepted_in_jump={acc_j} "
          f"rejected_in_jump={rej_j}  stale_at={None if t_stale is None else round(t_stale - T_JUMP, 2)} "
          f"degraded_at={None if t_deg is None else round(t_deg - T_JUMP, 2)} "
          f"fresh_at={None if t_fresh is None else round(t_fresh - T_JUMP, 2)} "
          f"running_again_at={None if t_back is None else round(t_back - T_JUMP, 2)} (s after jump start)  "
          f"recover={None if t_back is None or t_fresh is None else round(t_back - t_fresh, 2)} s  "
          f"accepted_after={acc_after} rejected_after={rej_after}  max_lat={max_lat:.4f} m  max_pos_err={max_pos:.4f} m")
    assert {s for s, ti in zip(states, t) if 1.0 <= ti < T_JUMP} == {St.RUNNING}
    assert set(r["brain"]) == {Brain.A}
    assert {s for s, w in zip(states, win) if w} == {St.RUNNING, St.DEGRADED}
    assert states[-1] is St.RUNNING
    # every offset fix rejected: 10 Hz for 2 s
    assert acc_j == 0 and rej_j == int(round(JUMP_S * 10.0))
    # the pose did not follow the bad GNSS
    assert max_lat < EXCURSION_BAR_M
    assert max_pos < EXCURSION_BAR_M
    # freshness trips DEGRADED
    assert t_stale - T_JUMP == pytest.approx(STALE_AT_S, abs=DT_CTRL / 2)
    assert t_deg - t_stale > UNHEALTHY_HOLD_S
    assert t_deg - T_JUMP == pytest.approx(DEGRADED_AT_S, abs=DT_CTRL / 2)
    # first fix after the jump is accepted, RUNNING after T_recover
    assert t_fresh == pytest.approx(T_JUMP + JUMP_S, abs=DT_CTRL / 2)
    assert acc_after > 0 and rej_after == 0
    assert t_back - t_fresh == pytest.approx(T_RECOVER_S - DT, abs=DT / 2)


@pytest.mark.parametrize("seed,frac,off", [(n, f, o) for n, (f, o) in enumerate(SEEDS)])
def test_g2_clean_run_false_reject_rate(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, seed, None)
    ekf = r["ekf"]
    n = ekf.n_fix_accepted + ekf.n_fix_rejected
    rate = ekf.n_fix_rejected / n
    print(f"G2 clean ekf seed={seed} gate={CHI2_GATE_2DOF} accepted={ekf.n_fix_accepted} "
          f"rejected={ekf.n_fix_rejected} false_reject_rate={rate * 100:.2f} % bound={FALSE_REJECT_BOUND * 100:.1f} %")
    assert n == int(round(T_END * 10.0)) - 1  # every 10 Hz fix after the initializing one
    assert rate <= FALSE_REJECT_BOUND
    assert {s for s, ti in zip(r["state"], r["t"]) if ti >= 1.0} == {St.RUNNING}
