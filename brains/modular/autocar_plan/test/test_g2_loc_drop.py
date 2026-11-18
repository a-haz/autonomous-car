"""G2 'GNSS dropout 2 s gives DEGRADED, not flip' on the bicycle loop.

Same loop as G1 (autocar_loc GT/INS stub -> centerline planner -> StanleyPID
-> bicycle), with the supervisor stepped every 20 ms on Health(loc =
Localization.healthy, everything else healthy). /scenario/inject/loc_drop
mutes GNSS, the stub dead reckons, and its covariance grows by the bound
documented in autocar_loc.gt_ins. The controller output goes through the
PR-02 command gate with the supervisor state, so the DEGRADED clamp acts on
the plant. RSS and occupancy inputs to the gate are empty here; the planner
owns obstacle stops, as in G1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from autocar_command_gate.gate import Cmd, CommandGate, GateParams
from autocar_ctrl.bicycle import L_TWIN, BicycleParams
from autocar_ctrl.stanley import StanleyPID
from autocar_loc.gt_ins import GtInsStub, Localization
from autocar_loc.pose import UNHEALTHY_HOLD_S, UNHEALTHY_SIM_M
from autocar_perc.source import Perception
from autocar_plan.simple_planner import CenterlinePlanner, closest_frenet
from autocar_supervisor.sm import DT, T_RECOVER_S, Brain, Health, St, Supervisor
from autocar_vehicle_interface.bicycle import BicycleBackend
from test_g1_closed_loop import DT_CTRL, LANELET_ID, PLAN_EVERY, SEEDS, V_CRUISE, _lane, _LatErr, _roadside_objects, _start

T_DROP = 10.0
T_END = 16.0


def _health(loc_ok: bool) -> Health:
    return Health(odd=True, loc=loc_ok, perc=True, brain=True, mcu=True, gate=True,
                  e_stop=False, collision=False, second_miss=False, cmd_age_s=0.0)


def _run(frac: float, off: float, drop_s: float) -> dict:
    lane = _lane()
    i0, st0 = _start(lane, frac, off)
    backend = BicycleBackend(st0, BicycleParams())

    def gt() -> tuple[float, float, float, float]:
        s = backend.state
        return (s.x, s.y, s.yaw, s.v)

    stub = GtInsStub(gt)
    loc = Localization(stub)
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
    rec = {"t": [], "state": [], "brain": [], "health": [], "pos_err": [], "lat": [], "gnss_ok": [],
           "v": [], "accel": [], "steer": [], "reason": []}
    injected = None
    for k in range(int(round(T_END / DT_CTRL))):
        t = k * DT_CTRL
        stamp = int(round(t * 1e9))
        if injected is None and t >= T_DROP - 1e-9:
            injected = stub.inject_loc_drop(drop_s, stamp)
            assert injected["ok"]
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
        rec["v"].append(st.v)
        rec["accel"].append(out.cmd.accel)
        rec["steer"].append(out.cmd.steer)
        rec["reason"].append(out.reason)
        rec["t"].append(t)
        rec["state"].append(state)
        rec["brain"].append(sup.brain)
        rec["health"].append(loc.health_m)
        rec["gnss_ok"].append(stub.gnss_ok)
        rec["pos_err"].append(math.hypot(pose.x - st.x, pose.y - st.y))
        rec["lat"].append(lat(st.x, st.y))
        backend.step(DT_CTRL)
    t = np.array(rec["t"])
    states = rec["state"]
    health = np.array(rec["health"])
    run_idx = [i for i, s in enumerate(states) if s is St.RUNNING]
    after = t >= T_DROP - 1e-9

    def first(pred, start: float):
        for i, ti in enumerate(t):
            if ti >= start - 1e-9 and pred(i):
                return float(ti)
        return None

    t_deg = first(lambda i: states[i] is St.DEGRADED, T_DROP)
    t_over = first(lambda i: health[i] > UNHEALTHY_SIM_M, T_DROP)
    t_under = first(lambda i: health[i] <= UNHEALTHY_SIM_M, t_over) if t_over is not None else None
    t_back = first(lambda i: states[i] is St.RUNNING, t_deg) if t_deg is not None else None
    win = after & (t <= T_DROP + drop_s + 1e-9)
    return {
        "t_running": float(t[run_idx[0]]) if run_idx else None,
        "pre_drop_states": {s for s, ti in zip(states, t) if 1.0 <= ti < T_DROP},
        "states_after": {s for s, a in zip(states, after) if a},
        "brains": set(rec["brain"]),
        "t_deg": t_deg, "t_over": t_over, "t_under": t_under, "t_back": t_back,
        "peak": float(health[after].max()),
        "max_pos_err": float(np.array(rec["pos_err"])[win].max()),
        "max_lat": float(np.array(rec["lat"])[after].max()),
        "final": states[-1],
        "rec": rec,
    }


def _fmt(x: float | None) -> str:
    return "none" if x is None else f"{x - T_DROP:.2f}"


def _print(tag: str, frac: float, off: float, drop_s: float, r: dict) -> None:
    print(f"G2 {tag} start={frac:.3f} off={off:+.2f} drop={drop_s:.2f} s  over_bar_at={_fmt(r['t_over'])} "
          f"degraded_at={_fmt(r['t_deg'])} under_bar_at={_fmt(r['t_under'])} running_again_at={_fmt(r['t_back'])} "
          f"(s after drop start)  peak={r['peak']:.3f} m  max_pos_err={r['max_pos_err']:.4f} m  "
          f"max_lat={r['max_lat']:.4f} m  states={sorted(s.name for s in r['states_after'])}")


def _common(r: dict) -> None:
    assert r["t_running"] is not None and r["pre_drop_states"] == {St.RUNNING}
    assert r["brains"] == {Brain.A}  # armed on the first tick, never switched
    assert r["states_after"] <= {St.RUNNING, St.DEGRADED}  # no HOLD, ESTOP, DISENGAGED
    assert r["final"] is St.RUNNING


@pytest.mark.parametrize("frac,off", SEEDS)
def test_g2_gnss_dropout_2s_degraded_not_flip(frac: float, off: float) -> None:
    r = _run(frac, off, 2.0)
    _print("drop", frac, off, 2.0, r)
    _common(r)
    assert r["t_deg"] is not None
    # Entered only after the health stayed over the bar for more than the hold.
    assert r["t_deg"] - r["t_over"] > UNHEALTHY_HOLD_S
    assert r["t_deg"] < T_DROP + 2.0
    # Health returns under the bar on the first fix after the drop, then the
    # supervisor needs T_recover / dt healthy ticks, the first one included,
    # before RUNNING (docs/design.md transition table).
    assert r["t_under"] == pytest.approx(T_DROP + 2.0, abs=DT_CTRL / 2)
    assert r["t_back"] - r["t_under"] == pytest.approx(T_RECOVER_S - DT, abs=DT / 2)
    assert r["peak"] > UNHEALTHY_SIM_M


@pytest.mark.parametrize("drop_s", (0.5, 0.9))
def test_g2_gnss_blip_under_1s_stays_running(drop_s: float) -> None:
    frac, off = SEEDS[0]
    r = _run(frac, off, drop_s)
    _print("blip", frac, off, drop_s, r)
    _common(r)
    assert r["t_deg"] is None
    assert r["states_after"] == {St.RUNNING}


# DEGRADED clamp on the plant (docs/design.md: gate owns it). The gate rewrites
# longitudinal to -a_hold while v > v_degrade, holds the last steer, reports
# reason=degraded, and goes back to passing the brain once RUNNING returns.
GP = GateParams(dt=DT_CTRL)
JERK_STEP = GP.jerk_max * GP.dt
# design.md settle window: "In practice the test runs 3.0 s at 50 Hz (150 ticks)
# and then asserts." The n = ceil((v0 - v_degrade) / (a_hold * dt)) tick count in
# the same paragraph is a lower bound ("only after") that ignores the jerk ramp.
SETTLE_BOUND_S = 3.0


def _clamp_view(r: dict) -> dict:
    rec = r["rec"]
    states = r["rec"]["state"]
    deg = [i for i, s in enumerate(states) if s is St.DEGRADED]
    i0, i1 = deg[0], deg[-1]
    assert deg == list(range(i0, i1 + 1))  # one contiguous DEGRADED stretch
    v = np.array(rec["v"])
    acc = np.array(rec["accel"])
    settle = next((i for i in range(i0, i1 + 1) if v[i] <= GP.v_degrade), None)
    return {
        "i0": i0, "i1": i1, "v": v, "acc": acc,
        "v0": float(v[i0]), "v_exit": float(v[i1]), "v_min": float(v[i0:i1 + 1].min()),
        "acc_min": float(acc[i0:i1 + 1].min()),
        "jerk_max": float(np.abs(np.diff(acc[i0 - 1:])).max()) / GP.dt,
        "t_settle": None if settle is None else (settle - i0) * DT_CTRL,
    }


def _jerk_aware_settle_s(v0: float) -> float:
    """Ideal time from 0 accel to v <= v_degrade with accel ramping to -a_hold at jerk_max."""
    dv = v0 - GP.v_degrade
    t_ramp = GP.a_hold / GP.jerk_max
    dv_ramp = GP.a_hold * GP.a_hold / (2.0 * GP.jerk_max)
    if dv <= dv_ramp:
        return math.sqrt(2.0 * dv / GP.jerk_max)
    return t_ramp + (dv - dv_ramp) / GP.a_hold


@pytest.mark.parametrize("seed,frac,off", [(n, f, o) for n, (f, o) in enumerate(SEEDS)])
def test_g2_degraded_clamp_slows_the_car(seed: int, frac: float, off: float) -> None:
    r = _run(frac, off, 2.0)
    rec = r["rec"]
    c = _clamp_view(r)
    i0, i1 = c["i0"], c["i1"]
    i_back = i1 + 1
    print(f"G2 clamp seed={seed} drop=2.00 s  degraded={rec['t'][i0] - T_DROP:.2f}..{rec['t'][i1] - T_DROP:.2f} s  "
          f"v_entry={c['v0']:.3f} v_min={c['v_min']:.3f} v_exit={c['v_exit']:.3f} m/s  "
          f"accel_min={c['acc_min']:.3f} m/s^2  max_jerk={c['jerk_max']:.3f} m/s^3  "
          f"max_lat={r['max_lat']:.4f} m  reason_after_running={rec['reason'][i_back]}")
    _common(r)
    # (a) the gate reports the clamp for every DEGRADED tick and holds the last steer
    assert all(rec["reason"][i] == "degraded" for i in range(i0, i1 + 1))
    assert all(rec["steer"][i] == rec["steer"][i0 - 1] for i in range(i0, i1 + 1))
    # (b) commanded accel goes negative, never past -a_hold, through the jerk clip
    assert c["acc_min"] < 0.0
    assert c["acc_min"] >= -GP.a_hold - 1e-9
    assert c["jerk_max"] * GP.dt <= JERK_STEP + 1e-9
    # (c) the car slows while DEGRADED
    assert c["v_min"] < c["v0"]
    assert c["v_exit"] < c["v0"]
    # (d) design catalog: loc_drop excursion < 1.0 m on the true state
    assert r["max_lat"] < 1.0
    # (e) the clamp clears on the first RUNNING tick
    assert rec["state"][i_back] is St.RUNNING
    assert rec["reason"][i_back] != "degraded"
    assert all(rec["reason"][i] in {"ok", "clip"} for i in range(i_back, len(rec["t"])))


def test_g2_long_drop_settles_to_v_degrade() -> None:
    frac, off = SEEDS[0]
    r = _run(frac, off, 4.0)
    c = _clamp_view(r)
    n_lower = math.ceil((c["v0"] - GP.v_degrade) / (GP.a_hold * GP.dt))
    print(f"G2 settle drop=4.00 s  v_entry={c['v0']:.3f} m/s  settle={c['t_settle']} s  "
          f"design_lower={n_lower * GP.dt:.2f} s  jerk_aware_ideal={_jerk_aware_settle_s(c['v0']):.2f} s  "
          f"bound={SETTLE_BOUND_S:.1f} s  v_min={c['v_min']:.3f} m/s  degraded_for={(c['i1'] - c['i0'] + 1) * DT_CTRL:.2f} s  "
          f"max_lat={r['max_lat']:.4f} m")
    _common(r)
    assert c["t_settle"] is not None
    assert n_lower * GP.dt - 1e-9 <= c["t_settle"] <= SETTLE_BOUND_S
    assert c["v"][c["i0"] + int(round(SETTLE_BOUND_S / DT_CTRL)):c["i1"] + 1].max() <= GP.v_degrade
    assert r["max_lat"] < 1.0
