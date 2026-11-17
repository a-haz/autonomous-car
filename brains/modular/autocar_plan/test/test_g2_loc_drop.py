"""G2 'GNSS dropout 2 s gives DEGRADED, not flip' on the bicycle loop.

Same loop as G1 (autocar_loc GT/INS stub -> centerline planner -> StanleyPID
-> bicycle), with the supervisor stepped every 20 ms on Health(loc =
Localization.healthy, everything else healthy). /scenario/inject/loc_drop
mutes GNSS, the stub dead reckons, and its covariance grows by the bound
documented in autocar_loc.gt_ins. No gate in this loop, so the DEGRADED
speed clamp is not applied; the excursion printed here is without it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
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
    lights = {"lights": []}
    traj = None
    rec = {"t": [], "state": [], "brain": [], "health": [], "pos_err": [], "lat": [], "gnss_ok": []}
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
        backend.set_command(ctrl.lateral(e, th, st.v), ctrl.longitudinal(st.v, vref, DT_CTRL))
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
