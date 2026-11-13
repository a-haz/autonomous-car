"""G0 open-loop figure-8 gate. Canned VehicleCommand, no brain. No ROS, no Isaac."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from autocar_command_gate.gate import Cmd, CommandGate, GateParams
from autocar_ctrl.bicycle import DELTA_MAX, L_TWIN, V_MAX, BicycleParams, BicycleState, clamp, wrap
from autocar_mcu_bridge.frame import MISS_LIMIT, PERIOD_S, TYPE_HOST_HB, pack
from autocar_mcu_bridge.loopback import LoopbackMcu
from autocar_supervisor.sm import Brain, Health, St, Supervisor
from autocar_vehicle_interface.bicycle import BicycleBackend

import make_figure8

ROOT = Path(__file__).resolve().parents[3]
CENTERLINE = ROOT / "maps" / "figure8_v0" / "centerline.csv"
DT = 0.02
V_CRUISE = 5.0
# G0 bar from docs/design.md Verification Gates.
LAT_ERR_BAR_M = 1.0
# 3 missed HOST_HB frames (docs/design.md MCU heartbeat).
LATCH_BOUND_S = MISS_LIMIT * PERIOD_S
GATE_P = GateParams()
# Plant must be able to apply the gate's emergency brake (a_min_brake_rear).
PLANT_P = BicycleParams(
    L=L_TWIN,
    delta_max=DELTA_MAX,
    v_max=V_MAX,
    a_min=-GATE_P.a_min_brake_rear,
    a_max=GATE_P.a_max_accel,
)


def _s_of(pts: list[tuple[float, float, float, float]]) -> list[float]:
    s = [0.0]
    for i in range(1, len(pts)):
        s.append(s[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return s


def _interp_at(
    pts: list[tuple[float, float, float, float]], s: list[float], s_q: float
) -> tuple[float, float, float, float]:
    total = s[-1]
    s_q = s_q % total if total > 0.0 else 0.0
    lo, hi = 0, len(s) - 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if s[mid] <= s_q:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    span = s[i + 1] - s[i]
    t = 0.0 if span <= 1e-12 else (s_q - s[i]) / span
    x = pts[i][0] + t * (pts[i + 1][0] - pts[i][0])
    y = pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
    yaw = pts[i][2] + t * wrap(pts[i + 1][2] - pts[i][2])
    kappa = pts[i][3] + t * (pts[i + 1][3] - pts[i][3])
    return x, y, wrap(yaw), kappa


def _closest_in_window(
    pts: list[tuple[float, float, float, float]],
    s_tbl: list[float],
    s_center: float,
    window: float,
    px: float,
    py: float,
    ds: float = 0.05,
) -> float:
    best_s, best_d = s_center, 1e9
    s_try = s_center - window
    while s_try <= s_center + window:
        tx, ty, _, _ = _interp_at(pts, s_tbl, s_try)
        d = math.hypot(px - tx, py - ty)
        if d < best_d:
            best_d, best_s = d, s_try
        s_try += ds
    return best_s


def _signed_cte(px: float, py: float, tx: float, ty: float, tyaw: float) -> float:
    """Positive when the path is to the left of the point (Stanley convention)."""
    return -math.sin(tyaw) * (tx - px) + math.cos(tyaw) * (ty - py)


def _canned_commands(
    pts: list[tuple[float, float, float, float]], v: float, dt: float, n_steps: int
) -> list[tuple[float, float]]:
    """Open-loop steer from planned s = v*t and centerline curvature. No pose feedback."""
    s_tbl = _s_of(pts)
    cmds: list[tuple[float, float]] = []
    for i in range(n_steps):
        _, _, _, kappa = _interp_at(pts, s_tbl, v * i * dt)
        delta = clamp(math.atan(kappa * L_TWIN), -DELTA_MAX, DELTA_MAX)
        cmds.append((delta, 0.0))
    return cmds


def _health(now_s: float, mcu_ok: bool, cmd_stamp_s: float) -> Health:
    return Health(
        odd=True,
        loc=True,
        perc=True,
        brain=True,
        mcu=mcu_ok,
        gate=True,
        e_stop=False,
        collision=False,
        second_miss=False,
        cmd_age_s=now_s - cmd_stamp_s,
    )


def _arm(sup: Supervisor, mcu: LoopbackMcu, now_s: float) -> float:
    """Reach RUNNING with heartbeats only. The plant does not move."""
    hb = pack(TYPE_HOST_HB, 0, int(now_s * 1000.0) & 0xFFFFFFFF, 0, 0)
    mcu.on_host_frame(now_s, hb)
    mcu.tick(now_s)
    sup.request_arm(Brain.A)
    for _ in range(2):
        now_s += DT
        hb = pack(TYPE_HOST_HB, 0, int(now_s * 1000.0) & 0xFFFFFFFF, 0, 0)
        mcu.on_host_frame(now_s, hb)
        mcu.tick(now_s)
        assert sup.step(_health(now_s, mcu.mcu_ok, now_s)) in (St.ARMING, St.RUNNING)
    assert sup.state is St.RUNNING
    assert mcu.mcu_ok
    return now_s


def _run_g0(inject_miss_at: int | None = None) -> dict[str, float | int | str | bool]:
    pts = make_figure8.load_centerline(CENTERLINE)
    s_tbl = _s_of(pts)
    length = s_tbl[-1]
    n_steps = int(length / V_CRUISE / DT) + 1
    canned = _canned_commands(pts, V_CRUISE, DT, n_steps)
    assert len(canned) == n_steps
    assert V_CRUISE <= V_MAX
    assert all(abs(delta) <= DELTA_MAX + 1e-12 and accel == 0.0 for delta, accel in canned)

    backend = BicycleBackend(BicycleState(0.0, 0.0, 0.0, V_CRUISE), PLANT_P)
    mcu = LoopbackMcu()
    sup = Supervisor()
    gate = CommandGate(GATE_P)
    now = _arm(sup, mcu, 0.0)

    rear_abs: list[float] = []
    front_abs: list[float] = []
    s_rear = 0.0
    s_front = 0.0
    empty_occ = np.zeros((0, 2), dtype=np.float32)
    t_last_hb = now
    t_latch = None
    t_left_running = None
    t_v0 = None
    seq = 0
    stop_horizon = n_steps + int(5.0 / DT)

    for i in range(stop_horizon):
        now += DT
        sending_hb = inject_miss_at is None or i < inject_miss_at
        if sending_hb:
            hb = pack(TYPE_HOST_HB, 0, int(now * 1000.0) & 0xFFFFFFFF, 0, 0)
            mcu.on_host_frame(now, hb)
            t_last_hb = now
        mcu.tick(now)
        if t_latch is None and mcu.latched:
            t_latch = now

        cmd_i = min(i, len(canned) - 1)
        delta, accel = canned[cmd_i]
        incoming = Cmd(now, delta, accel, False, seq, "canned")
        seq += 1
        st_before = sup.state
        sup.step(_health(now, mcu.mcu_ok, incoming.stamp_s))
        if t_left_running is None and st_before is St.RUNNING and sup.state is not St.RUNNING:
            t_left_running = now

        st = backend.state
        out = gate.tick(
            now,
            incoming,
            sup.state.name,
            None,
            empty_occ,
            (st.x, st.y, st.yaw),
            st.v,
            collision=False,
        )
        applied_delta, applied_accel = out.cmd.steer, out.cmd.accel
        if mcu.latched:
            applied_accel = min(applied_accel, -GATE_P.a_min_brake_rear)
        backend.set_command(applied_delta, applied_accel)
        backend.step(DT)
        st = backend.state

        if i < n_steps:
            s_rear = _closest_in_window(pts, s_tbl, s_rear, 8.0, st.x, st.y)
            rx, ry, ryaw, _ = _interp_at(pts, s_tbl, s_rear)
            rear_abs.append(abs(_signed_cte(st.x, st.y, rx, ry, ryaw)))
            fx = st.x + L_TWIN * math.cos(st.yaw)
            fy = st.y + L_TWIN * math.sin(st.yaw)
            s_front = _closest_in_window(pts, s_tbl, s_front, 8.0, fx, fy)
            ftx, fty, ftyaw, _ = _interp_at(pts, s_tbl, s_front)
            front_abs.append(abs(_signed_cte(fx, fy, ftx, fty, ftyaw)))

        if t_v0 is None and inject_miss_at is not None and i >= inject_miss_at and st.v <= 1e-12:
            t_v0 = now
            break
        if inject_miss_at is None and i + 1 >= n_steps:
            break

    max_rear = max(rear_abs) if rear_abs else float("nan")
    max_front = max(front_abs) if front_abs else float("nan")
    rms_rear = math.sqrt(sum(e * e for e in rear_abs) / len(rear_abs)) if rear_abs else float("nan")
    rms_front = math.sqrt(sum(e * e for e in front_abs) / len(front_abs)) if front_abs else float("nan")
    return {
        "n_steps": len(rear_abs),
        "max_rear": max_rear,
        "rms_rear": rms_rear,
        "max_front": max_front,
        "rms_front": rms_front,
        "final_v": backend.state.v,
        "mcu_ok": mcu.mcu_ok,
        "sup_state": sup.state.name,
        "t_last_hb": t_last_hb,
        "t_latch": t_latch if t_latch is not None else float("nan"),
        "t_left_running": t_left_running if t_left_running is not None else float("nan"),
        "t_v0": t_v0 if t_v0 is not None else float("nan"),
        "latched": mcu.latched,
    }


def test_g0_open_loop_tracks_figure8() -> None:
    r = _run_g0()
    print(
        f"G0 open-loop v={V_CRUISE} m/s  max_rear={r['max_rear']:.4f} rms_rear={r['rms_rear']:.4f}  "
        f"max_front={r['max_front']:.4f} rms_front={r['rms_front']:.4f}"
    )
    assert r["mcu_ok"] is True
    assert r["sup_state"] == "RUNNING"
    assert r["final_v"] == V_CRUISE
    assert r["max_rear"] < LAT_ERR_BAR_M
    assert r["max_front"] < LAT_ERR_BAR_M


def test_g0_mcu_miss_stops_the_car() -> None:
    pts = make_figure8.load_centerline(CENTERLINE)
    n_steps = int(_s_of(pts)[-1] / V_CRUISE / DT) + 1
    inject_at = n_steps // 2
    r = _run_g0(inject_miss_at=inject_at)
    latch_s = r["t_latch"] - r["t_last_hb"]
    left_s = r["t_left_running"] - r["t_last_hb"]
    stop_s = r["t_v0"] - r["t_last_hb"]
    # After the inhibit opens, the gate emergency-brakes at a_min_brake_rear.
    stop_bound_s = LATCH_BOUND_S + V_CRUISE / GATE_P.a_min_brake_rear + DT
    print(
        f"G0 MCU miss  latch={latch_s:.4f}s  left_RUNNING={left_s:.4f}s  "
        f"stop={stop_s:.4f}s  bound={stop_bound_s:.4f}s  state={r['sup_state']}"
    )
    assert r["latched"] is True
    assert r["mcu_ok"] is False
    assert r["sup_state"] == "ESTOP"
    assert math.isclose(latch_s, LATCH_BOUND_S, abs_tol=1e-12)
    assert math.isclose(left_s, LATCH_BOUND_S, abs_tol=1e-12)
    assert r["final_v"] == 0.0
    assert math.isfinite(r["t_v0"])
    assert stop_s <= stop_bound_s
