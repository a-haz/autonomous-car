"""Generate the figure-8 centerline under maps/figure8_v0 (PR-03a).

Two G2 clothoid loops with zero curvature at the crossing. Each loop is an
entry clothoid, a transition to a slightly wider far arc (chosen so the
loop is symmetric about the y-axis and therefore closes), and the mirror
exit. Pure Python, no ROS. Output is deterministic.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

# Entry radius is well above L/tan(delta_max) ~ 3.94 m, so a 5 m/s cruise
# stays inside delta_max and delta_dot_max. The far-arc radius is solved
# in solve_r_far so each loop meets the +y axis at heading pi.
R_ENTRY = 20.0
CLOTHOID_LEN = 16.0
TRANSITION_LEN = 8.0
SAMPLE_DS = 0.25
INTEGRATE_DS = 0.001

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "maps" / "figure8_v0" / "centerline.csv"
HEADER = ("x", "y", "yaw", "curvature")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _arc_step(x: float, y: float, yaw: float, ds: float, kappa: float) -> tuple[float, float, float]:
    if abs(kappa) < 1e-12:
        return x + ds * math.cos(yaw), y + ds * math.sin(yaw), yaw
    dpsi = kappa * ds
    r = 1.0 / kappa
    x2 = x + r * (math.sin(yaw + dpsi) - math.sin(yaw))
    y2 = y - r * (math.cos(yaw + dpsi) - math.cos(yaw))
    return x2, y2, yaw + dpsi


def _advance_state(
    x: float, y: float, yaw: float, k0: float, k1: float, length: float
) -> tuple[float, float, float]:
    if length <= 0.0:
        return x, y, yaw
    n = max(1, int(round(length / INTEGRATE_DS)))
    ds = length / n
    for i in range(1, n + 1):
        kappa = k0 + (k1 - k0) * ((i - 0.5) * ds / length)
        x, y, yaw = _arc_step(x, y, yaw, ds, kappa)
    return x, y, yaw


def _heading_before_arc(r_far: float) -> float:
    k1 = 1.0 / R_ENTRY
    kfar = 1.0 / r_far
    return CLOTHOID_LEN / (2.0 * R_ENTRY) + 0.5 * (k1 + kfar) * TRANSITION_LEN


def _x_at_far_axis(r_far: float) -> float:
    k1 = 1.0 / R_ENTRY
    kfar = 1.0 / r_far
    theta = _heading_before_arc(r_far)
    x, y, yaw = _advance_state(0.0, 0.0, 0.0, 0.0, k1, CLOTHOID_LEN)
    x, y, yaw = _advance_state(x, y, yaw, k1, kfar, TRANSITION_LEN)
    x, y, yaw = _advance_state(x, y, yaw, kfar, kfar, (math.pi - theta) * r_far)
    return x


def solve_r_far() -> float:
    """Far-arc radius that puts the top of the left loop on the +y axis."""
    lo, hi = R_ENTRY, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _x_at_far_axis(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _advance(
    points: list[tuple[float, float, float, float]],
    x: float,
    y: float,
    yaw: float,
    k0: float,
    k1: float,
    length: float,
    s: float,
    next_s: float,
) -> tuple[float, float, float, float, float]:
    if length <= 0.0:
        return x, y, yaw, s, next_s
    n = max(1, int(round(length / INTEGRATE_DS)))
    ds = length / n
    for i in range(1, n + 1):
        kappa = k0 + (k1 - k0) * ((i - 0.5) * ds / length)
        x, y, yaw = _arc_step(x, y, yaw, ds, kappa)
        s += ds
        if s + 1e-12 >= next_s:
            k_end = k0 + (k1 - k0) * (i * ds / length)
            points.append((x, y, wrap(yaw), k_end))
            next_s += SAMPLE_DS
    return x, y, yaw, s, next_s


def _loop(
    points: list[tuple[float, float, float, float]],
    x: float,
    y: float,
    yaw: float,
    sign: float,
    r_far: float,
    s: float,
    next_s: float,
) -> tuple[float, float, float, float, float]:
    k1 = sign / R_ENTRY
    kfar = sign / r_far
    theta = _heading_before_arc(r_far)
    arc_len = (math.pi - theta) * r_far
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, 0.0, k1, CLOTHOID_LEN, s, next_s)
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, k1, kfar, TRANSITION_LEN, s, next_s)
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, kfar, kfar, arc_len, s, next_s)
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, kfar, kfar, arc_len, s, next_s)
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, kfar, k1, TRANSITION_LEN, s, next_s)
    x, y, yaw, s, next_s = _advance(points, x, y, yaw, k1, 0.0, CLOTHOID_LEN, s, next_s)
    return x, y, yaw, s, next_s


def generate_centerline() -> list[tuple[float, float, float, float]]:
    """Closed figure-8: left (CCW) loop then right (CW) loop, start at the origin heading +x."""
    r_far = solve_r_far()
    points: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)]
    x, y, yaw, s, next_s = 0.0, 0.0, 0.0, 0.0, SAMPLE_DS
    x, y, yaw, s, next_s = _loop(points, x, y, yaw, 1.0, r_far, s, next_s)
    if math.hypot(x, y) < 1e-6:
        x, y, yaw = 0.0, 0.0, 0.0
        if math.hypot(points[-1][0], points[-1][1]) > 1e-12:
            points.append((0.0, 0.0, 0.0, 0.0))
        next_s = s + SAMPLE_DS
    x, y, yaw, s, next_s = _loop(points, x, y, yaw, -1.0, r_far, s, next_s)
    end = (x, y, wrap(yaw), 0.0)
    if not points or math.hypot(points[-1][0] - end[0], points[-1][1] - end[1]) > 1e-9:
        points.append(end)
    if math.hypot(points[-1][0], points[-1][1]) < 1e-6 and abs(wrap(points[-1][2])) < 1e-6:
        points[-1] = (0.0, 0.0, 0.0, 0.0)
    return points


def write_centerline(path: Path = OUT_PATH) -> Path:
    points = generate_centerline()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(HEADER)
        for x, y, yaw, kappa in points:
            writer.writerow([f"{x:.12f}", f"{y:.12f}", f"{yaw:.12f}", f"{kappa:.12f}"])
    return path


def load_centerline(path: Path = OUT_PATH) -> list[tuple[float, float, float, float]]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        if list(rows.fieldnames or []) != list(HEADER):
            raise ValueError(f"unexpected header: {rows.fieldnames}")
        return [
            (float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["curvature"]))
            for r in rows
        ]


def main() -> None:
    out = write_centerline()
    pts = load_centerline(out)
    print(f"wrote {out} ({len(pts)} samples)")


if __name__ == "__main__":
    main()
