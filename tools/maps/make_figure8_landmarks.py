"""Generate the synthetic 2D landmark map for maps/figure8_v0 (PR-06b).

Round poles on both sides of the figure-8 centerline plus a rectangular
boundary wall around the track. The simulated 2D lidar raycasts this map
and the NDT scan matcher uses points sampled from it as its target cloud.
It stands in for the offline NDT PCD until a recorded map exists.

Poles sit POLE_OFFSET_M left and right of the centerline every
POLE_SPACING_M of arc length. A pole is dropped if it is closer than
POLE_CLEARANCE_M to any centerline sample (this keeps the crossing clear)
or closer than POLE_MIN_GAP_M to a pole already placed. Pure Python, no
randomness, so the output is deterministic.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from make_figure8 import OUT_PATH as CENTERLINE_PATH
from make_figure8 import load_centerline

POLE_OFFSET_M = 5.0
POLE_SPACING_M = 7.0
POLE_RADIUS_M = 0.15
POLE_CLEARANCE_M = 3.5
POLE_MIN_GAP_M = 3.0
WALL_MARGIN_M = 8.0

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "maps" / "figure8_v0" / "landmarks.csv"
HEADER = ("kind", "x0", "y0", "x1", "y1", "radius")


def generate_landmarks(centerline=None) -> list[tuple[str, float, float, float, float, float]]:
    pts = centerline if centerline is not None else load_centerline(CENTERLINE_PATH)
    xy = [(p[0], p[1]) for p in pts]
    rows: list[tuple[str, float, float, float, float, float]] = []
    poles: list[tuple[float, float]] = []
    s, next_s = 0.0, 0.0
    for i, (x, y, yaw, _k) in enumerate(pts):
        if i > 0:
            s += math.hypot(x - pts[i - 1][0], y - pts[i - 1][1])
        if s + 1e-9 < next_s:
            continue
        next_s += POLE_SPACING_M
        for side in (1.0, -1.0):
            px = x - side * POLE_OFFSET_M * math.sin(yaw)
            py = y + side * POLE_OFFSET_M * math.cos(yaw)
            if min(math.hypot(px - cx, py - cy) for cx, cy in xy) < POLE_CLEARANCE_M:
                continue
            if any(math.hypot(px - qx, py - qy) < POLE_MIN_GAP_M for qx, qy in poles):
                continue
            poles.append((px, py))
            rows.append(("pole", px, py, px, py, POLE_RADIUS_M))
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    x0, x1 = min(xs) - WALL_MARGIN_M, max(xs) + WALL_MARGIN_M
    y0, y1 = min(ys) - WALL_MARGIN_M, max(ys) + WALL_MARGIN_M
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        rows.append(("wall", a[0], a[1], b[0], b[1], 0.0))
    return rows


def write_landmarks(path: Path = OUT_PATH) -> Path:
    rows = generate_landmarks()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for kind, a, b, c, d, r in rows:
            w.writerow([kind, f"{a:.6f}", f"{b:.6f}", f"{c:.6f}", f"{d:.6f}", f"{r:.3f}"])
    return path


def main() -> None:
    out = write_landmarks()
    rows = generate_landmarks()
    n_pole = sum(r[0] == "pole" for r in rows)
    print(f"wrote {out} ({n_pole} poles, {len(rows) - n_pole} walls)")


if __name__ == "__main__":
    main()
