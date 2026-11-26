"""Build the 2D NDT cell map for maps/figure8_v0 from a recorded sim lap (PR-06b).

Stand-in for the offline NDT mapper that will produce ndt.pcd for ndt_omp.
One open-loop lap of the figure-8 in the bicycle twin: constant
V_LAP_MPS, steering fed forward from the centerline curvature at the
distance driven (no feedback). Every 1 / SCAN_HZ the seeded SimLidar2D
(lidar2d.py, default parameters) scans maps/figure8_v0/landmarks.csv from
the ground-truth pose, and the returns are put in the map frame with that
pose. The stacked cloud is voxelized at VOXEL_M = 0.2 m (one centroid per
occupied voxel), then cut into square NDT cells of CELL_SIZES_M (the cell
sizes the Ndt2D passes use). Every cell with at least MIN_CELL_VOXELS (1)
voxel centroid is kept; see autocar_loc.ndt2d for why.

Output format, maps/figure8_v0/ndt_cells.csv, one row per cell:

    cell_m   cell size in m
    ix, iy   integer cell index, the cell covers
             [ix * cell_m, (ix + 1) * cell_m) x [iy * cell_m, (iy + 1) * cell_m) in map
    n        voxel centroids in the cell
    mean_x, mean_y            mean of those centroids, m
    cov_xx, cov_xy, cov_yy    their covariance (divided by n), m^2

The build is deterministic for a fixed BUILD_SEED. It uses lidar seed
BUILD_SEED = 7, which no test scan stream uses, so the tests do not match
scans against the noise the map was built from.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for sub in ("brains/modular/autocar_ctrl", "brains/modular/autocar_loc",
            "vehicle_interface/autocar_vehicle_interface", "tools/maps"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import numpy as np  # noqa: E402
from autocar_ctrl.bicycle import BicycleParams, BicycleState  # noqa: E402
from autocar_loc.ndt2d import MIN_CELL_VOXELS, PASSES  # noqa: E402
from autocar_vehicle_interface.bicycle import BicycleBackend  # noqa: E402
from autocar_vehicle_interface.lidar2d import LandmarkMap, SimLidar2D  # noqa: E402

from make_figure8 import load_centerline  # noqa: E402

BUILD_SEED = 7
V_LAP_MPS = 5.0
DT = 0.02
SCAN_HZ = 10.0
VOXEL_M = 0.2
CELL_SIZES_M = tuple(sorted({c for c, _ in PASSES}, reverse=True))
MAP_DIR = ROOT / "maps" / "figure8_v0"
OUT_PATH = MAP_DIR / "ndt_cells.csv"
HEADER = ("cell_m", "ix", "iy", "n", "mean_x", "mean_y", "cov_xx", "cov_xy", "cov_yy")


def record_lap(seed: int = BUILD_SEED) -> np.ndarray:
    """Stacked lidar returns in map frame over one open-loop lap, (N, 2)."""
    cl = np.array(load_centerline(MAP_DIR / "centerline.csv"))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1])))])
    lap = float(s[-1])
    params = BicycleParams()
    be = BicycleBackend(BicycleState(cl[0, 0], cl[0, 1], cl[0, 2], V_LAP_MPS), params)
    be.attach_lidar(SimLidar2D(LandmarkMap.from_csv(MAP_DIR / "landmarks.csv"), seed=seed))
    every = int(round(1.0 / (SCAN_HZ * DT)))
    out, dist, k = [], 0.0, 0
    while dist < lap:
        if k % every == 0:
            st = be.state
            p = be.lidar.points(be.lidar_scan())
            c, sn = math.cos(st.yaw), math.sin(st.yaw)
            out.append(np.column_stack([c * p[:, 0] - sn * p[:, 1] + st.x, sn * p[:, 0] + c * p[:, 1] + st.y]))
        kappa = float(np.interp(dist, s, cl[:, 3]))
        be.set_command(math.atan(params.L * kappa), 0.0)
        before = be.state
        be.step(DT)
        dist += math.hypot(be.state.x - before.x, be.state.y - before.y)
        k += 1
    return np.vstack(out)


def voxelize(pts: np.ndarray, voxel: float = VOXEL_M) -> np.ndarray:
    """One centroid per occupied voxel, sorted by voxel index."""
    ij = np.floor(pts / voxel).astype(np.int64)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = np.bincount(inv)
    return np.column_stack([np.bincount(inv, pts[:, 0]) / n, np.bincount(inv, pts[:, 1]) / n])


def cells(vox: np.ndarray, cell: float) -> list[tuple]:
    ij = np.floor(vox / cell).astype(np.int64)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    rows = []
    for u, (ix, iy) in enumerate(uniq):
        q = vox[inv == u]
        if len(q) < MIN_CELL_VOXELS:
            continue
        mu = q.mean(axis=0)
        cv = np.cov((q - mu).T, bias=True)
        rows.append((cell, int(ix), int(iy), len(q), mu[0], mu[1], cv[0, 0], cv[0, 1], cv[1, 1]))
    return rows


def build(seed: int = BUILD_SEED) -> list[tuple]:
    vox = voxelize(record_lap(seed))
    return [r for c in CELL_SIZES_M for r in cells(vox, c)]


def write_cell_map(path: Path = OUT_PATH, seed: int = BUILD_SEED) -> Path:
    rows = build(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for c, ix, iy, n, mx, my, cxx, cxy, cyy in rows:
            w.writerow([f"{c:.1f}", ix, iy, n, f"{mx:.6f}", f"{my:.6f}", f"{cxx:.6e}", f"{cxy:.6e}", f"{cyy:.6e}"])
    return path


def main() -> None:
    out = write_cell_map()
    rows = build()
    for c in CELL_SIZES_M:
        print(f"{c} m cells: {sum(r[0] == c for r in rows)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
