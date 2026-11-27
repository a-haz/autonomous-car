"""PR-06b per-match NDT covariance (Laplace, NdtResult.cov_hessian).

Bounds stated before measuring:
- Corridor (two parallel walls, no other structure) on the built-map path:
  for every scan, the Hessian covariance variance along the walls is at
  least RATIO_MIN = 4 times the variance across them. The reasoning: along
  a wall a 1.0 m NDT cell spreads its points over about 1/12 m^2, across it
  the final pass floors the variance at 0.1**2 = 0.01 m^2, so the plain
  Laplace estimate should give a ratio near 8. The matcher must still
  report not converged on these scans.
- Bad initial guess (4 m, 30 deg) on the built map, seeds 1 to 3, 10 poses
  each: a match is either not converged or within WRONG_POSE_M of the truth.
  Its Hessian covariance must not make it look tighter than the truth
  either: printed only.
"""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest
from autocar_loc.ndt2d import Ndt2D
from autocar_vehicle_interface.lidar2d import LandmarkMap, SimLidar2D

import build_ndt_map
from test_ndt2d import (BAD_GUESS_POS_M, BAD_GUESS_YAW_DEG, GUESS_POS_M, GUESS_YAW_DEG, MAP_DIR, SEEDS,
                        WRONG_POSE_M, _guess, _poses, _wrap)

RATIO_MIN = 4.0
CORRIDOR = LandmarkMap(np.zeros((0, 3)), np.array([[-100.0, 5.0, 300.0, 5.0], [-100.0, -5.0, 300.0, -5.0]]))
CORRIDOR_BUILD_SEED = build_ndt_map.BUILD_SEED


def corridor_cell_map(path) -> Ndt2D:
    """Built-map path for the corridor: scans from a straight drive, voxelized and celled
    with the same functions as tools/maps/build_ndt_map.py."""
    lid = SimLidar2D(CORRIDOR, seed=CORRIDOR_BUILD_SEED)
    pts = []
    for x in np.arange(-60.0, 260.0, 0.5):
        p = lid.points(lid.scan(x, 0.0, 0.0))
        pts.append(p + np.array([x, 0.0]))
    vox = build_ndt_map.voxelize(np.vstack(pts))
    rows = [r for c in build_ndt_map.CELL_SIZES_M for r in build_ndt_map.cells(vox, c)]
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(build_ndt_map.HEADER)
        w.writerows(rows)
    return Ndt2D.from_cell_map(path)


def test_corridor_hessian_cov_is_long_along_the_walls(tmp_path):
    ndt = corridor_cell_map(tmp_path / "corridor.csv")
    lid = SimLidar2D(CORRIDOR, seed=21)
    rng = np.random.default_rng(21)
    ratios = []
    for x in (0.0, 13.0, 37.0, 71.0, 120.0):
        pts = lid.points(lid.scan(x, 0.0, 0.0))
        res = ndt.align(pts, _guess(rng, x, 0.0, 0.0, GUESS_POS_M, GUESS_YAW_DEG))
        c = res.cov_hessian
        ratios.append(c[0, 0] / c[1, 1])
        print(f"\ncorridor x={x:.0f}: converged={res.converged} reason={res.reason} along_err={res.x - x:+.3f} m "
              f"hessian sigma along={math.sqrt(c[0, 0]):.4f} across={math.sqrt(c[1, 1]):.4f} m "
              f"ratio={c[0, 0] / c[1, 1]:.2f} (bound >= {RATIO_MIN})  fixed sigma along={math.sqrt(res.cov[0, 0]):.3g} m")
        assert not res.converged
    assert min(ratios) >= RATIO_MIN


@pytest.mark.parametrize("seed", SEEDS)
def test_built_map_bad_guess_never_confident_and_wrong(seed):
    import make_figure8
    lmap = LandmarkMap.from_csv(MAP_DIR / "landmarks.csv")
    ndt = Ndt2D.from_cell_map(MAP_DIR / "ndt_cells.csv")
    cl = np.array(make_figure8.load_centerline(MAP_DIR / "centerline.csv"))
    rng, poses = _poses(100 + seed, cl, 10)
    lidar = SimLidar2D(lmap, seed=100 + seed)
    n_conv, errs = 0, []
    for x, y, yaw in poses:
        pts = lidar.points(lidar.scan(x, y, yaw))
        res = ndt.align(pts, _guess(rng, x, y, yaw, BAD_GUESS_POS_M, BAD_GUESS_YAW_DEG))
        e = math.hypot(res.x - x, res.y - y)
        if res.converged:
            n_conv += 1
            errs.append(e)
            assert e <= WRONG_POSE_M, f"confident wrong pose {e:.2f} m"
    print(f"\nbuilt map bad guess seed {seed}: converged {n_conv}/10, errors of converged "
          f"{[round(v, 3) for v in errs]} m (bound {WRONG_POSE_M} m)")
