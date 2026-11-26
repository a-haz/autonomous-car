"""PR-06b offline map step: the NDT cell map built from a recorded sim lap
(tools/maps/build_ndt_map.py, maps/figure8_v0/ndt_cells.csv) as the Ndt2D
target, instead of points sampled from landmarks.csv.

Bounds stated before measuring:
- Map quality: at least MAP_SHARE_MIN = 95 % of the 1.0 m cell means (the
  cells the two fine passes use) lie within MAP_DIST_M = 0.10 m of a true
  pole (as a solid disc) or wall segment in landmarks.csv. The 2.0 m cells
  are printed only.
- Matching along the figure-8 with the built map: the same bounds as
  test_ndt2d (converged rate, position and heading error, no confident
  wrong pose), same seeds, poses and guesses. Runtime per match is printed
  against the 30 ms budget and asserted only against the lenient CI bar.
- The build seed is not any lidar seed used by a test.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
from autocar_loc.ndt2d import Ndt2D
from autocar_vehicle_interface.lidar2d import LandmarkMap, SimLidar2D

import build_ndt_map
from test_ndt2d import (BUDGET_MS, GUESS_POS_M, GUESS_YAW_DEG, MAP_DIR, MIN_CONVERGED_RATE, N_POSES,
                        POS_MAX_BAR_M, POS_MEAN_BAR_M, RUNTIME_CI_BAR_MS, SEEDS, WRONG_POSE_M,
                        YAW_MAX_BAR_DEG, YAW_MEAN_BAR_DEG, _guess, _poses, _wrap)

CELL_MAP = MAP_DIR / "ndt_cells.csv"
MAP_DIST_M = 0.10
MAP_SHARE_MIN = 0.95
TEST_LIDAR_SEEDS = set(SEEDS) | {100 + s for s in SEEDS} | {11} | {1000, 1001, 1002}


@pytest.fixture(scope="module")
def lmap() -> LandmarkMap:
    return LandmarkMap.from_csv(MAP_DIR / "landmarks.csv")


@pytest.fixture(scope="module")
def built() -> Ndt2D:
    return Ndt2D.from_cell_map(CELL_MAP)


@pytest.fixture(scope="module")
def centerline() -> np.ndarray:
    import make_figure8
    return np.array(make_figure8.load_centerline(MAP_DIR / "centerline.csv"))


def test_cell_map_checked_in_matches_build(tmp_path):
    assert build_ndt_map.BUILD_SEED not in TEST_LIDAR_SEEDS
    out = build_ndt_map.write_cell_map(tmp_path / "ndt_cells.csv")
    assert out.read_text() == CELL_MAP.read_text()


def _dist_to_map(p: np.ndarray, lmap: LandmarkMap) -> np.ndarray:
    d = np.full(len(p), np.inf)
    if len(lmap.poles):
        dc = np.hypot(p[:, None, 0] - lmap.poles[None, :, 0], p[:, None, 1] - lmap.poles[None, :, 1])
        d = np.minimum(d, np.maximum(dc - lmap.poles[None, :, 2], 0.0).min(axis=1))
    for x0, y0, x1, y1 in lmap.walls:
        a, e = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
        t = np.clip(((p - a) @ e) / (e @ e), 0.0, 1.0)
        d = np.minimum(d, np.hypot(*(p - (a + t[:, None] * e)).T))
    return d


def test_cell_map_quality(lmap):
    import csv
    with CELL_MAP.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for cell in (2.0, 1.0):
        m = np.array([(float(r["mean_x"]), float(r["mean_y"])) for r in rows if float(r["cell_m"]) == cell])
        d = _dist_to_map(m, lmap)
        share = float((d <= MAP_DIST_M).mean())
        print(f"\ncell map {cell} m: cells={len(m)} within {MAP_DIST_M} m of a true pole or wall: "
              f"{share * 100:.1f} % (bound {MAP_SHARE_MIN * 100:.0f} % on 1.0 m cells) "
              f"median {np.median(d):.4f} m max {d.max():.4f} m")
        if cell == 1.0:
            assert share >= MAP_SHARE_MIN


@pytest.mark.parametrize("seed", SEEDS)
def test_built_map_converges_along_figure8(seed, lmap, built, centerline):
    rng, poses = _poses(seed, centerline, N_POSES)
    lidar = SimLidar2D(lmap, seed=seed)
    pos_err, yaw_err, ms, conv = [], [], [], 0
    for x, y, yaw in poses:
        pts = lidar.points(lidar.scan(x, y, yaw))
        guess = _guess(rng, x, y, yaw, GUESS_POS_M, GUESS_YAW_DEG)
        t0 = time.perf_counter()
        res = built.align(pts, guess)
        ms.append((time.perf_counter() - t0) * 1e3)
        e = math.hypot(res.x - x, res.y - y)
        if res.converged:
            assert e <= WRONG_POSE_M, f"confident wrong pose: {e:.3f} m"
            conv += 1
            pos_err.append(e)
            yaw_err.append(abs(math.degrees(_wrap(res.yaw - yaw))))
    rate = conv / len(poses)
    print(f"\nbuilt map seed {seed}: converged {conv}/{len(poses)} = {rate:.2f}, "
          f"pos err mean {np.mean(pos_err):.4f} max {np.max(pos_err):.4f} m, "
          f"yaw err mean {np.mean(yaw_err):.4f} max {np.max(yaw_err):.4f} deg, "
          f"runtime mean {np.mean(ms):.1f} max {np.max(ms):.1f} ms (budget {BUDGET_MS:.0f} ms)")
    assert rate >= MIN_CONVERGED_RATE
    assert np.mean(pos_err) <= POS_MEAN_BAR_M and np.max(pos_err) <= POS_MAX_BAR_M
    assert np.mean(yaw_err) <= YAW_MEAN_BAR_DEG and np.max(yaw_err) <= YAW_MAX_BAR_DEG
    assert np.mean(ms) <= RUNTIME_CI_BAR_MS
