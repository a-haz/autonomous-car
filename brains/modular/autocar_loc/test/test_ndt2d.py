"""PR-06b: synthetic figure-8 landmark map, simulated 2D lidar and the
Python v0 NDT scan matcher (stand-in for ndt_omp). Standalone: the EKF does
not use the matcher yet.

Bounds stated before measuring. Per seed, over N_POSES poses along the
figure-8 with an initial guess GUESS_POS_M off in a random direction and
GUESS_YAW_DEG off in yaw with a random sign:
- converged rate >= MIN_CONVERGED_RATE
- on converged scans, position error mean <= POS_MEAN_BAR_M and
  max <= POS_MAX_BAR_M, heading error mean <= YAW_MEAN_BAR_DEG and
  max <= YAW_MAX_BAR_DEG
- no converged scan is more than WRONG_POSE_M off (no confident wrong pose)
Runtime per scan is printed against the 30 ms design budget. It is only
asserted against RUNTIME_CI_BAR_MS, ten times the budget, so a slow shared
CI runner does not fail the suite; the real budget is for ndt_omp.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest
from autocar_loc.ndt2d import Ndt2D
from autocar_vehicle_interface.lidar2d import LandmarkMap, Lidar2DParams, SimLidar2D

import make_figure8
import make_figure8_landmarks

ROOT = Path(__file__).resolve().parents[4]
MAP_DIR = ROOT / "maps" / "figure8_v0"

SEEDS = (1, 2, 3)
N_POSES = 20
GUESS_POS_M = 0.5
GUESS_YAW_DEG = 5.0
MIN_CONVERGED_RATE = 0.90
POS_MEAN_BAR_M = 0.10
POS_MAX_BAR_M = 0.30
YAW_MEAN_BAR_DEG = 0.5
YAW_MAX_BAR_DEG = 1.5
WRONG_POSE_M = 0.5
BUDGET_MS = 30.0
RUNTIME_CI_BAR_MS = 300.0
BAD_GUESS_POS_M = 4.0
BAD_GUESS_YAW_DEG = 30.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@pytest.fixture(scope="module")
def lmap() -> LandmarkMap:
    return LandmarkMap.from_csv(MAP_DIR / "landmarks.csv")


@pytest.fixture(scope="module")
def ndt(lmap) -> Ndt2D:
    return Ndt2D(lmap.sample_points())


@pytest.fixture(scope="module")
def centerline() -> np.ndarray:
    return np.array(make_figure8.load_centerline(MAP_DIR / "centerline.csv"))


def test_landmarks_checked_in_match_generator(tmp_path):
    out = make_figure8_landmarks.write_landmarks(tmp_path / "landmarks.csv")
    assert out.read_text() == (MAP_DIR / "landmarks.csv").read_text()
    assert (MAP_DIR / "VERSION").read_text().strip() == "2"


def test_map_keeps_poles_off_the_track(lmap, centerline):
    assert len(lmap.poles) > 0 and len(lmap.walls) == 4
    d = np.hypot(lmap.poles[:, None, 0] - centerline[None, :, 0],
                 lmap.poles[:, None, 1] - centerline[None, :, 1]).min(axis=1)
    assert d.min() >= make_figure8_landmarks.POLE_CLEARANCE_M - 1e-9


def test_lidar_raycast_geometry():
    m = LandmarkMap(np.array([[5.0, 0.0, 0.5]]), np.array([[-10.0, 3.0, 10.0, 3.0]]))
    lid = SimLidar2D(m, Lidar2DParams(n_beams=4, sigma_range=0.0, dropout=0.0), seed=0)
    r = lid.true_ranges(0.0, 0.0, 0.0)  # beams at -pi, -pi/2, 0, pi/2
    assert math.isinf(r[0]) and math.isinf(r[1])
    assert r[2] == pytest.approx(4.5)
    assert r[3] == pytest.approx(3.0)
    r_rot = lid.true_ranges(0.0, 0.0, math.pi / 2)  # beam 1 now points along +x
    assert r_rot[1] == pytest.approx(4.5)


def test_lidar_noise_dropout_seeded(lmap, centerline):
    x, y, yaw = centerline[100, :3]
    a = SimLidar2D(lmap, seed=7).scan(x, y, yaw)
    b = SimLidar2D(lmap, seed=7).scan(x, y, yaw)
    np.testing.assert_array_equal(a, b)
    lid = SimLidar2D(lmap, Lidar2DParams(sigma_range=0.0, dropout=0.5), seed=7)
    truth = lid.true_ranges(x, y, yaw)
    hits = np.isfinite(truth) & (truth <= lid.params.range_max)
    r = lid.scan(x, y, yaw)
    kept = np.isfinite(r)
    assert np.all(hits[kept])
    assert 0.3 < kept.sum() / hits.sum() < 0.7


def test_backend_lidar_is_opt_in(lmap):
    from autocar_ctrl.bicycle import BicycleState
    from autocar_vehicle_interface.bicycle import BicycleBackend

    plain = BicycleBackend(BicycleState(v=5.0))
    with_lidar = BicycleBackend(BicycleState(v=5.0))
    with_lidar.attach_lidar(SimLidar2D(lmap, seed=1))
    for be in (plain, with_lidar):
        be.set_command(0.1, 0.5)
    for _ in range(50):
        plain.step(0.02)
        with_lidar.step(0.02)
        with_lidar.lidar_scan()
    assert plain.state == with_lidar.state
    with pytest.raises(RuntimeError):
        plain.lidar_scan()


def _poses(seed: int, centerline: np.ndarray, n: int):
    rng = np.random.default_rng(seed)
    idx = (np.arange(n) * len(centerline) // n + rng.integers(0, len(centerline) // n)) % len(centerline)
    return rng, centerline[idx, :3]


def _guess(rng, x, y, yaw, pos_m, yaw_deg):
    a = rng.uniform(-math.pi, math.pi)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    return (x + pos_m * math.cos(a), y + pos_m * math.sin(a), _wrap(yaw + sgn * math.radians(yaw_deg)))


@pytest.mark.parametrize("seed", SEEDS)
def test_ndt_converges_along_figure8(seed, lmap, ndt, centerline):
    rng, poses = _poses(seed, centerline, N_POSES)
    lidar = SimLidar2D(lmap, seed=seed)
    pos_err, yaw_err, ms, conv, npts = [], [], [], 0, []
    for x, y, yaw in poses:
        pts = lidar.points(lidar.scan(x, y, yaw))
        guess = _guess(rng, x, y, yaw, GUESS_POS_M, GUESS_YAW_DEG)
        t0 = time.perf_counter()
        res = ndt.align(pts, guess)
        ms.append((time.perf_counter() - t0) * 1e3)
        npts.append(len(pts))
        e = math.hypot(res.x - x, res.y - y)
        if res.converged:
            assert e <= WRONG_POSE_M, f"confident wrong pose: {e:.3f} m"
            conv += 1
            pos_err.append(e)
            yaw_err.append(abs(math.degrees(_wrap(res.yaw - yaw))))
    rate = conv / len(poses)
    print(f"\nseed {seed}: converged {conv}/{len(poses)} = {rate:.2f}, "
          f"pos err mean {np.mean(pos_err):.4f} max {np.max(pos_err):.4f} m, "
          f"yaw err mean {np.mean(yaw_err):.4f} max {np.max(yaw_err):.4f} deg, "
          f"runtime mean {np.mean(ms):.1f} max {np.max(ms):.1f} ms (budget {BUDGET_MS:.0f} ms), "
          f"points per scan mean {np.mean(npts):.0f} min {np.min(npts)}")
    assert rate >= MIN_CONVERGED_RATE
    assert np.mean(pos_err) <= POS_MEAN_BAR_M and np.max(pos_err) <= POS_MAX_BAR_M
    assert np.mean(yaw_err) <= YAW_MEAN_BAR_DEG and np.max(yaw_err) <= YAW_MAX_BAR_DEG
    assert np.mean(ms) <= RUNTIME_CI_BAR_MS


def test_ndt_single_wall_is_not_converged():
    """Degenerate scan: one long straight wall. Along-wall position is not
    observable, so the matcher must say not converged."""
    m = LandmarkMap(np.zeros((0, 3)), np.array([[-200.0, 5.0, 200.0, 5.0]]))
    ndt = Ndt2D(m.sample_points())
    lidar = SimLidar2D(m, seed=11)
    rng = np.random.default_rng(11)
    for k in range(5):
        x, y, yaw = 3.0 * k, 0.0, 0.1 * k
        pts = lidar.points(lidar.scan(x, y, yaw))
        res = ndt.align(pts, _guess(rng, x, y, yaw, GUESS_POS_M, GUESS_YAW_DEG))
        print(f"\nsingle wall {k}: points {len(pts)} converged {res.converged} reason {res.reason} "
              f"sigma_x {math.sqrt(res.cov[0, 0]):.3g} m")
        assert not res.converged


@pytest.mark.parametrize("seed", SEEDS)
def test_ndt_bad_guess_never_confident_and_wrong(seed, lmap, ndt, centerline):
    rng, poses = _poses(100 + seed, centerline, 10)
    lidar = SimLidar2D(lmap, seed=100 + seed)
    n_conv = 0
    for x, y, yaw in poses:
        pts = lidar.points(lidar.scan(x, y, yaw))
        res = ndt.align(pts, _guess(rng, x, y, yaw, BAD_GUESS_POS_M, BAD_GUESS_YAW_DEG))
        e = math.hypot(res.x - x, res.y - y)
        if res.converged:
            n_conv += 1
            assert e <= WRONG_POSE_M, f"confident wrong pose {e:.2f} m"
    print(f"\nbad guess seed {seed}: converged {n_conv}/10 (all converged within {WRONG_POSE_M} m)")


def test_ndt_too_few_points_is_not_converged(ndt):
    res = ndt.align(np.zeros((5, 2)), (0.0, 0.0, 0.0))
    assert not res.converged and res.reason == "too few points"
