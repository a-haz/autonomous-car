"""PR-06a localization interface and GT/INS stub tests. No ROS."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
from autocar_loc.gt_ins import GtInsStub, Localization
from autocar_loc.pose import (
    COV_UNUSED,
    IPITCH,
    IROLL,
    IX,
    IY,
    IYAW,
    IZ,
    POSE_TOPIC,
    T_ABS_STALE_S,
    HealthMonitor,
    PoseEstimate,
    pose_covariance,
    position_uncertainty,
)

DESIGN = Path(__file__).resolve().parents[4] / "docs" / "design.md"


def test_topic_matches_design() -> None:
    assert POSE_TOPIC == "/localization/pose"
    assert re.search(r"\| `/localization/pose` \| `geometry_msgs/PoseWithCovarianceStamped` \| 50 Hz", DESIGN.read_text())


def test_uncertainty_is_sqrt_pxx_plus_pyy() -> None:
    p = pose_covariance(0.09, 0.16, 0.5, cov_xy=0.05)
    assert position_uncertainty(p) == pytest.approx(0.5)
    assert position_uncertainty(p.reshape(36)) == pytest.approx(0.5)
    assert position_uncertainty(np.diag([0.09, 0.16])) == pytest.approx(0.5)
    assert position_uncertainty(np.diag([0.09, 0.16, 9.0])) == pytest.approx(0.5)
    assert position_uncertainty(pose_covariance(0.0, 0.0, 0.0)) == 0.0


def test_covariance_layout() -> None:
    p = pose_covariance(1.0, 2.0, 3.0, cov_xy=0.5)
    assert p.shape == (6, 6) and np.allclose(p, p.T)
    assert (p[IX, IX], p[IY, IY], p[IYAW, IYAW], p[IX, IY]) == (1.0, 2.0, 3.0, 0.5)
    for i in (IZ, IROLL, IPITCH):
        assert p[i, i] == COV_UNUSED
    with pytest.raises(ValueError):
        pose_covariance(-1.0, 0.0, 0.0)


def test_to_msg_shape() -> None:
    e = PoseEstimate(5, 1.0, 2.0, math.pi / 2, 3.0, pose_covariance(0.04, 0.04, 0.01))
    m = e.to_msg()
    assert m["header"] == {"stamp_ns": 5, "frame_id": "map"}
    assert m["pose"]["position"] == (1.0, 2.0, 0.0)
    qx, qy, qz, qw = m["pose"]["orientation"]
    assert 2.0 * math.atan2(qz, qw) == pytest.approx(math.pi / 2)
    assert len(m["covariance"]) == 36 and m["covariance"][0] == 0.04 and m["covariance"][7] == 0.04


def test_noiseless_stub_is_ground_truth() -> None:
    gt = (3.0, -4.0, 0.7, 2.5)
    e = GtInsStub(lambda: gt).estimate(10)
    assert (e.x, e.y, e.theta, e.v) == gt
    assert e.sigma_pos == 0.0 and e.stamp_ns == 10


def test_stub_covariance_matches_sigma() -> None:
    e = GtInsStub(lambda: (0.0, 0.0, 0.0, 0.0), sigma_xy=0.2, sigma_yaw=0.01, sigma_v=0.1, seed=1).estimate(0)
    assert e.cov[IX, IX] == pytest.approx(0.04) and e.cov[IY, IY] == pytest.approx(0.04)
    assert e.cov[IYAW, IYAW] == pytest.approx(1e-4) and e.var_v == pytest.approx(0.01)
    assert e.sigma_pos == pytest.approx(0.2 * math.sqrt(2.0))


def test_stub_noise_statistics() -> None:
    n, sxy, syaw, sv = 20000, 0.3, 0.02, 0.1
    stub = GtInsStub(lambda: (10.0, -5.0, 0.0, 4.0), sigma_xy=sxy, sigma_yaw=syaw, sigma_v=sv, seed=7)
    s = np.array([[e.x, e.y, e.theta, e.v] for e in (stub.estimate(k) for k in range(n))])
    mean, std = s.mean(axis=0), s.std(axis=0, ddof=1)
    for k, (mu, sig) in enumerate(((10.0, sxy), (-5.0, sxy), (0.0, syaw), (4.0, sv))):
        assert abs(mean[k] - mu) < 5.0 * sig / math.sqrt(n)
        assert std[k] == pytest.approx(sig, rel=0.05)
    # Empirical position spread agrees with the reported sqrt(P_xx + P_yy).
    emp = math.sqrt(np.var(s[:, 0], ddof=1) + np.var(s[:, 1], ddof=1))
    assert emp == pytest.approx(stub.estimate(n).sigma_pos, rel=0.05)
    assert abs(np.corrcoef(s[:, 0], s[:, 1])[0, 1]) < 0.05


def test_stub_seed_reproducible() -> None:
    def seq(seed: int | None) -> list[tuple[float, float, float, float]]:
        st = GtInsStub(lambda: (0.0, 0.0, 3.1, 1.0), sigma_xy=0.1, sigma_yaw=0.1, sigma_v=0.1, seed=seed)
        return [(e.x, e.y, e.theta, e.v) for e in (st.estimate(k) for k in range(50))]

    assert seq(42) == seq(42)
    assert seq(42) != seq(43)
    assert all(-math.pi <= t < math.pi for _, _, t, _ in seq(42))


def test_health_drops_after_hold() -> None:
    mon = HealthMonitor(bar_m=0.8, hold_s=1.0)
    bad = pose_covariance(0.5, 0.5, 0.0)  # sqrt(1.0) = 1.0 m > 0.8 m
    good = pose_covariance(0.1, 0.1, 0.0)
    ms = 1_000_000
    assert mon.update(PoseEstimate(0, 0, 0, 0, 0, good))
    assert mon.update(PoseEstimate(100 * ms, 0, 0, 0, 0, bad))
    assert mon.update(PoseEstimate(1100 * ms, 0, 0, 0, 0, bad))
    assert not mon.update(PoseEstimate(1120 * ms, 0, 0, 0, 0, bad))
    assert mon.update(PoseEstimate(1140 * ms, 0, 0, 0, 0, good))


def test_health_drops_on_stale_fix_with_small_covariance() -> None:
    assert T_ABS_STALE_S == 0.5
    mon = HealthMonitor(bar_m=0.8, hold_s=1.0, t_abs_stale_s=0.5)
    good = pose_covariance(0.01, 0.01, 0.0)
    ms = 1_000_000

    def est(t_ms: int, fix_ms: int | None) -> PoseEstimate:
        return PoseEstimate(t_ms * ms, 0, 0, 0, 0, good, abs_fix_ns=None if fix_ms is None else fix_ms * ms)

    assert mon.update(est(0, 0))
    assert mon.update(est(500, 0))  # age 0.5 s is not stale yet
    assert not mon.bad(est(500, 0)) and mon.bad(est(520, 0))
    assert mon.update(est(520, 0))  # stale, hold starts
    assert mon.update(est(1520, 0))
    assert not mon.update(est(1540, 0))
    assert mon.update(est(1560, 1560))  # fresh fix clears it
    assert mon.update(est(9000, None))  # no freshness reported: covariance only


def test_backends_report_last_abs_fix() -> None:
    from autocar_loc.ekf import GNSS_RATE_HZ, EkfLoc

    ms = 1_000_000
    stub = GtInsStub(lambda: (0.0, 0.0, 0.0, 1.0))
    assert stub.estimate(0).abs_fix_ns == 0
    assert stub.estimate(20 * ms).abs_fix_ns == 20 * ms
    assert stub.inject_loc_drop(1.0, 40 * ms)["ok"]
    e = stub.estimate(60 * ms)
    assert e.abs_fix_ns == 20 * ms and e.abs_age_s == pytest.approx(0.04)
    assert stub.estimate(1040 * ms).abs_fix_ns == 1040 * ms

    ekf = EkfLoc(lambda: (0.0, 0.0, 0.0, 1.0), seed=0)
    fixes = [ekf.estimate(k * 20 * ms).abs_fix_ns for k in range(11)]
    period = int(round(1e3 / GNSS_RATE_HZ)) * ms
    assert fixes[0] == 0 and fixes[4] == 0 and fixes[5] == period and fixes[10] == 2 * period
    assert ekf.inject_loc_drop(1.0, 220 * ms)["ok"]
    assert all(ekf.estimate(t * ms).abs_fix_ns == 2 * period for t in range(220, 1220, 20))
    assert ekf.estimate(1220 * ms).abs_fix_ns == 1220 * ms


def test_localization_front_end() -> None:
    loc = Localization(GtInsStub(lambda: (1.0, 2.0, 0.0, 0.0)))
    assert loc.pose is None
    e = loc.step(20_000_000)
    assert loc.pose is e and loc.healthy and e.xyyaw() == (1.0, 2.0, 0.0)


def test_loc_drop_hook_topic() -> None:
    from autocar_loc.gt_ins import A_BOUND, DR_ACCEL_PSD, LOC_DROP_SRV
    from autocar_ctrl.bicycle import A_MIN

    assert LOC_DROP_SRV == "/scenario/inject/loc_drop"
    assert re.search(r"\| `/scenario/inject/loc_drop` \| `duration_s=2.0` \|", DESIGN.read_text())
    srv = (DESIGN.parents[1] / "messages" / "autocar_msgs" / "srv" / "InjectLocDrop.srv").read_text()
    assert "float32 duration_s" in srv
    assert A_BOUND == abs(A_MIN) and DR_ACCEL_PSD == A_BOUND**2


def test_loc_drop_rejects_bad_requests() -> None:
    stub = GtInsStub(lambda: (0.0, 0.0, 0.0, 0.0))
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert not stub.inject_loc_drop(bad, 0)["ok"]
    assert stub.inject_loc_drop(1.0, 0)["ok"]
    assert not stub.inject_loc_drop(1.0, 0)["ok"]
    assert not stub.gnss_ok


def test_loc_drop_dead_reckons_and_grows_covariance() -> None:
    ms = 1_000_000
    s = {"x": 0.0}
    stub = GtInsStub(lambda: (s["x"], 0.0, 0.0, 5.0))
    first = stub.estimate(0)
    assert first.sigma_pos == 0.0
    assert stub.inject_loc_drop(2.0, 20 * ms)["ok"]
    sig = []
    for k in range(1, 101):
        s["x"] = 5.0 * 0.02 * k
        e = stub.estimate(20 * ms * k)
        sig.append(e.sigma_pos)
        assert e.x == pytest.approx(s["x"])  # straight line at constant v: DR is exact
        t = 0.02 * k  # time since the last fix at stamp 0
        assert e.sigma_pos == pytest.approx(math.sqrt(2.0 * 9.0 * t**3 / 3.0))
    assert all(b > a for a, b in zip(sig[1:], sig[2:]))
    s["x"] = 5.0 * 0.02 * 101
    back = stub.estimate(20 * ms * 101)  # 2.02 s, drop over: fix again
    assert stub.gnss_ok and back.sigma_pos == 0.0 and back.x == s["x"]


def test_loc_drop_dead_reckoning_drifts_on_a_curve() -> None:
    """On a turn the DR pose drifts from GT; the reported sigma bounds the drift."""
    ms, v, r = 1_000_000, 5.0, 20.0
    s = {"t": 0.0}

    def gt() -> tuple[float, float, float, float]:
        a = v * s["t"] / r
        return (r * math.sin(a), r * (1.0 - math.cos(a)), a, v)

    stub = GtInsStub(gt)
    stub.estimate(0)
    stub.inject_loc_drop(2.0, 20 * ms)
    worst = 0.0
    for k in range(1, 100):
        s["t"] = 0.02 * k
        e = stub.estimate(20 * ms * k)
        gx, gy, _, _ = gt()
        err = math.hypot(e.x - gx, e.y - gy)
        worst = max(worst, err)
        if k > 5:
            assert err < e.sigma_pos
    assert worst > 0.0


def test_localization_reports_health_value() -> None:
    stub = GtInsStub(lambda: (0.0, 0.0, 0.0, 0.0))
    loc = Localization(stub)
    loc.step(0)
    assert loc.health_m == 0.0 and loc.healthy
    stub.inject_loc_drop(3.0, 20_000_000)
    for k in range(1, 100):
        loc.step(20_000_000 * k)
    assert loc.health_m > 0.8 and not loc.healthy
