"""HealthMonitor HOLD escalation on PoseEstimate.abs_suspect and sigma."""
from __future__ import annotations

from autocar_loc.pose import HOLD_SIGMA_M, T_ABS_STALE_S, HealthMonitor, PoseEstimate, pose_covariance

S = 1_000_000_000


def _est(t_s: float, fix_s: float, var: float, suspect: bool | None) -> PoseEstimate:
    return PoseEstimate(int(t_s * S), 0.0, 0.0, 0.0, 0.0, pose_covariance(var, var, 0.0),
                        abs_fix_ns=int(fix_s * S), abs_suspect=suspect)


def test_latched_source_requires_hold() -> None:
    hm = HealthMonitor()
    hm.update(_est(1.0, 1.0, 0.01, True))
    assert hm.hold_required


def test_sigma_over_bound_with_stale_fix_requires_hold() -> None:
    var = (HOLD_SIGMA_M * 1.01) ** 2 / 2.0
    hm = HealthMonitor()
    hm.update(_est(10.0, 10.0 - 2.0 * T_ABS_STALE_S, var, False))
    assert hm.hold_required
    hm.update(_est(10.0, 10.0, var, False))  # fresh fix: DEGRADED rules only
    assert not hm.hold_required


def test_backend_without_latch_never_escalates() -> None:
    hm = HealthMonitor()
    hm.update(_est(10.0, 0.0, 100.0, None))  # GT/INS stub style envelope
    assert not hm.hold_required
