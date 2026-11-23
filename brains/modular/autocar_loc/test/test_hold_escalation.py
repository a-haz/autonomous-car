"""HealthMonitor HOLD escalation on PoseEstimate.abs_suspect and sigma."""
from __future__ import annotations

from autocar_loc.pose import (HOLD_SIGMA_M, T_ABS_STALE_S, T_DEGRADED_MAX_S, UNHEALTHY_HOLD_S, HealthMonitor,
                              PoseEstimate, pose_covariance)

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


def test_degraded_dwell_limit_requires_hold() -> None:
    t_bad = 10.0
    limit = UNHEALTHY_HOLD_S + T_DEGRADED_MAX_S
    for suspect in (False, None):  # filter and stub style backends alike
        hm = HealthMonitor()
        hm.update(_est(t_bad, t_bad - 2.0 * T_ABS_STALE_S, 0.01, suspect))
        hm.update(_est(t_bad + limit - 0.02, t_bad - 2.0 * T_ABS_STALE_S, 0.01, suspect))
        assert not hm.hold_required
        hm.update(_est(t_bad + limit + 0.02, t_bad - 2.0 * T_ABS_STALE_S, 0.01, suspect))
        assert hm.hold_required
        hm.update(_est(t_bad + limit + 0.04, t_bad + limit + 0.04, 0.01, suspect))  # fresh fix
        assert not hm.hold_required
