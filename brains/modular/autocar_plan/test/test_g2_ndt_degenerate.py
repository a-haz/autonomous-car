"""PR-06b degeneracy-aware NDT covariance (NdtResult.cov_hessian_deg,
EkfLoc(ndt_r="hessian_deg")).

Bounds stated before measuring:
- (a) Corridor of two straight walls, built-map path, the five positions of
  test_ndt_hessian_cov: coverage rule, at every position the actual
  along-wall match error is at most COVER_K = 3 times the along-wall sigma
  of the new R. The along/across ratio and whether the across-wall sigma
  sits on the 0.01 m floor are printed.
- (b) Corridor EKF, GNSS off after 1 s, with the fixed-covariance
  degeneracy check turned off in the matcher for this test only: max
  along-wall error with the new R is at most ALONG_MARGIN_M = 0.10 m above
  the same run with NDT off. The plain Hessian R is printed only.
- (c) Built map, clean loop of test_g2_ndt, 3 seeds: the mean NDT match
  NEES with the new R lies inside the 95 % chi-square interval for the
  mean of N independent 3 dof samples (Wilson-Hilferty quantiles of
  chi-square with 3N dof, divided by N). This fails today and is marked
  xfail (strict) with the measured numbers; see the test.
- (d) GNSS-only outage of test_g2_ndt with the new R: no HOLD, RUNNING at
  the end, max lateral and max pose error at most MARGIN_M = 0.05 m above
  the plain Hessian R numbers of a8529f4 (LAST_LAT_M, LAST_POS_M).
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from autocar_loc.ekf import EkfLoc
from autocar_loc.ndt2d import HESSIAN_FLOOR_XY_M
from autocar_supervisor.sm import St
from autocar_vehicle_interface.lidar2d import SimLidar2D

import test_g2_ndt as base
from test_g2_ndt_hessian import ALONG_MARGIN_M, CORRIDOR, MARGIN_M, _BUILT, _corridor_run, _Recorder, _wh, corridor_cell_map

COVER_K = 3.0
GUESS_POS_M = 0.5
GUESS_YAW_DEG = 5.0
LAST_LAT_M = (0.1046, 0.1899, 0.1756)
LAST_POS_M = (0.0328, 0.0364, 0.0416)
CORRIDOR_XS = (0.0, 13.0, 37.0, 71.0, 120.0)


def _guess(rng, x: float) -> tuple[float, float, float]:
    """Same draw as test_ndt2d._guess at (x, 0, 0): 0.5 m random direction, 5 deg random sign."""
    a = rng.uniform(-math.pi, math.pi)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    return (x + GUESS_POS_M * math.cos(a), GUESS_POS_M * math.sin(a), sgn * math.radians(GUESS_YAW_DEG))


@pytest.fixture
def deg_r(monkeypatch):
    rec = _Recorder(_BUILT)
    monkeypatch.setattr(base, "_NDT", rec)
    monkeypatch.setattr(base, "EkfLoc", functools.partial(EkfLoc, ndt_r="hessian_deg"))
    return rec


def test_corridor_new_r_covers_along_wall_error(tmp_path):
    ndt = corridor_cell_map(tmp_path / "corridor.csv")
    lid = SimLidar2D(CORRIDOR, seed=21)
    rng = np.random.default_rng(21)
    for x in CORRIDOR_XS:
        pts = lid.points(lid.scan(x, 0.0, 0.0))
        res = ndt.align(pts, _guess(rng, x))
        c, ch = res.cov_hessian_deg, res.cov_hessian
        sa, sx = math.sqrt(c[0, 0]), math.sqrt(c[1, 1])
        err = res.x - x
        print(f"\ncorridor x={x:.0f}: along_err={err:+.3f} m new R sigma along={sa:.4f} across={sx:.4f} m "
              f"ratio(var)={c[0, 0] / c[1, 1]:.1f} across_on_floor={abs(sx - HESSIAN_FLOOR_XY_M) < 1e-6}  "
              f"plain Hessian sigma along={math.sqrt(ch[0, 0]):.4f} m  covered={abs(err) <= COVER_K * sa}")
        assert abs(err) <= COVER_K * sa


def test_corridor_ekf_check_off_new_r_not_pulled(tmp_path):
    ndt = corridor_cell_map(tmp_path / "corridor.csv")
    ndt.check_degenerate = False
    off, _ = _corridor_run(None, None)
    w = slice(int(round(1.0 / 0.02)), None)
    base_max = float(np.abs(off[w]).max())
    out, res = [f"ndt off max |along err|={base_max:.4f} m"], {}
    for mode in ("hessian", "hessian_deg"):
        a, ekf = _corridor_run(ndt, mode)
        res[mode] = float(np.abs(a[w]).max())
        out.append(f"{mode} max |along err|={res[mode]:.4f} m final={a[-1]:+.4f} (accepted={ekf.n_ndt_accepted} "
                   f"rejected={ekf.n_ndt_rejected} not_converged={ekf.n_ndt_not_converged})")
    print("\ncorridor EKF, degeneracy check off: " + "  ".join(out))
    assert res["hessian_deg"] <= base_max + ALONG_MARGIN_M


# Known failure, marked after measuring; the interval was not loosened and
# the scale was not fitted. Measured mean NEES with the new R: 0.588, 0.818,
# 0.597 for seeds 0, 1, 2 against about [2.73, 3.28]. On the figure-8 no
# position direction is degenerate, so the new R equals the plain Hessian R
# on every fix and stays conservative.
@pytest.mark.xfail(strict=True, reason="new R conservative: mean NEES 0.59 to 0.82, strict interval about [2.73, 3.28]")
@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_new_r_strict_nees(deg_r, seed, frac, off):
    r = base._run(frac, off, seed, base.T_CLEAN)
    assert len(deg_r.results) == len(r["meas"])
    nd, nh = [], []
    for m, res in zip(r["meas"], deg_r.results):
        if not res.converged:
            continue
        tx, ty, tyaw = m["truth"]
        e = np.array([res.x - tx, res.y - ty, base._wrap(res.yaw - tyaw)])
        nd.append(float(e @ np.linalg.solve(res.cov_hessian_deg, e)))
        nh.append(float(e @ np.linalg.solve(res.cov_hessian, e)))
    n = len(nd)
    lo, hi = _wh(-1.96, 3 * n) / n, _wh(1.96, 3 * n) / n
    md = float(np.mean(nd))
    n_changed = sum(abs(a - b) > 1e-12 for a, b in zip(nd, nh))
    print(f"\nG2 new R strict NEES seed={seed} fixes={n} mean={md:.3f} median={np.median(nd):.3f} "
          f"interval [{lo:.3f}, {hi:.3f}] inside={lo <= md <= hi}  plain Hessian mean={np.mean(nh):.3f}  "
          f"fixes where new R differs from plain={n_changed}")
    assert lo <= md <= hi


@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_new_r_gnss_outage(deg_r, seed, frac, off):
    r = base._run(frac, off, seed, base.T_END_OUT, outage=(base.T_OUT, base.OUTAGE_S))
    t, states = r["t"], r["state"]
    win = t >= base.T_OUT - 1e-9
    t1 = base.T_OUT + base.OUTAGE_S
    na, nr, nn, ns = (base._delta(r, k, base.T_OUT, t1) for k in ("nacc", "nrej", "nnc", "nslow"))
    max_lat = float(r["lat"][win].max())
    max_pos = float(r["pos_err"][win].max())
    print(f"\nG2 new R gnss_outage seed={seed} timeline: {base._timeline(r)}  ndt in outage: accepted={na} "
          f"rejected={nr} not_converged={nn} slow={ns}  max_lat={max_lat:.4f} m (last {LAST_LAT_M[seed]}) "
          f"max_pos_err={max_pos:.4f} m (last {LAST_POS_M[seed]})")
    assert St.HOLD not in states and not r["hold_req"].any()
    assert states[-1] is St.RUNNING
    assert max_lat <= LAST_LAT_M[seed] + MARGIN_M
    assert max_pos <= LAST_POS_M[seed] + MARGIN_M
