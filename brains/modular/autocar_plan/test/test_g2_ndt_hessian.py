"""G2 NDT-on runs with the built cell map and the Laplace (Hessian) R.

Bounds stated before measuring:
- Clean NEES (test_g2_ndt clean loop, seeds as there): for the converged
  NDT fixes, the mean of e' inv(R) e over a run (e = match pose minus true
  pose, yaw wrapped, 3 dof) lies inside the 95 % chi-square interval for 3
  dof, [CHI2_3_LO, CHI2_3_HI] = [0.216, 9.348], with the Hessian R. The
  fixed R mean is printed alongside, not asserted (it was 0.065 to 0.103
  on the built map). The tighter interval for the mean of N independent
  samples is printed, not asserted.
- GNSS-only outage (test_g2_ndt outage loop) with the Hessian R: no HOLD,
  RUNNING at the end, and per seed max lateral excursion and max pose error
  no more than MARGIN_M = 0.05 m above the built-map fixed-R numbers of
  f554c18 (LAST_LAT_M, LAST_POS_M).
- Corridor (two parallel walls only) driven straight with GNSS off after
  1 s: the along-wall EKF error with NDT on, with either R, is no more than
  ALONG_MARGIN_M = 0.10 m above the same run with NDT off. NDT must not pull
  the pose along the walls.
"""
from __future__ import annotations

import csv
import functools
import math

import numpy as np
import pytest
from autocar_loc.ekf import EkfLoc
from autocar_loc.ndt2d import Ndt2D
from autocar_supervisor.sm import St
from autocar_vehicle_interface.lidar2d import LandmarkMap, SimLidar2D

import test_g2_ndt as base
import build_ndt_map

CHI2_3_LO = 0.216
CHI2_3_HI = 9.348
MARGIN_M = 0.05
LAST_LAT_M = (0.1002, 0.1814, 0.1749)
LAST_POS_M = (0.0261, 0.0246, 0.0241)
ALONG_MARGIN_M = 0.10
CORRIDOR = LandmarkMap(np.zeros((0, 3)), np.array([[-100.0, 5.0, 300.0, 5.0], [-100.0, -5.0, 300.0, -5.0]]))


def corridor_cell_map(path) -> Ndt2D:
    """Same corridor and built-map path as test_ndt_hessian_cov (build seed 7)."""
    lid = SimLidar2D(CORRIDOR, seed=build_ndt_map.BUILD_SEED)
    pts = [lid.points(lid.scan(x, 0.0, 0.0)) + np.array([x, 0.0]) for x in np.arange(-60.0, 260.0, 0.5)]
    vox = build_ndt_map.voxelize(np.vstack(pts))
    rows = [r for c in build_ndt_map.CELL_SIZES_M for r in build_ndt_map.cells(vox, c)]
    with open(path, "w", newline="\n") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(build_ndt_map.HEADER)
        w.writerows(rows)
    return Ndt2D.from_cell_map(path)


_BUILT = Ndt2D.from_cell_map(base.ROOT / "maps" / "figure8_v0" / "ndt_cells.csv")


class _Recorder:
    def __init__(self, ndt: Ndt2D) -> None:
        self.ndt, self.results = ndt, []

    def align(self, pts, guess):
        res = self.ndt.align(pts, guess)
        self.results.append(res)
        return res


@pytest.fixture
def hessian_r(monkeypatch):
    rec = _Recorder(_BUILT)
    monkeypatch.setattr(base, "_NDT", rec)
    monkeypatch.setattr(base, "EkfLoc", functools.partial(EkfLoc, ndt_r="hessian"))
    return rec


def _wh(p_z: float, k: float) -> float:
    """Wilson-Hilferty quantile of chi-square with k dof."""
    return k * (1.0 - 2.0 / (9.0 * k) + p_z * math.sqrt(2.0 / (9.0 * k))) ** 3


@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_hessian_r_clean_nees(hessian_r, seed, frac, off):
    r = base._run(frac, off, seed, base.T_CLEAN)
    ekf = r["ekf"]
    assert len(hessian_r.results) == len(r["meas"])
    nf, nh = [], []
    for m, res in zip(r["meas"], hessian_r.results):
        if not res.converged:
            continue
        tx, ty, tyaw = m["truth"]
        e = np.array([res.x - tx, res.y - ty, base._wrap(res.yaw - tyaw)])
        nf.append(float(e @ np.linalg.solve(res.cov, e)))
        nh.append(float(e @ np.linalg.solve(res.cov_hessian, e)))
    n = len(nh)
    lo_n, hi_n = _wh(-1.96, 3 * n) / n, _wh(1.96, 3 * n) / n
    mh, mf = float(np.mean(nh)), float(np.mean(nf))
    print(f"\nG2 hessian R clean seed={seed} fixes={n} accepted={ekf.n_ndt_accepted} rejected={ekf.n_ndt_rejected} "
          f"not_converged={ekf.n_ndt_not_converged} slow={ekf.n_ndt_slow}  NEES mean hessian={mh:.3f} "
          f"median={np.median(nh):.3f} fixed={mf:.3f}  interval 3 dof [{CHI2_3_LO}, {CHI2_3_HI}]  "
          f"mean-of-{n} interval [{lo_n:.3f}, {hi_n:.3f}] hessian inside={lo_n <= mh <= hi_n} (printed only)  "
          f"filter NEES mean={r['fnees'][r['t'] >= 1.0].mean():.3f}")
    assert {s for s, ti in zip(r["state"], r["t"]) if ti >= 1.0} == {St.RUNNING}
    assert CHI2_3_LO <= mh <= CHI2_3_HI


@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_hessian_r_gnss_outage(hessian_r, seed, frac, off):
    r = base._run(frac, off, seed, base.T_END_OUT, outage=(base.T_OUT, base.OUTAGE_S))
    t, states = r["t"], r["state"]
    win = t >= base.T_OUT - 1e-9
    t1 = base.T_OUT + base.OUTAGE_S
    na, nr, nn, ns = (base._delta(r, k, base.T_OUT, t1) for k in ("nacc", "nrej", "nnc", "nslow"))
    max_lat = float(r["lat"][win].max())
    max_pos = float(r["pos_err"][win].max())
    print(f"\nG2 hessian R gnss_outage seed={seed} timeline: {base._timeline(r)}  ndt in outage: accepted={na} "
          f"rejected={nr} not_converged={nn} slow={ns}  max_lat={max_lat:.4f} m (last {LAST_LAT_M[seed]}) "
          f"max_pos_err={max_pos:.4f} m (last {LAST_POS_M[seed]})")
    assert St.HOLD not in states and not r["hold_req"].any()
    assert states[-1] is St.RUNNING
    assert max_lat <= LAST_LAT_M[seed] + MARGIN_M
    assert max_pos <= LAST_POS_M[seed] + MARGIN_M


def _corridor_run(ndt, ndt_r: str | None) -> np.ndarray:
    v, dt, t_end = 5.0, 0.02, 10.0
    state = {"t": 0.0}

    def gt():
        return (v * state["t"], 0.0, 0.0, v)

    lid = SimLidar2D(CORRIDOR, seed=31)

    def scan():
        return lid.points(lid.scan(v * state["t"], 0.0, 0.0))

    ekf = EkfLoc(gt, seed=31) if ndt_r is None else EkfLoc(gt, seed=31, ndt=ndt, scan=scan, ndt_r=ndt_r)
    along = []
    for k in range(int(round(t_end / dt))):
        state["t"] = k * dt
        stamp = int(round(state["t"] * 1e9))
        if k == int(round(1.0 / dt)):
            assert ekf.inject_gnss_outage(t_end, stamp)["ok"]
        ekf.estimate(stamp)
        along.append(ekf.s[0] - v * state["t"])
    return np.array(along), ekf


def test_corridor_ekf_not_pulled_along_walls(tmp_path):
    ndt = corridor_cell_map(tmp_path / "corridor.csv")
    off, _ = _corridor_run(None, None)
    w = slice(int(round(1.0 / 0.02)), None)
    base_max = float(np.abs(off[w]).max())
    out = [f"ndt off max |along err|={base_max:.4f} m"]
    for mode in ("fixed", "hessian"):
        a, ekf = _corridor_run(ndt, mode)
        m = float(np.abs(a[w]).max())
        out.append(f"{mode} R max |along err|={m:.4f} m final={a[-1]:+.4f} (accepted={ekf.n_ndt_accepted} "
                   f"rejected={ekf.n_ndt_rejected} not_converged={ekf.n_ndt_not_converged})")
        assert m <= base_max + ALONG_MARGIN_M
    print("\ncorridor EKF: " + "  ".join(out))
