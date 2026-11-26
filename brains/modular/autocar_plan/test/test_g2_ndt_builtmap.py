"""G2 NDT-on runs with the built cell map (maps/figure8_v0/ndt_cells.csv)
as the Ndt2D target instead of the landmark target.

The runs, seeds and bounds are exactly those of test_g2_ndt (GNSS-only
outage and clean NEES); only the matcher target changes. Those bounds were
stated before measuring. The lidar seeds of these runs (1000 to 1002) differ
from the map build seed (build_ndt_map.BUILD_SEED = 7).
"""
from __future__ import annotations

import pytest
from autocar_loc.ndt2d import Ndt2D

import test_g2_ndt as base

CELL_MAP = base.ROOT / "maps" / "figure8_v0" / "ndt_cells.csv"
_BUILT = Ndt2D.from_cell_map(CELL_MAP)


@pytest.fixture
def built_target(monkeypatch):
    monkeypatch.setattr(base, "_NDT", _BUILT)


@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_built_map_gnss_outage_keeps_running(built_target, seed, frac, off):
    print("\n[built map target]", end="")
    base.test_g2_ndt_gnss_outage_keeps_running(seed, frac, off)


@pytest.mark.parametrize("seed,frac,off", base.PARAMS)
def test_g2_built_map_clean_nees(built_target, seed, frac, off):
    print("\n[built map target]", end="")
    base.test_g2_ndt_clean_nees(seed, frac, off)
