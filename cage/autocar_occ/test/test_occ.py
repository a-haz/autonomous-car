"""PR-02b occupancy: inflate once, then centroids."""
from __future__ import annotations

import numpy as np

from autocar_command_gate.gate import slice_to_xy
from autocar_occ.slice import OCCUPIED, from_lidar_xy, inflate_once, project_xy, empty_slice


def test_inflate_once_radius() -> None:
    sl = empty_slice(resolution=0.10, width=21, height=21, origin_x=-1.05, origin_y=-1.05)
    sl.cells[10, 10] = OCCUPIED
    out = inflate_once(sl, r_inflate=0.20, track=0.20)
    assert abs(out.inflate_m - 0.30) < 1e-9
    n = int(np.count_nonzero(out.cells == OCCUPIED))
    assert n > 1
    raw = int(np.count_nonzero(sl.cells == OCCUPIED))
    assert raw == 1


def test_lidar_project_and_centroids() -> None:
    pts = np.array([[0.05, 0.05]], dtype=np.float32)
    sl = from_lidar_xy(
        pts,
        r_inflate=0.0,
        track=0.0,
        resolution=0.10,
        width=4,
        height=4,
        origin_x=0.0,
        origin_y=0.0,
    )
    xy = slice_to_xy(sl)
    assert xy.shape[0] >= 1
    assert np.all(xy[:, 0] >= 0.0)


def test_project_empty() -> None:
    sl = empty_slice()
    out = project_xy(np.zeros((0, 2), np.float32), sl)
    assert int(np.count_nonzero(out.cells == OCCUPIED)) == 0
