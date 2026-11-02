"""PR-02b occupancy: inflate once, then centroids."""
from __future__ import annotations

import numpy as np

from autocar_command_gate.gate import GateParams, occupancy_hit, slice_to_xy
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


def test_gate_holds_for_slice_obstacle_dead_ahead() -> None:
    p = GateParams()
    for track, r_inflate, v in ((1.55, 0.30, 11.0), (0.25, 0.08, 8.0)):
        t = min(p.horizon_s, v / p.a_min_brake_rear)
        reach = v * t - 0.5 * p.a_min_brake_rear * t * t
        for ox in np.arange(0.5, reach, 0.25):
            for oy in (-0.5 * track, 0.0, 0.5 * track):
                sl = from_lidar_xy(np.array([[ox, oy]], np.float32), r_inflate=r_inflate, track=track)
                assert occupancy_hit(slice_to_xy(sl), (0.0, 0.0, 0.0), v, p, sl.resolution), (track, ox, oy)


def test_project_empty() -> None:
    sl = empty_slice()
    out = project_xy(np.zeros((0, 2), np.float32), sl)
    assert int(np.count_nonzero(out.cells == OCCUPIED)) == 0


def _inflate_reference(cells: np.ndarray, r: int) -> np.ndarray:
    """Per-cell disc stamping: the original inflate_once loop."""
    out = cells.copy()
    h, w = cells.shape
    for y, x in zip(*np.nonzero(cells == OCCUPIED), strict=True):
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        out[y0:y1, x0:x1][(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = OCCUPIED
    return out


def test_inflate_once_matches_per_cell_discs() -> None:
    rng = np.random.default_rng(7)
    for h, w, res, r_inflate, track in ((40, 60, 0.10, 0.30, 1.55), (25, 9, 0.10, 0.08, 0.25), (7, 5, 0.05, 1.0, 1.0)):
        sl = empty_slice(resolution=res, width=w, height=h)
        sl.cells[rng.random((h, w)) < 0.03] = OCCUPIED
        sl.cells[0, 0] = sl.cells[h - 1, w - 1] = OCCUPIED
        sl.cells[rng.random((h, w)) < 0.05] = 0
        r = int(np.ceil((r_inflate + 0.5 * track) / res))
        out = inflate_once(sl, r_inflate=r_inflate, track=track)
        assert np.array_equal(out.cells, _inflate_reference(sl.cells, r)), (h, w, r)
