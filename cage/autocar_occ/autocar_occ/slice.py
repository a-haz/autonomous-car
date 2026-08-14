"""OccupancySlice producer. Inflate once by r_inflate + track/2."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

FREE = 0
UNKNOWN = 127
OCCUPIED = 255


@dataclass(slots=True)
class Slice:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    bumper_z: float
    inflate_m: float
    cells: NDArray[np.uint8]
    source: str = "lidar"


def empty_slice(
    *,
    resolution: float = 0.10,
    width: int = 200,
    height: int = 200,
    origin_x: float = -10.0,
    origin_y: float = -10.0,
    bumper_z: float = 0.40,
    inflate_m: float = 0.0,
) -> Slice:
    return Slice(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        bumper_z=bumper_z,
        inflate_m=inflate_m,
        cells=np.full((height, width), UNKNOWN, dtype=np.uint8),
    )


def project_xy(points_xy: NDArray[np.float32], sl: Slice) -> Slice:
    cells = sl.cells.copy()
    if points_xy.size == 0:
        return replace(sl, cells=cells)
    ii = np.floor((points_xy[:, 0] - sl.origin_x) / sl.resolution).astype(np.int32)
    jj = np.floor((points_xy[:, 1] - sl.origin_y) / sl.resolution).astype(np.int32)
    m = (ii >= 0) & (ii < sl.width) & (jj >= 0) & (jj < sl.height)
    cells[jj[m], ii[m]] = OCCUPIED
    return Slice(
        resolution=sl.resolution,
        width=sl.width,
        height=sl.height,
        origin_x=sl.origin_x,
        origin_y=sl.origin_y,
        bumper_z=sl.bumper_z,
        inflate_m=sl.inflate_m,
        cells=cells,
        source=sl.source,
    )


def inflate_once(sl: Slice, r_inflate: float, track: float) -> Slice:
    inflate_m = r_inflate + 0.5 * track
    if inflate_m <= 0.0:
        return replace(sl, inflate_m=0.0)
    r = int(math.ceil(inflate_m / sl.resolution))
    occ = sl.cells == OCCUPIED
    out = sl.cells.copy()
    ys, xs = np.nonzero(occ)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        y0, y1 = max(0, y - r), min(sl.height, y + r + 1)
        x0, x1 = max(0, x - r), min(sl.width, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        region = out[y0:y1, x0:x1]
        region[mask] = OCCUPIED
    return Slice(
        resolution=sl.resolution,
        width=sl.width,
        height=sl.height,
        origin_x=sl.origin_x,
        origin_y=sl.origin_y,
        bumper_z=sl.bumper_z,
        inflate_m=inflate_m,
        cells=out,
        source=sl.source,
    )


def from_lidar_xy(
    points_xy: NDArray[np.float32],
    *,
    r_inflate: float,
    track: float,
    resolution: float = 0.10,
    width: int = 200,
    height: int = 200,
    origin_x: float = -10.0,
    origin_y: float = -10.0,
    bumper_z: float = 0.40,
) -> Slice:
    sl = empty_slice(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        bumper_z=bumper_z,
    )
    return inflate_once(project_xy(points_xy, sl), r_inflate, track)
