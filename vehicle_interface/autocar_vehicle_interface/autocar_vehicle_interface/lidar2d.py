"""Simulated 2D lidar for the bicycle twin (PR-06b). Pure Python + numpy.

Raycasts a LandmarkMap (round poles and wall segments, for example
maps/figure8_v0/landmarks.csv) from the true pose. Range noise is Gaussian
with sigma_range; each beam is dropped (returned as NaN) with probability
dropout. Beams that hit nothing within range_max are NaN too.

The lidar owns its own seeded numpy Generator, so attaching it to a
BicycleBackend draws nothing from any other random stream and leaves the
existing G1 and G2 runs unchanged. It is opt-in: the backend has no lidar
unless one is attached.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

LIDAR_HZ = 10.0
N_BEAMS = 360
RANGE_MIN_M = 0.3
RANGE_MAX_M = 30.0
SIGMA_RANGE_M = 0.02
DROPOUT = 0.05


@dataclass(frozen=True)
class LandmarkMap:
    poles: NDArray[np.float64]  # (P, 3): x, y, radius
    walls: NDArray[np.float64]  # (W, 4): x0, y0, x1, y1

    @classmethod
    def from_csv(cls, path: Path) -> "LandmarkMap":
        poles, walls = [], []
        with Path(path).open(newline="") as f:
            for r in csv.DictReader(f):
                if r["kind"] == "pole":
                    poles.append((float(r["x0"]), float(r["y0"]), float(r["radius"])))
                elif r["kind"] == "wall":
                    walls.append((float(r["x0"]), float(r["y0"]), float(r["x1"]), float(r["y1"])))
                else:
                    raise ValueError(f"unknown landmark kind {r['kind']!r}")
        return cls(np.array(poles, float).reshape(-1, 3), np.array(walls, float).reshape(-1, 4))

    def sample_points(self, spacing: float = 0.05) -> NDArray[np.float64]:
        """Dense points on every surface: the NDT target cloud."""
        out = []
        for x, y, r in self.poles:
            n = max(8, int(math.ceil(2.0 * math.pi * r / spacing)))
            a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            out.append(np.column_stack([x + r * np.cos(a), y + r * np.sin(a)]))
        for x0, y0, x1, y1 in self.walls:
            n = max(2, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / spacing)) + 1)
            t = np.linspace(0.0, 1.0, n)
            out.append(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
        return np.vstack(out) if out else np.zeros((0, 2))


@dataclass(frozen=True)
class Lidar2DParams:
    n_beams: int = N_BEAMS
    range_min: float = RANGE_MIN_M
    range_max: float = RANGE_MAX_M
    sigma_range: float = SIGMA_RANGE_M
    dropout: float = DROPOUT


class SimLidar2D:
    def __init__(self, lmap: LandmarkMap, params: Lidar2DParams | None = None,
                 seed: int | None = None) -> None:
        self.map = lmap
        self.params = params or Lidar2DParams()
        p = self.params
        if p.n_beams < 1 or p.range_max <= p.range_min or p.sigma_range < 0.0 or not 0.0 <= p.dropout < 1.0:
            raise ValueError("bad lidar params")
        self.angles = np.arange(p.n_beams) * (2.0 * math.pi / p.n_beams) - math.pi
        self._rng = np.random.default_rng(seed)

    def true_ranges(self, x: float, y: float, yaw: float) -> NDArray[np.float64]:
        """Noise-free first-hit range per beam, inf where nothing is hit."""
        a = self.angles + yaw
        d = np.column_stack([np.cos(a), np.sin(a)])  # (B, 2)
        best = np.full(len(a), np.inf)
        if len(self.map.poles):
            c = self.map.poles[:, :2] - np.array([x, y])  # (P, 2)
            r = self.map.poles[:, 2]
            b = d @ c.T  # (B, P) projection of centre on beam
            disc = b * b - (np.sum(c * c, axis=1) - r * r)[None, :]
            t = b - np.sqrt(np.where(disc >= 0.0, disc, np.inf))
            t = np.where((disc >= 0.0) & (t > 0.0), t, np.inf)
            best = np.minimum(best, t.min(axis=1))
        if len(self.map.walls):
            p0 = self.map.walls[:, :2] - np.array([x, y])
            e = self.map.walls[:, 2:] - self.map.walls[:, :2]  # (W, 2)
            den = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # cross(d, e)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (p0[None, :, 0] * e[None, :, 1] - p0[None, :, 1] * e[None, :, 0]) / den
                u = (p0[None, :, 0] * d[:, 1:2] - p0[None, :, 1] * d[:, 0:1]) / den
            ok = (np.abs(den) > 1e-12) & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
            best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
        return best

    def scan(self, x: float, y: float, yaw: float) -> NDArray[np.float64]:
        """Ranges per beam (sensor frame = vehicle frame), NaN for no return."""
        p = self.params
        r = self.true_ranges(x, y, yaw)
        noise = self._rng.standard_normal(len(r)) * p.sigma_range
        drop = self._rng.random(len(r)) < p.dropout
        r = r + noise
        bad = ~np.isfinite(r) | drop | (r < p.range_min) | (r > p.range_max)
        return np.where(bad, np.nan, r)

    def points(self, ranges: NDArray[np.float64]) -> NDArray[np.float64]:
        """Valid returns as (N, 2) points in the sensor frame."""
        ok = np.isfinite(ranges)
        return np.column_stack([ranges[ok] * np.cos(self.angles[ok]),
                                ranges[ok] * np.sin(self.angles[ok])])
