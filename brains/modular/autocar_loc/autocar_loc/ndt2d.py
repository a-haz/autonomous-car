"""2D NDT scan matcher (PR-06b). Python v0 stand-in for ndt_omp.

This is NOT the production matcher. docs/design.md picks ndt_omp (pinned
in autocar_loc/NDT_PIN.md, which is not written yet) for lidar
localization. This module is a small pure numpy version of the same idea,
in 2D, so the figure-8 sim can have an absolute fix that does not depend
on GNSS before the real library is wired in. It is standalone: nothing in
the EKF calls it yet.

Method (Biber and Strasser style). The target cloud is cut into square
cells; each cell with enough points keeps a mean and a covariance whose
eigenvalues are floored at sigma_floor**2. A scan point q scores
exp(-0.5 d' inv(S) d) against the cell it falls in and the 8 around it,
d = q - mean. The pose (x, y, yaw) maximizes the summed score with damped
Newton steps. Three passes run coarse to fine, (cell, sigma_floor) =
(2.0, 0.6), (1.0, 0.25), (1.0, 0.1) m, so a guess about 0.5 m off is still
inside the basin of the first pass. A Newton step is only taken if it
raises the score (backtracking by halving).

Outputs, from the final pass:
- fitness: fraction of scan points whose best cell gives a squared
  Mahalanobis distance under CHI2_INLIER_2DOF.
- covariance (3x3, x y yaw): inverse of a conservative Gauss-Newton
  information. Each matched cell contributes once, not once per point,
  because points on one pole or wall patch are not independent. A cell
  whose shape is a line (eigenvalue ratio over PLANAR_RATIO) only
  constrains its normal; any other cell constrains both axes. The noise is
  the cell spread plus SIGMA_MEAS_M. A single straight wall therefore
  leaves the along-wall axis unconstrained and the covariance says so.
  A fixed model-error term (COV_MODEL_XY_M, COV_MODEL_YAW_RAD) is added
  for the pole centroid bias that the cell noise does not cover.
- converged: the Newton steps settled within MAX_ITER, fitness is at least
  MIN_FITNESS, at least MIN_INLIERS points matched, and the covariance is
  tight (sqrt of the largest x/y eigenvalue at most MAX_SIGMA_XY_M, yaw
  sigma at most MAX_SIGMA_YAW_RAD). A caller must not use the pose when
  converged is False.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PASSES = ((2.0, 0.6), (1.0, 0.25), (1.0, 0.1))  # (cell m, sigma_floor m)
MIN_CELL_POINTS = 3
PLANAR_RATIO = 10.0
SIGMA_MEAS_M = 0.05
CHI2_INLIER_2DOF = 9.21  # 99 %
MAX_ITER = 30
STEP_TOL_M = 1e-3
STEP_TOL_RAD = 1e-4
MAX_STEP_M = 0.5
MAX_STEP_RAD = 0.1
MIN_FITNESS = 0.5
MIN_INLIERS = 30
MAX_SIGMA_XY_M = 0.3
MAX_SIGMA_YAW_RAD = math.radians(2.0)
# Model error added to the reported covariance (PR-06b slice 2). A pole cell
# keeps the centroid of the whole circle, but the lidar only sees the near
# arc, whose centroid sits up to 2 r / pi = 0.095 m (r = 0.15 m) closer to
# the sensor. That bias is not in the per-cell noise, so the pose carries
# about that much unmodeled error per axis, and about 0.1 m over a 10 m
# lever arm, 0.01 rad, in yaw. Without this term the measured NEES of the
# match against the true pose was 3.65 to 5.27 per seed (consistent: 3).
COV_MODEL_XY_M = 0.10
COV_MODEL_YAW_RAD = 0.01

_NB = np.array([(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class NdtResult:
    x: float
    y: float
    yaw: float
    cov: NDArray[np.float64]  # 3x3, x y yaw
    fitness: float
    n_inliers: int
    n_points: int
    iterations: int
    converged: bool
    reason: str


class _Grid:
    def __init__(self, pts: NDArray[np.float64], cell: float, floor: float) -> None:
        self.cell = cell
        self.origin = pts.min(axis=0) - 2.0 * cell
        ij = np.floor((pts - self.origin) / cell).astype(np.int64)
        self.shape = tuple(ij.max(axis=0) + 3)
        nx, ny = self.shape
        key = ij[:, 0] * ny + ij[:, 1]
        order = np.argsort(key, kind="stable")
        key, p = key[order], pts[order]
        uniq, start, count = np.unique(key, return_index=True, return_counts=True)
        self.valid = np.zeros(nx * ny, bool)
        self.mean = np.zeros((nx * ny, 2))
        self.ainv = np.zeros((nx * ny, 2, 2))
        self.cov_raw = np.zeros((nx * ny, 2, 2))
        self.planar = np.zeros(nx * ny, bool)
        self.normal = np.zeros((nx * ny, 2))
        self.lam_min = np.zeros(nx * ny)
        for k, s, c in zip(uniq, start, count):
            if c < MIN_CELL_POINTS:
                continue
            q = p[s:s + c]
            mu = q.mean(axis=0)
            cov = np.cov((q - mu).T, bias=True)
            lam, vec = np.linalg.eigh(cov)
            lam_f = np.maximum(lam, floor * floor)
            self.valid[k] = True
            self.mean[k] = mu
            self.cov_raw[k] = cov
            self.ainv[k] = (vec / lam_f) @ vec.T
            self.planar[k] = lam[1] > PLANAR_RATIO * max(lam[0], 1e-6)
            self.normal[k] = vec[:, 0]
            self.lam_min[k] = max(lam[0], 0.0)

    def neighbours(self, q: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
        ij = np.floor((q - self.origin) / self.cell).astype(np.int64)
        cand = ij[:, None, :] + _NB[None, :, :]  # (N, 9, 2)
        nx, ny = self.shape
        inside = (cand[..., 0] >= 0) & (cand[..., 0] < nx) & (cand[..., 1] >= 0) & (cand[..., 1] < ny)
        k = np.where(inside, cand[..., 0] * ny + cand[..., 1], 0)
        return k, inside & self.valid[k]


class Ndt2D:
    """Target map in, scan + initial guess in, pose + covariance out."""

    def __init__(self, target_points: NDArray[np.float64]) -> None:
        pts = np.asarray(target_points, float).reshape(-1, 2)
        if len(pts) < MIN_CELL_POINTS:
            raise ValueError("target cloud too small")
        self.grids = [_Grid(pts, c, f) for c, f in PASSES]

    @staticmethod
    def _transform(p: NDArray[np.float64], x: float, y: float, yaw: float):
        c, s = math.cos(yaw), math.sin(yaw)
        q = np.column_stack([c * p[:, 0] - s * p[:, 1] + x, s * p[:, 0] + c * p[:, 1] + y])
        dq = np.column_stack([-s * p[:, 0] - c * p[:, 1], c * p[:, 0] - s * p[:, 1]])  # dq/dyaw
        return q, dq

    def _eval(self, g: _Grid, p, pose, need_derivs: bool = True):
        x, y, yaw = pose
        q, dq = self._transform(p, x, y, yaw)
        k, ok = g.neighbours(q)
        d = q[:, None, :] - g.mean[k]  # (N, 9, 2)
        A = g.ainv[k]  # (N, 9, 2, 2)
        Ad = np.einsum("nkij,nkj->nki", A, d)
        m = np.einsum("nki,nki->nk", d, Ad)
        w = np.where(ok, np.exp(-0.5 * np.minimum(m, 50.0)), 0.0)  # (N, 9)
        score = float(w.sum())
        if not need_derivs:
            return score, None, None
        # J = [I | dq]; gradient of the score is -w * Ad' J
        gad = np.concatenate([Ad, np.einsum("nki,ni->nk", Ad, dq)[..., None]], axis=-1)  # (N,9,3)
        grad = -np.einsum("nk,nkc->c", w, gad)
        JAJ = np.zeros((p.shape[0], 9, 3, 3))
        JAJ[..., :2, :2] = A
        Adq = np.einsum("nkij,nj->nki", A, dq)
        JAJ[..., :2, 2] = Adq
        JAJ[..., 2, :2] = Adq
        JAJ[..., 2, 2] = np.einsum("ni,nki->nk", dq, Adq)
        d2 = -(q - np.array([x, y]))  # second derivative of q wrt yaw
        JAJ[..., 2, 2] += np.einsum("nki,ni->nk", Ad, d2)
        H = np.einsum("nk,nkc,nkd->cd", w, gad, gad) - np.einsum("nk,nkcd->cd", w, JAJ)
        return score, grad, H

    def _newton(self, g: _Grid, p, pose):
        """Damped Newton ascent with backtracking: a step is taken only if
        it raises the score. Settled means the step fell under the
        tolerance or no step along the Newton direction raised the score."""
        pose = tuple(pose)
        settled = False
        it = 0
        score, grad, H = self._eval(g, p, pose)
        for it in range(1, MAX_ITER + 1):
            if score <= 0.0:
                break
            negH = -H
            lam_min = np.linalg.eigvalsh(negH).min()
            mu = max(0.0, -lam_min) + 1e-6 * max(1.0, abs(np.trace(negH)))
            step = np.linalg.solve(negH + mu * np.eye(3), grad)
            step *= min(1.0, MAX_STEP_M / max(math.hypot(step[0], step[1]), 1e-12),
                        MAX_STEP_RAD / max(abs(step[2]), 1e-12))
            accepted = False
            for _ in range(8):
                cand = (pose[0] + step[0], pose[1] + step[1], _wrap(pose[2] + step[2]))
                s_new, _, _ = self._eval(g, p, cand, need_derivs=False)
                if s_new >= score:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                settled = True
                break
            pose = cand
            score, grad, H = self._eval(g, p, pose)
            if math.hypot(step[0], step[1]) < STEP_TOL_M and abs(step[2]) < STEP_TOL_RAD:
                settled = True
                break
        return pose, settled, it

    def _quality(self, g: _Grid, p, pose):
        x, y, yaw = pose
        q, dq = self._transform(p, x, y, yaw)
        k, ok = g.neighbours(q)
        d = q[:, None, :] - g.mean[k]
        m = np.einsum("nki,nkij,nkj->nk", d, g.ainv[k], d)
        m = np.where(ok, m, np.inf)
        best = m.argmin(axis=1)
        mbest = m[np.arange(len(q)), best]
        inl = mbest < CHI2_INLIER_2DOF
        cells = k[np.arange(len(q)), best][inl]
        info = np.zeros((3, 3))
        dq_in = dq[inl]
        for c in np.unique(cells):
            sel = cells == c
            J = np.zeros((2, 3))
            J[:, :2] = np.eye(2)
            J[:, 2] = dq_in[sel].mean(axis=0)
            if g.planar[c]:
                n = g.normal[c]
                Jn = n @ J
                info += np.outer(Jn, Jn) / (g.lam_min[c] + SIGMA_MEAS_M**2)
            else:
                W = np.linalg.inv(g.cov_raw[c] + SIGMA_MEAS_M**2 * np.eye(2))
                info += J.T @ W @ J
        n_in = int(inl.sum())
        return info, n_in

    def align(self, scan_points: NDArray[np.float64], guess: tuple[float, float, float]) -> NdtResult:
        p = np.asarray(scan_points, float).reshape(-1, 2)
        big = np.diag([1e6, 1e6, 1e6])
        x, y, yaw = (float(v) for v in guess)
        if len(p) < MIN_INLIERS:
            return NdtResult(x, y, yaw, big, 0.0, 0, len(p), 0, False, "too few points")
        pose = (x, y, _wrap(yaw))
        settled = False
        its = 0
        for g in self.grids:
            pose, settled, it = self._newton(g, p, pose)
            its += it
        info, n_in = self._quality(self.grids[-1], p, pose)
        fitness = n_in / len(p)
        try:
            ev = np.linalg.eigvalsh(info)
            cov = np.linalg.inv(info) if ev.min() > 1e-9 * max(ev.max(), 1e-12) else big
        except np.linalg.LinAlgError:
            cov = big
        cov = 0.5 * (cov + cov.T) + np.diag([COV_MODEL_XY_M**2, COV_MODEL_XY_M**2, COV_MODEL_YAW_RAD**2])
        sig_xy = math.sqrt(max(np.linalg.eigvalsh(cov[:2, :2]).max(), 0.0))
        sig_yaw = math.sqrt(max(cov[2, 2], 0.0))
        if not settled:
            reason = "not settled"
        elif n_in < MIN_INLIERS:
            reason = "too few inliers"
        elif fitness < MIN_FITNESS:
            reason = "low fitness"
        elif sig_xy > MAX_SIGMA_XY_M or sig_yaw > MAX_SIGMA_YAW_RAD:
            reason = "degenerate"
        else:
            reason = "ok"
        return NdtResult(pose[0], pose[1], pose[2], cov, fitness, n_in, len(p), its,
                         reason == "ok", reason)
