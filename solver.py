from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import scipy.linalg
from config import SimulationConfig
from geometry import Mesh
from utils import EPSILON_0

@dataclass
class SolverResult:
    charges: np.ndarray
    capacitance: float
    residual_norm: float
    condition_number: float
    Q_plate1: float
    Q_plate2: float

def _build_matrix(mesh: Mesh, epsilon_r: float) -> np.ndarray:
    N = len(mesh.centers)
    k = 1.0 / (4.0 * math.pi * EPSILON_0 * epsilon_r)
    diff = mesh.centers[:, None, :] - mesh.centers[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(dist > 0, k / dist, 0.0)

    k_self = 1.0 / (2.0 * math.pi * EPSILON_0 * epsilon_r)
    dx = mesh.dx
    dy = mesh.dy
    diag = k_self * (np.arcsinh(dy / dx) / dy + np.arcsinh(dx / dy) / dx)
    np.fill_diagonal(A, diag)

    return A

def solve(cfg: SimulationConfig, mesh: Mesh) -> SolverResult:
    A = _build_matrix(mesh, cfg.epsilon_r)
    N = len(mesh.centers)
    phi_nom = mesh.potentials.copy()
    s = float(np.mean(np.diag(A)))
    A_aug = np.zeros((N + 1, N + 1))
    A_aug[:N, :N] = A
    A_aug[:N, N] = -s
    A_aug[N, :N] = +s
    rhs = np.zeros(N + 1)
    rhs[:N] = phi_nom
    cond = float(np.linalg.cond(A_aug))
    sol = scipy.linalg.solve(A_aug, rhs, assume_a='gen')
    charges = sol[:N]
    V0 = float(sol[N]) * s
    phi_real = phi_nom + V0
    residual = np.linalg.norm(A @ charges - phi_real)
    phi_norm = np.linalg.norm(phi_real)
    residual_norm = residual / phi_norm if phi_norm > 0 else residual
    Q1 = float(np.sum(charges[:mesh.n_plate1]))
    Q2 = float(np.sum(charges[mesh.n_plate1:]))
    V = cfg.voltage
    Q_positive = Q1 if cfg.plate1.potential > cfg.plate2.potential else Q2
    capacitance = abs(Q_positive / V)

    return SolverResult(
        charges=charges,
        capacitance=capacitance,
        residual_norm=residual_norm,
        condition_number=cond,
        Q_plate1=Q1,
        Q_plate2=Q2,
    )
