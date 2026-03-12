from __future__ import annotations
import math
import numpy as np
from config import SimulationConfig
from geometry import Mesh
from solver import SolverResult
from utils import EPSILON_0

_BLOCK_SIZE = 500
_VIS_POINT_BLOCK_SIZE = 200
_VIS_SOURCE_BLOCK_SIZE = 4000

def compute_potential(
    points: np.ndarray,
    mesh:   Mesh,
    result: SolverResult,
    epsilon_r: float,
) -> np.ndarray:
    k = 1.0 / (4.0 * math.pi * EPSILON_0 * epsilon_r)
    M = len(points)
    phi = np.zeros(M)

    for start in range(0, M, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, M)
        pts = points[start:end]
        diff = pts[:, None, :] - mesh.centers[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist = np.where(dist > 0, 1.0 / dist, 0.0)
        phi[start:end] = k * (inv_dist * result.charges[None, :]).sum(axis=1)

    return phi


def compute_field(
    points: np.ndarray,
    mesh:   Mesh,
    result: SolverResult,
    epsilon_r: float,
) -> np.ndarray:
    k = 1.0 / (4.0 * math.pi * EPSILON_0 * epsilon_r)
    M = len(points)
    E = np.zeros((M, 3))

    for start in range(0, M, _BLOCK_SIZE):
        end = min(start + _BLOCK_SIZE, M)
        pts = points[start:end]
        diff = pts[:, None, :] - mesh.centers[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist3 = np.where(dist > 0, 1.0 / dist**3, 0.0)
        weights = result.charges[None, :] * inv_dist3
        E[start:end] = k * (weights[:, :, None] * diff).sum(axis=1)

    return E

def _panel_quadrature_sources(
    mesh: Mesh,
    charges: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    if quadrature_order < 1:
        raise ValueError(f"quadrature_order должен быть >= 1, получено {quadrature_order}")

    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(quadrature_order)
    u_nodes, v_nodes = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
    uv_weights = 0.25 * np.outer(weights_1d, weights_1d).ravel()
    n_quad = quadrature_order**2
    u_offsets = 0.5 * mesh.dx[:, None] * u_nodes.ravel()[None, :]
    v_offsets = 0.5 * mesh.dy[:, None] * v_nodes.ravel()[None, :]
    centers = mesh.centers[:, None, :]
    quad_positions = np.empty((len(mesh.centers), n_quad, 3))
    quad_positions[:, :, 0] = centers[:, :, 0] + u_offsets
    quad_positions[:, :, 1] = centers[:, :, 1] + v_offsets
    quad_positions[:, :, 2] = centers[:, :, 2]
    quad_charges = charges[:, None] * uv_weights[None, :]

    return quad_positions.reshape(-1, 3), quad_charges.reshape(-1)


def _compute_distributed(
    points: np.ndarray,
    mesh: Mesh,
    result: SolverResult,
    epsilon_r: float,
    quadrature_order: int,
    want_potential: bool,
    want_field: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    k = 1.0 / (4.0 * math.pi * EPSILON_0 * epsilon_r)
    source_positions, source_charges = _panel_quadrature_sources(
        mesh, result.charges, quadrature_order=quadrature_order
    )

    M = len(points)
    phi = np.zeros(M) if want_potential else None
    E = np.zeros((M, 3)) if want_field else None

    for p_start in range(0, M, _VIS_POINT_BLOCK_SIZE):
        p_end = min(p_start + _VIS_POINT_BLOCK_SIZE, M)
        pts = points[p_start:p_end]
        phi_block = np.zeros(len(pts)) if want_potential else None
        E_block = np.zeros((len(pts), 3)) if want_field else None

        for s_start in range(0, len(source_positions), _VIS_SOURCE_BLOCK_SIZE):
            s_end = min(s_start + _VIS_SOURCE_BLOCK_SIZE, len(source_positions))
            src_pos = source_positions[s_start:s_end]
            src_q = source_charges[s_start:s_end]

            diff = pts[:, None, :] - src_pos[None, :, :]
            dist = np.linalg.norm(diff, axis=2)

            if want_potential:
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_dist = np.where(dist > 0, 1.0 / dist, 0.0)
                phi_block += (inv_dist * src_q[None, :]).sum(axis=1)

            if want_field:
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_dist3 = np.where(dist > 0, 1.0 / dist**3, 0.0)
                weights = src_q[None, :] * inv_dist3
                E_block += (weights[:, :, None] * diff).sum(axis=1)

        if want_potential:
            phi[p_start:p_end] = k * phi_block
        if want_field:
            E[p_start:p_end] = k * E_block

    return phi, E

def compute_potential_distributed(
    points: np.ndarray,
    mesh: Mesh,
    result: SolverResult,
    epsilon_r: float,
    quadrature_order: int = 2,
) -> np.ndarray:
    phi, _ = _compute_distributed(
        points,
        mesh,
        result,
        epsilon_r,
        quadrature_order=quadrature_order,
        want_potential=True,
        want_field=False,
    )
    return phi

def compute_field_distributed(
    points: np.ndarray,
    mesh: Mesh,
    result: SolverResult,
    epsilon_r: float,
    quadrature_order: int = 2,
) -> np.ndarray:
    _, E = _compute_distributed(
        points,
        mesh,
        result,
        epsilon_r,
        quadrature_order=quadrature_order,
        want_potential=False,
        want_field=True,
    )
    return E

def compute_potential_field_distributed(
    points: np.ndarray,
    mesh: Mesh,
    result: SolverResult,
    epsilon_r: float,
    quadrature_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    phi, E = _compute_distributed(
        points,
        mesh,
        result,
        epsilon_r,
        quadrature_order=quadrature_order,
        want_potential=True,
        want_field=True,
    )
    return phi, E

def make_grid_xz(
    cfg: SimulationConfig,
    mesh: Mesh,
    n_points: int = 40,
    margin: float = 0.5,
    y_slice: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = max(cfg.plate1.length, cfg.plate2.length)
    T1, T2 = cfg.plate1.thickness, cfg.plate2.thickness
    z_top = cfg.plate1_z_center + T1 / 2
    z_bot = cfg.plate2_z_center - T2 / 2
    x_half = L / 2 * (1 + margin)
    z_span = (z_top - z_bot) * (1 + margin)
    z_mid = (z_top + z_bot) / 2
    x_vals = np.linspace(-x_half, x_half, n_points)
    z_vals = np.linspace(z_mid - z_span / 2, z_mid + z_span / 2, n_points)
    X, Z = np.meshgrid(x_vals, z_vals)
    points = np.column_stack([X.ravel(), np.full(n_points**2, y_slice), Z.ravel()])
    return X, Z, points


def make_grid_xy(
    cfg: SimulationConfig,
    n_points: int = 40,
    margin: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    L = max(cfg.plate1.length, cfg.plate2.length)
    W = max(cfg.plate1.width,  cfg.plate2.width)
    x_vals = np.linspace(-L / 2 * (1 + margin), L / 2 * (1 + margin), n_points)
    y_vals = np.linspace(-W / 2 * (1 + margin), W / 2 * (1 + margin), n_points)
    X, Y = np.meshgrid(x_vals, y_vals)

    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(n_points**2)])
    return X, Y, points
