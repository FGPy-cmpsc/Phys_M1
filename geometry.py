from __future__ import annotations
from typing import NamedTuple
import numpy as np
from config import PlateConfig, SimulationConfig

class Mesh(NamedTuple):
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    potentials: np.ndarray
    n_plate1: int

def _discretize_face(
    center: np.ndarray,
    u_vec:  np.ndarray,
    v_vec:  np.ndarray,
    normal: np.ndarray,
    lu: float,
    lv: float,
    nu: int,
    nv: int,
    potential: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    du = lu / nu
    dv = lv / nv
    u_offsets = (np.arange(nu) - (nu - 1) / 2) * du
    v_offsets = (np.arange(nv) - (nv - 1) / 2) * dv
    uu, vv = np.meshgrid(u_offsets, v_offsets, indexing='ij')
    uu = uu.ravel()
    vv = vv.ravel()
    n_elem = nu * nv
    centers = center[None, :] + uu[:, None] * u_vec[None, :] + vv[:, None] * v_vec[None, :]
    normals = np.tile(normal, (n_elem, 1))
    areas = np.full(n_elem, du * dv)
    dx_arr = np.full(n_elem, du)
    dy_arr = np.full(n_elem, dv)
    pots = np.full(n_elem, potential)

    return centers, normals, areas, dx_arr, dy_arr, pots

def _discretize_plate(
    plate: PlateConfig,
    z_center: float,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    L, W, T = plate.length, plate.width, plate.thickness
    phi = plate.potential
    a = min(L, W) / n_target
    nL = max(1, round(L / a))
    nW = max(1, round(W / a))
    ex = np.array([1., 0., 0.])
    ey = np.array([0., 1., 0.])
    ez = np.array([0., 0., 1.])
    c = np.array([0., 0., z_center])

    faces = [
        (c + ez * T/2,   ex,    ey,   +ez,   L, W, nL, nW),
        (c - ez * T/2,   ex,    ey,   -ez,   L, W, nL, nW),
    ]

    all_centers = []
    all_normals = []
    all_areas = []
    all_dx = []
    all_dy = []
    all_pots = []

    for fc, u_vec, v_vec, normal, lu, lv, nu, nv in faces:
        cen, nor, ar, dx_, dy_, po = _discretize_face(
            fc, u_vec, v_vec, normal, lu, lv, nu, nv, phi
        )
        all_centers.append(cen)
        all_normals.append(nor)
        all_areas.append(ar)
        all_dx.append(dx_)
        all_dy.append(dy_)
        all_pots.append(po)

    centers = np.vstack(all_centers)
    normals = np.vstack(all_normals)
    areas = np.concatenate(all_areas)
    dx = np.concatenate(all_dx)
    dy = np.concatenate(all_dy)
    pots = np.concatenate(all_pots)

    return centers, normals, areas, dx, dy, pots

def build_mesh(cfg: SimulationConfig) -> Mesh:
    c1, n1, a1, dx1, dy1, p1 = _discretize_plate(
        cfg.plate1,
        z_center=cfg.plate1_z_center,
        n_target=cfg.n_elements,
    )

    c2, n2, a2, dx2, dy2, p2 = _discretize_plate(
        cfg.plate2,
        z_center=cfg.plate2_z_center,
        n_target=cfg.n_elements,
    )

    centers = np.vstack([c1, c2])
    normals = np.vstack([n1, n2])
    areas = np.concatenate([a1, a2])
    dx = np.concatenate([dx1, dx2])
    dy = np.concatenate([dy1, dy2])
    potentials = np.concatenate([p1, p2])
    n_plate1 = len(c1)
    _validate_mesh(cfg, centers, areas, n_plate1)

    return Mesh(
        centers=centers,
        normals=normals,
        areas=areas,
        dx=dx,
        dy=dy,
        potentials=potentials,
        n_plate1=n_plate1,
    )


def _validate_mesh(
    cfg: SimulationConfig,
    centers: np.ndarray,
    areas: np.ndarray,
    n_plate1: int,
) -> None:
    def surface_area(p):
        return 2 * p.length * p.width

    S1_expected = surface_area(cfg.plate1)
    S2_expected = surface_area(cfg.plate2)
    S1_got = float(np.sum(areas[:n_plate1]))
    S2_got = float(np.sum(areas[n_plate1:]))

    tol = 1e-6
    if abs(S1_got - S1_expected) / S1_expected > tol:
        raise ValueError(
            f"Площадь поверхности пластины 1: ожидалось {S1_expected:.6f} м², "
            f"получено {S1_got:.6f} м² (ошибка {abs(S1_got-S1_expected)/S1_expected*100:.4f}%)"
        )
    if abs(S2_got - S2_expected) / S2_expected > tol:
        raise ValueError(
            f"Площадь поверхности пластины 2: ожидалось {S2_expected:.6f} м², "
            f"получено {S2_got:.6f} м² (ошибка {abs(S2_got-S2_expected)/S2_expected*100:.4f}%)"
        )

    z1_min = float(np.min(centers[:n_plate1, 2]))
    z2_max = float(np.max(centers[n_plate1:, 2]))
    if z1_min <= z2_max:
        raise ValueError(
            f"Пластины пересекаются по оси Z: "
            f"min(z1)={z1_min:.4f} м ≤ max(z2)={z2_max:.4f} м"
        )

    if np.any(areas <= 0):
        raise ValueError("Обнаружены элементы с нулевой или отрицательной площадью")
