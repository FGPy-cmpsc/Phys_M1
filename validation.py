from __future__ import annotations
import math
import warnings
from config import SimulationConfig, PlateConfig
from utils import (
    N_ELEMENTS_MIN,
    N_ELEMENTS_MAX,
    FLATNESS_RATIO,
    breakdown_field,
)


class ValidationError(ValueError):
    pass


class ValidationWarning(UserWarning):
    pass


def _warn(msg: str) -> None:
    warnings.warn(msg, ValidationWarning, stacklevel=3)


def _validate_plate(plate: PlateConfig, name: str) -> None:
    if plate.length <= 0:
        raise ValidationError(f"{name}: длина должна быть > 0, получено {plate.length}")
    if plate.width <= 0:
        raise ValidationError(f"{name}: ширина должна быть > 0, получено {plate.width}")
    if plate.thickness <= 0:
        raise ValidationError(f"{name}: толщина должна быть > 0, получено {plate.thickness}")
    if not math.isfinite(plate.potential):
        raise ValidationError(f"{name}: потенциал должен быть конечным числом, получено {plate.potential}")

    min_side = min(plate.length, plate.width)
    max_side = max(plate.length, plate.width)
    if max_side / min_side > 100:
        _warn(
            f"{name}: соотношение сторон {max_side/min_side:.1f} — "
            "очень вытянутая пластина, дискретизация может быть неравномерной"
        )

    if plate.thickness >= min_side:
        raise ValidationError(
            f"{name}: толщина T={plate.thickness*1e3:.1f} мм ≥ min(L,W)={min_side*1e3:.1f} мм"
        )
    if plate.thickness > min_side / FLATNESS_RATIO:
        _warn(
            f"{name}: T/min(L,W) = {plate.thickness/min_side:.2f} > 1/{FLATNESS_RATIO} — "
            "вклад боковых граней может быть заметным"
        )


def validate_config(cfg: SimulationConfig) -> None:
    _validate_plate(cfg.plate1, "Пластина 1")
    _validate_plate(cfg.plate2, "Пластина 2")

    if not math.isfinite(cfg.gap):
        raise ValidationError(f"Зазор должен быть конечным числом, получено {cfg.gap}")
    if cfg.gap <= 0:
        raise ValidationError(f"Зазор между пластинами должен быть > 0, получено {cfg.gap}")
    if cfg.plate1.potential == cfg.plate2.potential:
        raise ValidationError("Потенциалы пластин равны")
    if not math.isfinite(cfg.epsilon_r):
        raise ValidationError(f"epsilon_r должна быть конечным числом, получено {cfg.epsilon_r}")
    if cfg.epsilon_r <= 0:
        raise ValidationError(f"epsilon_r должна быть > 0, получено {cfg.epsilon_r}")
    if cfg.epsilon_r < 1.0 - 1e-9:
        raise ValidationError(f"epsilon_r не может быть меньше 1, получено {cfg.epsilon_r}")
    if cfg.epsilon_r > 1e5:
        _warn(f"epsilon_r = {cfg.epsilon_r:.2e} очень велика")
    if not isinstance(cfg.n_elements, int):
        raise ValidationError(f"n_elements должно быть целым числом, получено {cfg.n_elements}")
    if cfg.n_elements < N_ELEMENTS_MIN:
        raise ValidationError(
            f"n_elements = {cfg.n_elements} слишком мало (минимум {N_ELEMENTS_MIN})"
        )

    n_total_estimate = _estimate_total_elements(cfg)
    if n_total_estimate > N_ELEMENTS_MAX:
        raise ValidationError(
            f"Оценочное суммарное число элементов ~{n_total_estimate} превышает максимум {N_ELEMENTS_MAX}"
        )
    if n_total_estimate < 20:
        _warn(f"Оценочное суммарное число элементов ~{n_total_estimate} очень мало")

    min_element_size = _estimate_min_element_size(cfg)
    if not math.isfinite(min_element_size) or min_element_size <= 0:
        raise ValidationError(
            f"Вычислен некорректный размер элемента ({min_element_size:.4f} м)"
        )
    if min_element_size > cfg.gap:
        raise ValidationError(
            f"Размер элемента (примерно {min_element_size:.4f} м) больше зазора ({cfg.gap:.4f} м)"
        )
    if min_element_size > cfg.gap / 2:
        _warn(
            f"Размер элемента (примерно {min_element_size:.4f} м) больше половины зазора ({cfg.gap:.4f} м)"
        )

    e_approx = abs(cfg.voltage) / cfg.gap
    e_break = breakdown_field(cfg.epsilon_r)
    if e_approx > e_break:
        _warn(
            f"Примерная напряжённость поля E ≈ {e_approx:.2e} В/м превышает "
            f"пробивную ({e_break:.2e} В/м)"
        )

    min_plate_side = min(
        cfg.plate1.length,
        cfg.plate1.width,
        cfg.plate2.length,
        cfg.plate2.width,
    )
    if cfg.gap > min_plate_side:
        _warn(
            f"Зазор ({cfg.gap} м) больше меньшей стороны пластины ({min_plate_side} м)"
        )


def _estimate_min_element_size(cfg: SimulationConfig) -> float:
    min_side = min(
        cfg.plate1.length,
        cfg.plate1.width,
        cfg.plate2.length,
        cfg.plate2.width,
    )
    return min_side / cfg.n_elements


def _estimate_total_elements(cfg: SimulationConfig) -> int:
    def plate_elements(p: PlateConfig, n: int) -> int:
        a = min(p.length, p.width) / n
        return 2 * round(p.length / a) * round(p.width / a)

    return plate_elements(cfg.plate1, cfg.n_elements) + plate_elements(cfg.plate2, cfg.n_elements)
