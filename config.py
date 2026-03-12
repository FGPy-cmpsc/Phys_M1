from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class PlateConfig:
    length: float
    width: float
    thickness: float
    potential: float

@dataclass
class SimulationConfig:
    plate1: PlateConfig
    plate2: PlateConfig
    gap: float
    epsilon_r: float
    n_elements: int

    @property
    def voltage(self) -> float:
        return self.plate1.potential - self.plate2.potential

    @property
    def plate1_z_center(self) -> float:
        return self.gap / 2 + self.plate1.thickness / 2

    @property
    def plate2_z_center(self) -> float:
        return -(self.gap / 2 + self.plate2.thickness / 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SimulationConfig:
        return cls(
            plate1=PlateConfig(**d["plate1"]),
            plate2=PlateConfig(**d["plate2"]),
            gap=d["gap"],
            epsilon_r=d["epsilon_r"],
            n_elements=d["n_elements"],
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> SimulationConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
