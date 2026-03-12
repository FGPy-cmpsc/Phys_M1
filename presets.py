from config import PlateConfig, SimulationConfig

PRESETS: dict[str, SimulationConfig] = {
    "ideal": SimulationConfig(
        plate1=PlateConfig(length=0.10, width=0.10, thickness=0.003, potential=+50.0),
        plate2=PlateConfig(length=0.10, width=0.10, thickness=0.003, potential=-50.0),
        gap=0.008,
        epsilon_r=1.0,
        n_elements=15,
    ),

    "fringe": SimulationConfig(
        plate1=PlateConfig(length=0.05, width=0.05, thickness=0.002, potential=+50.0),
        plate2=PlateConfig(length=0.05, width=0.05, thickness=0.002, potential=-50.0),
        gap=0.05,
        epsilon_r=1.0,
        n_elements=6,
    ),

    "asymmetric": SimulationConfig(
        plate1=PlateConfig(length=0.10, width=0.10, thickness=0.003, potential=+50.0),
        plate2=PlateConfig(length=0.05, width=0.05, thickness=0.003, potential=-50.0),
        gap=0.010,
        epsilon_r=1.0,
        n_elements=8,
    ),

    "dielectric": SimulationConfig(
        plate1=PlateConfig(length=0.10, width=0.10, thickness=0.003, potential=+50.0),
        plate2=PlateConfig(length=0.10, width=0.10, thickness=0.003, potential=-50.0),
        gap=0.008,
        epsilon_r=4.5,
        n_elements=15,
    ),
}
