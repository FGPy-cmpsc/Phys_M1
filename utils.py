EPSILON_0 = 8.854187817e-12
E_BREAKDOWN_AIR = 3e6
E_BREAKDOWN_DIELECTRIC = 1e7
N_ELEMENTS_MIN = 3
N_ELEMENTS_MAX = 3000
FLATNESS_RATIO = 10

def breakdown_field(epsilon_r: float) -> float:
    if abs(epsilon_r - 1.0) < 0.1:
        return E_BREAKDOWN_AIR
    return E_BREAKDOWN_DIELECTRIC
