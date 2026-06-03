import numpy as np
from bioniumx.physics import equilibrium_temperature, habitable_zone_bounds, habitability_score, mean_molecular_weight, scale_height

def test_equilibrium_temperature():
    # Earth analog: T_star=5778, R_star=1.0, a=1.0
    T_eq = equilibrium_temperature(5778, 1.0, 1.0)
    assert 250 < T_eq < 260  # ~255K

def test_habitable_zone_bounds():
    inner, outer = habitable_zone_bounds(5778, 1.0)
    assert inner < 1.0
    assert outer > 1.0

def test_habitability_score():
    score = habitability_score(255, 1.0)
    assert score > 0.9  # Earth is highly habitable

def test_habitability_score_rejects_non_physical_catalog_values():
    assert habitability_score(255, 0.0) == 0.0
    assert habitability_score(255, -1.0) == 0.0
    assert habitability_score(255, np.nan) == 0.0
    assert habitability_score(0.0, 1.0) == 0.0

def test_mean_molecular_weight():
    mu = mean_molecular_weight({"H2": 0.85, "He": 0.15})
    assert 2.0 < mu < 3.0

def test_scale_height():
    H = scale_height(255, 28.97, 9.81)
    assert 7000 < H < 9000  # ~8km for Earth
