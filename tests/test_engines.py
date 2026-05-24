import numpy as np
import pytest
from bioniumx.habitability.indices import calculate_esi
from bioniumx.thermo.disequilibrium import shomate_equation
from bioniumx.spectra.transfer import calculate_scale_height, calculate_transit_depth_variation

def test_esi_calculation():
    # ESI for Earth should be exactly 1.0
    esi_earth = calculate_esi(flux=1.0, radius=1.0)
    assert np.isclose(esi_earth, 1.0), f"Expected 1.0, got {esi_earth}"
    
    # ESI for a totally different planet
    esi_other = calculate_esi(flux=2.0, radius=2.0)
    assert esi_other < 1.0

def test_shomate_equation():
    # H2O gas at 298.15 K (NIST values)
    # Coeffs for H2O(g) 298. - 1700. K
    coeffs_H2O = [
        30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967
    ]
    T = 298.15
    Cp, H, S, dG = shomate_equation(T, coeffs_H2O)
    
    # Check physical plausibility
    assert Cp > 0, "Heat capacity must be positive"
    assert S > 0, "Entropy must be positive"
    # H2O(g) standard enthalpy of formation is approx -241.8 kJ/mol
    assert np.isclose(H, -241.8, atol=1.0)

def test_scale_height():
    # Earth-like atmosphere: T=288 K, mu=28.97 (air), g=9.81
    H_earth = calculate_scale_height(T=288.0, mu=28.97, g=9.81)
    
    # Scale height for Earth is ~8.5 km
    assert 8000 < H_earth < 9000, f"Expected ~8500m, got {H_earth}"

def test_transit_depth_variation():
    # Jupiter crossing the Sun
    R_jup = 6.99e7
    R_sun = 6.96e8
    H_jup = 27000 # ~27km scale height
    
    delta_ppm = calculate_transit_depth_variation(R_jup, R_sun, H_jup, N_scale_heights=5) * 1e6
    assert delta_ppm > 0
