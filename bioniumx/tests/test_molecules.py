import pytest
from bioniumx.molecules import BIOSIGNATURE_MOLECULES, get_template, compute_disequilibrium

def test_catalog_contains_expected():
    assert "H2O" in BIOSIGNATURE_MOLECULES
    assert "CH4" in BIOSIGNATURE_MOLECULES

def test_get_template():
    wl, depth = get_template("H2O")
    assert len(wl) == len(depth)
    assert wl.min() >= BIOSIGNATURE_MOLECULES["H2O"]["wavelength_range"][0]

def test_compute_disequilibrium():
    # Provide highly significant Earth-like signatures
    sigs = {"O2": 5.0, "CH4": 4.0, "H2O": 3.0}
    result = compute_disequilibrium(sigs)
    assert ("O2", "CH4") in result.detected_pairs
    assert result.is_significant()

def test_compute_disequilibrium_insignificant():
    sigs = {"O2": 1.0, "CH4": 1.0}
    result = compute_disequilibrium(sigs)
    assert not result.is_significant()
