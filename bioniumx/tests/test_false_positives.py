"""
Tests for false positive assessment framework.
"""
import pytest
import numpy as np
from bioniumx.assessment import (
    assess_false_positive,
    abiotic_o2_flux,
    check_abiotic_dms,
    check_abiotic_ph3,
) 


class TestAbioticO2Flux:
    """Tests for abiotic O₂ production calculations."""
    
    def test_photolysis_earth_like(self):
        """Earth-like photolysis should produce ~10^-12 mol cm^-2 s^-1."""
        result = abiotic_o2_flux(
            stellar_uv_flux=2e3,
            planet_radius_earth_radii=1.0,
            atmosphere_scale_height=8.5e3,
            mean_molecular_weight=28.97,
            scenario="photolysis",
            water_abundance=0.01,
        )
        
        assert result["scenario"] == "photolysis"
        assert result["o2_flux"] <= 5e-10
        assert result["o2_flux"] > 1e-15
        assert "source_column" in result
    
    def test_runaway_greenhouse(self):
        """Runaway greenhouse scenario should scale with temperature."""
        result_cool = abiotic_o2_flux(
            stellar_uv_flux=2e3,
            planet_radius_earth_radii=1.0,
            atmosphere_scale_height=1e4,
            mean_molecular_weight=18.0,
            scenario="runaway",
            temperature=300,
        )
        
        result_hot = abiotic_o2_flux(
            stellar_uv_flux=2e3,
            planet_radius_earth_radii=1.0,
            atmosphere_scale_height=1.2e4,
            mean_molecular_weight=18.0,
            scenario="runaway",
            temperature=500,
        )
        
        assert result_hot["o2_flux"] > result_cool["o2_flux"]


class TestAbioticDMS:
    """Tests for abiotic DMS feasibility checks."""
    
    def test_earthlike_conditions(self):
        """Earth-like T/P should disfavor abiotic DMS."""
        result = check_abiotic_dms(temperature=288, pressure=1.0, planet_class="earthlike")
        
        assert result["temperature_threshold_ok"] is True
        assert result["pressure_threshold_ok"] is True
        assert result["abiotic_compatible"] is False
        assert result["confidence_in_biosignature"] > 0.8


class TestAbioticPH3:
    """Tests for abiotic PH₃ feasibility checks."""
    
    def test_oxidizing_atmosphere(self):
        """Oxidizing atmosphere (Earth-like O₂) should disfavor abiotic PH₃."""
        result = check_abiotic_ph3(
            temperature=300,
            o2_abundance=0.21,
            atmosphere_type="oxidizing",
        )
        
        assert result["o2_incompatibility"] > 0.5
        assert result["abiotic_compatible"] is False
        assert result["confidence_in_biosignature"] > 0.8


class TestAssessmentFramework:
    """Tests for full 5-level assessment."""
    
    def test_earth_o2_strong_biosignature(self):
        """Earth-like O₂ should be strong biosignature."""
        result = assess_false_positive(
            molecule="O2",
            detection_sigma=5.0,
            measured_abundance=0.21,
            planet_temperature=288,
            planet_pressure=1.0,
            robust_checks_passed=2,
        )
        
        assert result.level_1_detected is True
        assert result.level_2_robust is True
        assert result.confidence_level >= 4
        assert result.level_5_confidence > 0.7
    
    def test_phosphine_on_cool_oxidizing_planet(self):
        """PH₃ on cool oxidizing planet is strong biosignature."""
        result = assess_false_positive(
            molecule="PH3",
            detection_sigma=4.5,
            measured_abundance=20e-9,
            planet_temperature=280,
            planet_pressure=1.0,
            o2_abundance=0.2,
            robust_checks_passed=1,
        )
        
        assert result.level_1_detected is True
        assert result.level_4_context_compatible is True
        assert result.is_biosignature(0.80) is True
    
    def test_low_significance_detection(self):
        """Very low significance detection should fail Level 1."""
        result = assess_false_positive(
            molecule="O2",
            detection_sigma=1.5,
            measured_abundance=0.21,
        )
        
        assert result.level_1_detected is False
        assert result.confidence_level == 0          