"""
False positive assessment framework for biosignatures.

This module helps determine if a detected molecule in an exoplanet's
atmosphere is truly a sign of life or just a natural abiotic process.

We use a 5-level Bayesian approach that checks:
1. Is the molecule really detected?
2. Is the detection solid (multiple observations)?
3. Could non-life produce this amount?
4. Does the planet's conditions rule out non-life production?
5. What's the probability this is actually life?

""" 

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FalsePositiveResult:
    """Results from checking if a biosignature is real or fake."""
    
    molecule: str = ""
    level_1_detected: bool = False
    level_2_robust: bool = False
    level_3_abiotic_flux: float = 0.0
    level_4_context_compatible: bool = False
    level_5_confidence: float = 0.0
    abiotic_scenarios: Dict[str, Dict] = field(default_factory=dict)
    flags: list = field(default_factory=list)
    confidence_level: int = 0
    
    def is_biosignature(self, posterior_threshold: float = 0.90) -> bool:
        """Is this probably a real biosignature?"""
        return self.level_5_confidence >= posterior_threshold


def abiotic_o2_flux(
    stellar_uv_flux: float,
    planet_radius_earth_radii: float,
    atmosphere_scale_height: float,
    mean_molecular_weight: float,
    scenario: str = "photolysis",
    water_abundance: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Dict[str, float]:
    """
    How much oxygen could be made without life?
    
    Two main ways non-life makes oxygen:
    1. UV sunlight breaks apart water/CO2 molecules (photolysis)
    2. Very hot planets lose hydrogen, leaving behind oxygen
    
    Returns how much oxygen we'd expect from non-life processes.
    """
    
    if scenario == "photolysis":
        if water_abundance is None:
            water_abundance = 0.01  # 1% water in atmosphere
        
        # How likely is UV to break water molecules
        sigma_photo = 1e-17  # cm² - technical measure
        
        # How much water is in the atmosphere column
        earth_water_column = 1e19  # molecules per cm²
        column_density = earth_water_column * water_abundance
        
        # Stronger UV = more oxygen produced
        earth_uv = 2e3  # erg cm⁻² s⁻¹
        efficiency = 0.3 * (1.0 - np.exp(-stellar_uv_flux / earth_uv))
        
        # Convert UV energy to photon count
        h_planck = 6.626e-27  # erg·s
        c = 3e10  # cm/s
        photon_flux = stellar_uv_flux / (h_planck * c / 1.5e-5)
        
        # Final oxygen production rate
        o2_flux = (photon_flux * sigma_photo * column_density / 2e19) * efficiency
        o2_flux = float(np.minimum(o2_flux, 5e-10))  # Can't exceed this limit
        
        return {
            "o2_flux": o2_flux,
            "scenario": "photolysis",
            "photo_efficiency": float(efficiency),
            "source_column": float(column_density),
        }
    
    elif scenario == "runaway":
        if temperature is None:
            temperature = 300
        
        # Very hot planets lose hydrogen from atmosphere
        # This leaves oxygen behind
        normal_temp = 300  # K
        temp_factor = np.exp((temperature - normal_temp) / 100.0)
        
        # Hydrogen escape rate goes up with temperature
        h2_loss = 1e9 * temp_factor  # molecules cm⁻² s⁻¹
        
        # Some of the lost hydrogen comes from water
        # That leaves oxygen behind
        efficiency = 0.1 * (stellar_uv_flux / 2e3)
        o2_flux = (h2_loss / 2e18) * efficiency
        o2_flux = float(np.minimum(o2_flux, 1e-9))
        
        return {
            "o2_flux": o2_flux,
            "scenario": "runaway",
            "temperature": float(temperature),
            "h2_escape_flux": float(h2_loss),
        }
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Use 'photolysis' or 'runaway'.")


def check_abiotic_dms(
    temperature: float,
    pressure: float,
    planet_class: str = "earthlike",
) -> Dict[str, any]:
    """
    Can non-life make dimethyl sulfide (DMS)?
    
    DMS is made by ocean bacteria on Earth. But it could also
    be made by chemistry on hot/cold planets.
    
    Returns whether the planet's temperature and pressure
    make non-life DMS production likely.
    """
    
    result = {
        "abiotic_compatible": False,
        "temperature_threshold_ok": False,
        "pressure_threshold_ok": False,
        "safe_margin": 0.0,
        "confidence_in_biosignature": 1.0,
    }
    
    # Set safe ranges for each planet type
    if planet_class == "earthlike":
        t_min, t_max = 250, 350  # K (liquid water range)
        p_min, p_max = 0.1, 10.0  # atm
        
    elif planet_class == "venuslike":
        t_min, t_max = 600, 900  # Very hot
        p_min, p_max = 10.0, 100.0  # High pressure
        
    else: 
        t_min, t_max = 250, 350 
        p_min, p_max = 0.1, 10.0
    
    # Check if planet is in safe range
    temp_ok = t_min <= temperature <= t_max
    result["temperature_threshold_ok"] = temp_ok
    
    press_ok = p_min <= pressure <= p_max
    result["pressure_threshold_ok"] = press_ok
    
    # How close to the middle of safe range?
    t_middle = (t_min + t_max) / 2.0
    t_distance = abs(temperature - t_middle) / ((t_max - t_min) / 2.0)
    t_margin = 1.0 - np.clip(t_distance, 0, 1)
    
    p_middle = np.sqrt(p_min * p_max)
    p_ratio = np.log10(pressure / p_middle) / np.log10(p_max / p_min)
    p_distance = abs(p_ratio)
    p_margin = 1.0 - np.clip(p_distance, 0, 1)
    
    margin = float((t_margin + p_margin) / 2.0)
    result["safe_margin"] = margin
    
    # If outside safe range, non-life DMS is more likely
    result["abiotic_compatible"] = not (temp_ok and press_ok)
    
    # Life DMS is more likely in safe ranges
    if result["abiotic_compatible"]:
        result["confidence_in_biosignature"] = 0.3 * margin
    else:
        result["confidence_in_biosignature"] = 0.9 + 0.1 * margin
    
    return result


def check_abiotic_ph3(
    temperature: float,
    o2_abundance: float,
    atmosphere_type: str = "oxidizing",
) -> Dict[str, any]:
    """
    Can non-life make phosphine (PH3)?
    
    Phosphine + Oxygen = They destroy each other in ~1 year.
    
    So if a planet has BOTH oxygen and phosphine, it's almost
    certainly from life (bacteria make both).
    
    Returns how likely non-life phosphine is.
    """
    
    result = {
        "abiotic_compatible": False,
        "o2_incompatibility": 0.0,
        "thermodynamic_barrier": 0.0,
        "confidence_in_biosignature": 0.5,
    }
    
    # Oxygen destroys phosphine fast
    if atmosphere_type == "oxidizing":
        o2_kill_rate = 1.0 - np.exp(-o2_abundance * 100)
        result["o2_incompatibility"] = float(np.clip(o2_kill_rate, 0, 1))
    else:
        # Low oxygen doesn't kill phosphine as fast
        o2_kill_rate = 0.1 * o2_abundance
        result["o2_incompatibility"] = float(o2_kill_rate)
    
    # Temperature affects how hard it is to make phosphine
    if temperature < 500:
        # Cold planets: almost impossible to make phosphine
        barrier = 0.95
    elif temperature < 1000:
        # Medium temp: harder to make
        barrier = 0.7 - 0.0003 * (temperature - 500)
    else:
        # Very hot planets: possible but extreme
        barrier = 0.3 + 0.0001 * (temperature - 1000)
    
    result["thermodynamic_barrier"] = float(np.clip(barrier, 0, 1))
    
    # Combined: how likely is non-life phosphine?
    non_life_chance = (1 - result["o2_incompatibility"]) * (1 - result["thermodynamic_barrier"])
    result["abiotic_compatible"] = non_life_chance > 0.3
    
    # Life phosphine is likely if non-life is unlikely
    result["confidence_in_biosignature"] = float(np.clip(1.0 - non_life_chance * 0.5, 0, 1))
    
    return result


def assess_false_positive(
    molecule: str,
    detection_sigma: float,
    measured_abundance: float,
    measured_flux: Optional[float] = None,
    planet_temperature: float = 288,
    planet_pressure: float = 1.0,
    stellar_uv_flux: float = 2e3,
    planet_radius_earth_radii: float = 1.0,
    scale_height: float = 8.5e3,
    mean_mw: float = 28.97,
    o2_abundance: Optional[float] = None,
    robust_checks_passed: int = 1,
    user_prior_biotic: Optional[float] = None,
) -> FalsePositiveResult:
    """
    The main function: Is this detected molecule really life?
    
    We check 5 levels:
    1. Is it really detected? (above noise)
    2. Is the detection confirmed? (multiple observations)
    3. Is there too much for non-life? (exceeds abiotic limit)
    4. Does the planet rule out non-life? (T/P checks)
    5. What's the final probability it's life? (Bayesian math)
    """
    
    result = FalsePositiveResult(molecule=molecule)
    
    # LEVEL 1: Is the molecule actually detected?
    detection_threshold = 2.0  # Need 2 sigma or more
    result.level_1_detected = detection_sigma >= detection_threshold
    
    if not result.level_1_detected:
        result.confidence_level = 0
        msg = f"Detection only {detection_sigma:.2f}σ (need {detection_threshold}σ)"
        result.flags.append(f"Level 1 FAILED: {msg}")
        return result
    
    result.flags.append(f"Level 1 PASSED: Detected at {detection_sigma:.2f}σ")
    

    # LEVEL 2: Is the detection solid?
    result.level_2_robust = robust_checks_passed >= 1
    
    if not result.level_2_robust:
        result.confidence_level = 1
        result.flags.append(f"Level 2 CAUTION: Only {robust_checks_passed} check(s) confirm it")
    else:
        result.flags.append(f"Level 2 PASSED: {robust_checks_passed} confirmation(s)")
    

    # LEVEL 3: Is there too much for non-life?
    abiotic_scenarios = {}
    abiotic_limit = 1e-12  # Default non-life limit
    
    # Calculate how much non-life would produce
    if molecule == "O2":
        photo = abiotic_o2_flux(
            stellar_uv_flux=stellar_uv_flux,
            planet_radius_earth_radii=planet_radius_earth_radii,
            atmosphere_scale_height=scale_height,
            mean_molecular_weight=mean_mw,
            scenario="photolysis",
        )
        abiotic_scenarios["photolysis"] = photo
        
        runaway = abiotic_o2_flux(
            stellar_uv_flux=stellar_uv_flux,
            planet_radius_earth_radii=planet_radius_earth_radii,
            atmosphere_scale_height=scale_height,
            mean_molecular_weight=mean_mw,
            scenario="runaway",
            temperature=planet_temperature,
        )
        abiotic_scenarios["runaway"] = runaway
        abiotic_limit = max(photo["o2_flux"], runaway["o2_flux"])
    
    elif molecule == "PH3":
        ph3_check = check_abiotic_ph3(
            temperature=planet_temperature,
            o2_abundance=o2_abundance if o2_abundance is not None else 0.21,
        )
        abiotic_scenarios["ph3_context"] = ph3_check
        # Very hard to make PH3 with oxygen
        abiotic_limit = 1e-15 if ph3_check["o2_incompatibility"] > 0.8 else 1e-12
    
    elif molecule == "DMS":
        dms_check = check_abiotic_dms(
            temperature=planet_temperature,
            pressure=planet_pressure,
        )
        abiotic_scenarios["dms_context"] = dms_check
        abiotic_limit = 1e-9 if dms_check["abiotic_compatible"] else 1e-11
    
    # Calculate the measured amount 
    if measured_flux is None:
        # Convert mixing ratio to flux
        # Higher abundance = higher flux
        measured_flux = measured_abundance * 1e-7
    
    # How many times the non-life limit is our measurement?
    result.level_3_abiotic_flux = float(measured_flux / (abiotic_limit + 1e-20))
    result.abiotic_scenarios = abiotic_scenarios
    
    if result.level_3_abiotic_flux > 1.0:
        msg = f"Measured is {result.level_3_abiotic_flux:.1f}× what non-life makes"
        result.flags.append(f"Level 3 PASSED: {msg}")
    else:
        msg = f"Only {result.level_3_abiotic_flux:.2f}× non-life limit"
        result.flags.append(f"Level 3 CAUTION: {msg}")
    

    # LEVEL 4: Does the planet's conditions rule out non-life?
    context_compatible = False
    
    if molecule == "O2":
        # Hot/dry planets: non-life can make O2 (hydrogen loss)
        # Cool/wet planets: life makes O2 (photosynthesis)
        if planet_temperature > 350:
            context_compatible = True
            result.flags.append("Level 4 PASSED: Too hot for normal life, non-life O2 ruled out")
        else:
            context_compatible = False
            result.flags.append("Level 4 CAUTION: Cool enough for photosynthesis (non-life still possible)")
    
    elif molecule == "PH3":
        # Oxygen in atmosphere means phosphine must be life
        o2_val = o2_abundance if o2_abundance is not None else 0.21
        if o2_val > 0.01:
            context_compatible = True
            result.flags.append("Level 4 PASSED: With oxygen, phosphine must be from life")
        else:
            context_compatible = False
            result.flags.append("Level 4 CAUTION: Low oxygen, non-life phosphine possible")
    
    elif molecule == "DMS":
        dms_check = abiotic_scenarios.get("dms_context", {})
        if not dms_check.get("abiotic_compatible", False):
            context_compatible = True
            result.flags.append("Level 4 PASSED: Conditions favor life DMS")
        else:
            context_compatible = False
            result.flags.append("Level 4 CAUTION: Conditions could allow non-life DMS")
    
    else:
        context_compatible = True
    
    result.level_4_context_compatible = context_compatible
    
   
    # LEVEL 5: Bayesian Final Calculation
    
    # Starting belief: how likely is life?
    if 250 < planet_temperature < 350 and 0.1 < planet_pressure < 10:
        life_likely = 0.15  # 15% in habitable zone
    else:
        life_likely = 0.01  # 1% outside habitable zone
    
    if user_prior_biotic is not None:
        life_likely = user_prior_biotic
    
    # Strong detection favors life
    detection_likelihood = 1.0 - np.exp(-detection_sigma / 2.0)
    
    # How likely is this much molecule if it's non-life?
    # If measured is much LESS than non-life limit, it's likely non-life
    # If measured is much MORE than non-life limit, it's likely life
    ratio = measured_flux / (abiotic_limit + 1e-20)
    if ratio > 1:
        # More than non-life would make
        non_life_likelihood = 0.01  # Very unlikely to be non-life
    else:
        # Less than non-life would make - could be non-life
        non_life_likelihood = 0.5 * (abiotic_limit / (measured_flux + 1e-20))
        non_life_likelihood = np.minimum(non_life_likelihood, 1.0)
    
    # Combine all the evidence
    flux_evidence = np.tanh(result.level_3_abiotic_flux)
    context_evidence = 0.95 if context_compatible else 0.2
    
    life_likelihood = detection_likelihood * flux_evidence * context_evidence
    
    # Bayesian formula: update our belief
    numerator = life_likelihood * life_likely
    denominator = numerator + non_life_likelihood * (1 - life_likely) + 1e-10
    
    final_belief = numerator / denominator
    result.level_5_confidence = float(np.clip(final_belief, 0, 1))
    
  
    # Final verdict
    if result.level_5_confidence > 0.9:
        result.confidence_level = 5
        result.flags.append("=" * 60)
        result.flags.append(f"FINAL: Very likely LIFE (P={result.level_5_confidence:.1%})")
    elif result.level_5_confidence > 0.75:
        result.confidence_level = 4
        result.flags.append("=" * 60)
        result.flags.append(f"FINAL: Probably LIFE (P={result.level_5_confidence:.1%})")
    elif result.level_5_confidence > 0.5:
        result.confidence_level = 3
        result.flags.append("=" * 60)
        result.flags.append(f"FINAL: Uncertain (P={result.level_5_confidence:.1%})")
    else:
        result.confidence_level = 2
        result.flags.append("=" * 60)
        result.flags.append(f"FINAL: Probably NOT LIFE (P={result.level_5_confidence:.1%})")
    
    return result  