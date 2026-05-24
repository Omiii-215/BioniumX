import numpy as np

def calculate_esi(flux: np.ndarray, radius: np.ndarray, flux_earth: float = 1.0, radius_earth: float = 1.0) -> np.ndarray:
    """
    Computes the physical Earth Similarity Index (ESI) across numpy arrays.
    
    The ESI provides a scale from 0 to 1, quantifying how closely an exoplanet's
    bulk physical properties resemble those of Earth.
    
    Parameters
    ----------
    flux : np.ndarray or float
        The incident stellar flux on the exoplanet.
    radius : np.ndarray or float
        The planetary radius.
    flux_earth : float, default 1.0
        The reference flux of Earth (normalized to 1).
    radius_earth : float, default 1.0
        The reference radius of Earth (normalized to 1).
        
    Returns
    -------
    np.ndarray or float
        The calculated ESI value(s) from 0 to 1.
    """
    flux_term = ((flux - flux_earth) / (flux + flux_earth)) ** 2
    rad_term = ((radius - radius_earth) / (radius + radius_earth)) ** 2
    
    return 1.0 - 0.5 * (flux_term + rad_term)


def calculate_sephi_telluricity(mass_mean: float, mass_std: float, radius_mean: float, radius_std: float) -> float:
    """
    Calculates a statistical likelihood of the planet possessing a rocky composition (Telluricity)
    using Gaussian probability based on mass and radius.
    
    This is a stub function for the Statistical-likelihood Exo-Planetary Habitability Index (SEPHI).
    
    Parameters
    ----------
    mass_mean : float
        Mean planetary mass (in Earth masses).
    mass_std : float
        Standard deviation of the mass measurement.
    radius_mean : float
        Mean planetary radius (in Earth radii).
    radius_std : float
        Standard deviation of the radius measurement.
        
    Returns
    -------
    float
        The probability [0, 1] that the planet is telluric (rocky).
    """
    from scipy.stats import norm
    
    # A simple probabilistic model: terrestrial planets usually have R < 1.6 R_earth and M < 3 M_earth
    # We integrate the joint probability distribution over the telluric domain
    
    # Probability that Radius < 1.6
    p_radius = norm.cdf(1.6, loc=radius_mean, scale=radius_std)
    
    # Probability that Mass < 3.0
    p_mass = norm.cdf(3.0, loc=mass_mean, scale=mass_std)
    
    # Joint independent probability
    return p_radius * p_mass
def calculate_biosignature_score(probas):
    """
    Calculate the heuristic Biosignature Score based on molecule probabilities.
    Input probas: dict mapping molecule name to probability [0.0 - 1.0]
    e.g. {'O2': 0.81, 'CH4': 0.75, 'O3': 0.1, 'H2O': 0.9, 'CO2': 0.5}

    Returns a score from 0.0 to 1.0 and a confidence category.
    """
    p_O2 = probas.get('O2', 0.0)
    p_CH4 = probas.get('CH4', 0.0)
    p_O3 = probas.get('O3', 0.0)
    p_H2O = probas.get('H2O', 0.0)

    # Base score is an average of the crucial life markers (O2, CH4, H2O)
    base_score = (p_O2 + p_CH4 + p_H2O) / 3.0

    # Bonus for Earth-like thermodynamic disequilibrium
    # (co-existence of O2/O3 and CH4)
    # They rapidly destroy each other, so co-presence means active
    # replenishment (life)
    disequilibrium_bonus = 0.0
    if (p_O2 > 0.5 or p_O3 > 0.5) and p_CH4 > 0.5:
        # Strong bonus
        disequilibrium_bonus = 0.3 * min(max(p_O2, p_O3), p_CH4)

    # Penalty if water is absent
    water_penalty = 0.0
    if p_H2O < 0.2:
        water_penalty = -0.2

    final_score = base_score + disequilibrium_bonus + water_penalty

    # Cap between 0 and 1
    final_score = max(0.0, min(1.0, final_score))

    # Determine confidence Category
    if final_score > 0.75:
        confidence = "HIGH"
    elif final_score > 0.4:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return final_score, confidence
