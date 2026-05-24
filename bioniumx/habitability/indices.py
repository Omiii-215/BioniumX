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
