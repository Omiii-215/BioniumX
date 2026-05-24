import numpy as np

# Physical constants
K_B = 1.380649e-23  # Boltzmann constant in J/K
AMU = 1.66053906660e-27  # Atomic mass unit in kg

def calculate_scale_height(T: float, mu: float, g: float) -> float:
    """
    Calculates the atmospheric scale height (H) of an exoplanet.
    
    H = (k_B * T) / (mu * g)
    
    Parameters
    ----------
    T : float
        Atmospheric temperature in Kelvin.
    mu : float
        Mean molecular weight in atomic mass units (AMU).
    g : float
        Acceleration due to gravity in m/s^2.
        
    Returns
    -------
    float
        Scale height in meters.
    """
    mass_kg = mu * AMU
    return (K_B * T) / (mass_kg * g)


def calculate_transit_depth_variation(R_p: float, R_star: float, H: float, N_scale_heights: float = 5.0) -> float:
    """
    Calculates the approximate transit depth variation (delta delta) caused by an atmospheric feature.
    
    Delta delta approx = (2 * R_p * N_sh * H) / (R_star^2)
    
    Parameters
    ----------
    R_p : float
        Planetary radius in meters.
    R_star : float
        Stellar radius in meters.
    H : float
        Atmospheric scale height in meters.
    N_scale_heights : float, default 5.0
        The typical number of scale heights a strong absorption feature spans.
        
    Returns
    -------
    float
        The variation in transit depth (fractional area, dimensionless).
        Multiply by 1e6 to get parts-per-million (ppm).
    """
    return (2.0 * R_p * N_scale_heights * H) / (R_star**2)

def calculate_chord_optical_depth(sigma: np.ndarray, n_profile: np.ndarray, dr: float) -> np.ndarray:
    """
    Computes the monochromatic chord optical depth (tau) along a line-of-sight.
    
    tau = sum(sigma * n * dr)
    
    Parameters
    ----------
    sigma : np.ndarray
        Wavelength-dependent absorption cross-sections (m^2/molecule).
    n_profile : np.ndarray
        Number density profile of the absorbing species along the chord (molecules/m^3).
    dr : float
        Path length step size in meters.
        
    Returns
    -------
    np.ndarray
        Wavelength-dependent optical depth.
    """
    # Assuming sigma shape is (n_wavelengths,) and n_profile shape is (n_steps,)
    # Total column density N = sum(n_profile * dr)
    column_density = np.sum(n_profile) * dr
    return sigma * column_density
