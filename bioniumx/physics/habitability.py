"""
Habitability and equilibrium temperature physics.
"""
import numpy as np


def equilibrium_temperature(
    T_star: float,
    R_star: float,
    a: float,
    albedo: float = 0.3,
    emissivity: float = 1.0,
) -> float:
    """
    Calculate the equilibrium temperature of an exoplanet.

    Assumes the planet is a rapid rotator (heat is redistributed globally).

    Parameters
    ----------
    T_star : float
        Stellar effective temperature in Kelvin.
    R_star : float
        Stellar radius in Solar radii (R_sun).
    a : float
        Semi-major axis of the planet in AU.
    albedo : float, optional
        Bond albedo of the planet (0 to 1). Default is 0.3 (Earth-like).
    emissivity : float, optional
        Thermal emissivity of the planet. Default 1.0 (blackbody).

    Returns
    -------
    T_eq : float
        The planetary equilibrium temperature in Kelvin.

    Examples
    --------
    >>> T_eq = equilibrium_temperature(T_star=5778, R_star=1.0, a=1.0)
    >>> print(f"Earth T_eq = {T_eq:.1f} K")
    """
    # Constants
    R_sun_m = 6.957e8
    AU_m = 1.496e11

    # Convert R_star to meters and a to meters
    R_star_m = R_star * R_sun_m
    a_m = a * AU_m

    # Energy balance: L_in = L_out
    # T_eq = T_star * sqrt(R_star / (2 * a)) * ( (1 - A) / e )^(1/4)
    # The factor of 2 in the denominator assumes global heat redistribution.
    term1 = T_star * np.sqrt(R_star_m / (2.0 * a_m))
    term2 = ((1.0 - albedo) / emissivity) ** 0.25

    return float(term1 * term2)


def habitable_zone_bounds(T_star: float, L_star: float) -> tuple:
    """
    Calculate the conservative habitable zone boundaries.

    Uses the Kopparapu et al. (2013) parametric equations for the
    Recent Venus (inner edge) and Early Mars (outer edge) limits.

    Parameters
    ----------
    T_star : float
        Stellar effective temperature in Kelvin.
    L_star : float
        Stellar luminosity in Solar luminosities (L_sun).

    Returns
    -------
    hz_inner, hz_outer : float
        Inner and outer boundaries of the habitable zone in AU.

    References
    ----------
    Kopparapu, R. K. et al. (2013), ApJ, 765, 131.

    Examples
    --------
    >>> inner, outer = habitable_zone_bounds(T_star=5778, L_star=1.0)
    """
    # Kopparapu coefficients for Recent Venus (inner) and Early Mars (outer)
    # Format: S_eff_sun, a, b, c, d
    coeff_inner = (1.776, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15)
    coeff_outer = (0.320, 5.4471e-5, 1.5275e-9, -1.1746e-12, -1.7511e-16)

    T_diff = T_star - 5780.0

    def calc_seff(c):
        seff0, a, b, c_val, d = c
        return seff0 + a * T_diff + b * T_diff**2 + c_val * T_diff**3 + d * T_diff**4

    S_eff_inner = calc_seff(coeff_inner)
    S_eff_outer = calc_seff(coeff_outer)

    # d = sqrt(L / S_eff)
    hz_inner = np.sqrt(L_star / S_eff_inner)
    hz_outer = np.sqrt(L_star / S_eff_outer)

    return float(hz_inner), float(hz_outer)


def habitability_score(T_eq: float, radius_Rearth: float, mass_Mearth: float = None) -> float:
    """
    Compute a heuristic Earth Similarity / Habitability Score.

    Combines the Earth Similarity Index (ESI) formulation with constraints
    on planet radius to heavily penalize gas giants and freezing/boiling worlds.

    Parameters
    ----------
    T_eq : float
        Planetary equilibrium temperature in K.
    radius_Rearth : float
        Planetary radius in Earth radii.
    mass_Mearth : float, optional
        Planetary mass in Earth masses. If None, estimated from radius.

    Returns
    -------
    score : float
        Habitability score between 0 and 1. Non-physical catalog values such
        as zero or negative radius return 0.0.
    """
    T_earth = 255.0  # Earth's equilibrium temp (albedo 0.3)
    R_earth = 1.0

    if not np.isfinite(T_eq) or not np.isfinite(radius_Rearth):
        return 0.0
    if T_eq <= 0.0 or radius_Rearth <= 0.0:
        return 0.0

    if mass_Mearth is None:
        # Simple M-R relation for rocky worlds (M ~ R^3.7)
        mass_Mearth = radius_Rearth ** 3.7

    # Earth Similarity Index (ESI) terms
    w_T = 5.58  # Weight for temperature
    w_R = 0.57  # Weight for radius

    esi_T = (1.0 - abs((T_eq - T_earth) / (T_eq + T_earth))) ** w_T
    esi_R = (1.0 - abs((radius_Rearth - R_earth) / (radius_Rearth + R_earth))) ** w_R

    base_esi = np.sqrt(esi_T * esi_R)

    # Penalty for definitely non-rocky planets (Radius > 1.6 usually means volatile envelope)
    rocky_probability = 1.0
    if radius_Rearth > 1.6:
        rocky_probability = np.exp(-2.0 * (radius_Rearth - 1.6))

    return float(max(0.0, min(1.0, base_esi * rocky_probability)))
