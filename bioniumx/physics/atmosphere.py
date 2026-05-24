"""
Atmospheric physics and thermodynamics.
"""
import numpy as np


def mean_molecular_weight(abundances: dict) -> float:
    """
    Calculate the mean molecular weight of an atmosphere.

    Parameters
    ----------
    abundances : dict
        Dictionary of molecular formulas and their mole fractions.
        E.g., {"H2": 0.85, "He": 0.15}. Fractions will be normalized to sum to 1.

    Returns
    -------
    mu : float
        Mean molecular weight in g/mol (equivalent to amu).

    Examples
    --------
    >>> mu_earth = mean_molecular_weight({"N2": 0.78, "O2": 0.21, "Ar": 0.01})
    """
    # Molecular weights in g/mol
    weights = {
        "H2": 2.016, "He": 4.0026, "H2O": 18.015, "CH4": 16.04,
        "CO2": 44.01, "CO": 28.01, "N2": 28.014, "O2": 31.999,
        "NH3": 17.031, "Ar": 39.948, "O3": 47.998, "N2O": 44.013
    }

    total_frac = sum(abundances.values())
    mu = 0.0

    for mol, frac in abundances.items():
        if mol not in weights:
            raise ValueError(f"Unknown molecule weight for {mol}")
        mu += weights[mol] * (frac / total_frac)

    return float(mu)


def scale_height(T_eq: float, mu: float, gravity: float) -> float:
    """
    Calculate the atmospheric pressure scale height.

    The scale height (H = kT / mg) is the vertical distance over which
    the atmospheric pressure falls by a factor of e.

    Parameters
    ----------
    T_eq : float
        Atmospheric temperature in Kelvin.
    mu : float
        Mean molecular weight in g/mol.
    gravity : float
        Planetary surface gravity in m/s^2.

    Returns
    -------
    H : float
        Scale height in meters.

    Examples
    --------
    >>> H_earth = scale_height(T_eq=255, mu=28.97, gravity=9.81)
    """
    k_B = 1.380649e-23  # Boltzmann constant J/K
    amu = 1.660539e-27  # Atomic mass unit kg

    m_kg = mu * amu
    H = (k_B * T_eq) / (m_kg * gravity)

    return float(H)
