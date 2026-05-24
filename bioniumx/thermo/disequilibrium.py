import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple

def shomate_equation(T: float, coeffs: List[float]) -> Tuple[float, float, float, float]:
    """
    Computes thermodynamic properties using the Shomate equation based on NIST-JANAF tables.
    
    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    coeffs : List[float]
        List of 7 Shomate coefficients: [A, B, C, D, E, F, G].
        
    Returns
    -------
    Tuple[float, float, float, float]
        (Cp, H, S, dG)
        Cp: Heat capacity (J/(mol*K))
        H: Standard enthalpy (kJ/mol)
        S: Standard entropy (J/(mol*K))
        dG: Standard Gibbs free energy (kJ/mol)
    """
    t = T / 1000.0
    A, B, C, D, E, F, G = coeffs
    
    Cp = A + B*t + C*(t**2) + D*(t**3) + E/(t**2)
    H = A*t + B*(t**2)/2.0 + C*(t**3)/3.0 + D*(t**4)/4.0 - E/t + F
    S = A*np.log(t) + B*t + C*(t**2)/2.0 + D*(t**3)/3.0 - E/(2.0*t**2) + G
    
    # Delta G = H - TS (Converting T*S to kJ/mol)
    dG = H - T * (S / 1000.0)
    
    return Cp, H, S, dG


def gibbs_free_energy_minimization(
    initial_moles: np.ndarray, 
    standard_gibbs: np.ndarray, 
    mass_balance_matrix: np.ndarray,
    element_totals: np.ndarray,
    T: float, 
    P: float
) -> np.ndarray:
    """
    Calculates the chemical equilibrium state by minimizing the total Gibbs Free Energy
    using Sequential Least Squares Programming (SLSQP).
    
    Parameters
    ----------
    initial_moles : np.ndarray
        Array of initial moles for each species.
    standard_gibbs : np.ndarray
        Array of standard Gibbs free energies (dG^circ) for each species at temperature T.
    mass_balance_matrix : np.ndarray
        Matrix defining the elemental composition of each species (rows = elements, cols = species).
    element_totals : np.ndarray
        Total moles of each element in the system.
    T : float
        Temperature in Kelvin.
    P : float
        Pressure in atm.
        
    Returns
    -------
    np.ndarray
        The optimized equilibrium moles for each species.
    """
    R = 8.31446261815324 / 1000.0  # Gas constant in kJ/(mol*K)
    
    def total_gibbs(moles: np.ndarray) -> float:
        total_m = np.sum(moles)
        if total_m <= 0: return np.inf
        
        # Avoid log(0)
        safe_moles = np.clip(moles, 1e-30, None)
        mole_fractions = safe_moles / total_m
        
        # G_i = G_i^circ + RT ln(x_i * P)
        partial_gibbs = standard_gibbs + R * T * np.log(mole_fractions * P)
        return float(np.sum(safe_moles * partial_gibbs))

    # Constraint function: mass_balance_matrix @ moles - element_totals = 0
    def mass_balance(moles: np.ndarray) -> np.ndarray:
        return mass_balance_matrix @ moles - element_totals

    constraints = {'type': 'eq', 'fun': mass_balance}
    
    # Moles must be strictly non-negative
    bounds = [(1e-30, None) for _ in range(len(initial_moles))]

    result = minimize(
        total_gibbs,
        initial_moles,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    return result.x

def calculate_available_gibbs_energy(observed_moles: np.ndarray, equilibrium_moles: np.ndarray, standard_gibbs: np.ndarray, T: float, P: float) -> float:
    """
    Calculates the Available Gibbs Free Energy (J/mol), a metric for chemical disequilibrium.
    
    Parameters
    ----------
    observed_moles : np.ndarray
    equilibrium_moles : np.ndarray
    standard_gibbs : np.ndarray
    T : float
    P : float
    
    Returns
    -------
    float
        Available free energy in Joules.
    """
    R = 8.31446261815324 / 1000.0  # kJ/(mol*K)
    
    def compute_G(moles):
        total = np.sum(moles)
        safe_moles = np.clip(moles, 1e-30, None)
        return np.sum(safe_moles * (standard_gibbs + R * T * np.log((safe_moles/total) * P)))
        
    G_initial = compute_G(observed_moles)
    G_final = compute_G(equilibrium_moles)
    
    # Available energy is the difference (converted to Joules)
    available_energy_kJ = G_initial - G_final
    return float(available_energy_kJ * 1000.0)
