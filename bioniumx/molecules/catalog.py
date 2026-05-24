"""
Biosignature molecule catalog and templates.
"""
import numpy as np


BIOSIGNATURE_MOLECULES = {
    "H2O": {"type": "solvent/habitability", "wavelength_range": (0.9, 3.0)},
    "CH4": {"type": "biosignature/methanogenesis", "wavelength_range": (1.6, 3.5)},
    "CO2": {"type": "background/habitability", "wavelength_range": (2.0, 4.5)},
    "O2": {"type": "biosignature/photosynthesis", "wavelength_range": (0.7, 1.3)},
    "O3": {"type": "biosignature/photochemical", "wavelength_range": (9.0, 10.0)},
    "N2O": {"type": "biosignature/denitrification", "wavelength_range": (3.8, 4.6)},
    "NH3": {"type": "biosignature/cold-planets", "wavelength_range": (10.0, 11.0)},
    "PH3": {"type": "biosignature/reducing", "wavelength_range": (4.0, 4.5)},
}


def get_template(molecule: str, resolving_power: float = 100):
    """
    Get a theoretical transmission/emission template for a molecule.

    This function uses the RADIS library to query the Harvard HITRAN
    database in real-time, computing the Voigt-broadened high-resolution
    cross section at typical exoplanetary conditions (T=1000K).

    Parameters
    ----------
    molecule : str
        The chemical formula (e.g., 'H2O', 'CH4', 'CO2').
    resolving_power : float, optional
        The spectral resolving power R = λ/Δλ of the requested template. Default 100.

    Returns
    -------
    wavelength : np.ndarray
        Template wavelength grid (microns).
    depth : np.ndarray
        Template absorbance cross-section.

    Raises
    ------
    ValueError
        If the molecule is not in the catalog.
    """
    if molecule not in BIOSIGNATURE_MOLECULES:
        raise ValueError(f"Molecule {molecule} not found in catalog.")

    wmin, wmax = BIOSIGNATURE_MOLECULES[molecule]["wavelength_range"]

    # Generate a logarithmic wavelength grid based on resolving power
    n_points = int(resolving_power * np.log(wmax / wmin))
    wl_grid = np.geomspace(wmin, wmax, n_points)

    try:
        from radis import calc_spectrum
    except ImportError:
        raise ImportError(
            "The 'radis' package is required to fetch real HITRAN physics. "
            "Install it via `pip install bionium-x[science]` or `pip install radis`."
        )

    # Convert wavelength bounds from microns to wavenumbers (cm^-1)
    # nu (cm^-1) = 10000 / lambda (um)
    nu_min = 10000.0 / wmax
    nu_max = 10000.0 / wmin

    print(f"RADIS: Fetching high-resolution HITRAN data for {molecule}...")
    # Calculate the spectrum using Harvard's HITRAN database
    # Tgas = 1000 K (typical Hot Jupiter/sub-Neptune)
    s = calc_spectrum(
        wavenum_min=nu_min,
        wavenum_max=nu_max,
        molecule=molecule,
        isotope="1",
        pressure=0.1,  # bar (typical transit probing pressure)
        Tgas=1000.0,   # K
        databank="hitran",
        truncation=5.0,
        neighbour_lines=5.0,
        warnings={"AccuracyError": "ignore", "MissingSelfBroadeningWarning": "ignore"}
    )

    # Retrieve the computed spectrum arrays
    # We want wavelength (nm) and absorbance, then convert nm to um
    wl_radis, absorbance = s.get("absorbance", wunit="nm")
    wl_radis = wl_radis / 1000.0

    # Interpolate the ultra-high-resolution RADIS spectrum down to our requested resolving power
    depth_interp = np.interp(wl_grid, wl_radis, absorbance)

    # Normalize to 0-1 for template cross-correlation
    if np.max(depth_interp) > 0:
        depth_interp = depth_interp / np.max(depth_interp)

    return wl_grid, depth_interp
