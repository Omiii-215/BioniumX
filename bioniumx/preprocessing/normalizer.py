"""
Continuum normalization for transmission and emission spectra.
"""
import numpy as np
from bioniumx.core import BioniumXObject
from bioniumx.spectra import TransmissionSpectrum, EmissionSpectrum


def continuum_normalize(spectrum: BioniumXObject, method: str = "polynomial", degree: int = 2):
    """
    Normalize a spectrum by dividing by its continuum.

    The continuum represents the baseline flux/depth devoid of absorption features.
    This function fits a model (e.g., polynomial) to the pseudo-continuum points
    and divides the entire spectrum by this model.

    Parameters
    ----------
    spectrum : BioniumXObject
        The input spectrum.
    method : str, optional
        Method for fitting the continuum. Currently supports "polynomial" or "mean".
        Default is "polynomial".
    degree : int, optional
        Degree of the polynomial if method is "polynomial". Default is 2.

    Returns
    -------
    normalized : BioniumXObject
        A new spectrum object normalized to its continuum.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Examples
    --------
    >>> spec_norm = continuum_normalize(spec, method="polynomial", degree=2)
    """
    if hasattr(spectrum, 'transit_depth'):
        y = spectrum.transit_depth
    elif hasattr(spectrum, 'flux'):
        y = spectrum.flux
    else:
        raise TypeError("Object must have either 'transit_depth' or 'flux' attribute.")

    x = spectrum.wavelength

    if method == "mean":
        continuum = np.ones_like(y) * np.mean(y)
    elif method == "polynomial":
        # Fit polynomial to the entire spectrum (in practice, one should mask lines)
        coeffs = np.polyfit(x, y, degree)
        continuum = np.polyval(coeffs, x)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Avoid division by zero
    continuum = np.where(continuum == 0, 1.0, continuum)
    y_norm = y / continuum

    # Error propagation
    err_norm = spectrum.err / continuum

    if isinstance(spectrum, TransmissionSpectrum):
        return TransmissionSpectrum(
            wavelength=x,
            transit_depth=y_norm,
            err=err_norm,
            **spectrum.meta
        )
    elif isinstance(spectrum, EmissionSpectrum):
        return EmissionSpectrum(
            wavelength=x,
            flux=y_norm,
            err=err_norm,
            **spectrum.meta
        )
    else:
        raise TypeError("Spectrum must be TransmissionSpectrum or EmissionSpectrum.")
