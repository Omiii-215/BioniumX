"""
Smoothing filters for transmission and emission spectra.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from bioniumx.core import BioniumXObject
from bioniumx.spectra import TransmissionSpectrum, EmissionSpectrum


def savitzky_golay(spectrum: BioniumXObject, window: int = 11, polyorder: int = 3):
    """
    Apply a Savitzky-Golay filter to smooth a spectrum.

    A Savitzky-Golay filter smooths data by fitting successive sub-sets
    of adjacent data points with a low-degree polynomial. It is excellent
    for preserving line shapes and peaks while reducing high-frequency noise.

    Parameters
    ----------
    spectrum : BioniumXObject
        The input spectrum (TransmissionSpectrum or EmissionSpectrum).
    window : int, optional
        The length of the filter window. Must be a positive odd integer. Default 11.
    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less than window. Default 3.

    Returns
    -------
    smoothed : BioniumXObject
        A new spectrum object of the same type with smoothed data.

    Raises
    ------
    ValueError
        If window is even or if polyorder >= window.

    Examples
    --------
    >>> spec = TransmissionSpectrum.read("data.h5")
    >>> smoothed_spec = savitzky_golay(spec, window=15, polyorder=3)
    """
    if window % 2 == 0:
        raise ValueError("Window length must be an odd integer.")
    if polyorder >= window:
        raise ValueError("polyorder must be less than window length.")

    if isinstance(spectrum, TransmissionSpectrum):
        smoothed_data = savgol_filter(spectrum.transit_depth, window, polyorder)
        return TransmissionSpectrum(
            wavelength=spectrum.wavelength,
            transit_depth=smoothed_data,
            err=spectrum.err,
            **spectrum.meta
        )
    elif isinstance(spectrum, EmissionSpectrum):
        smoothed_data = savgol_filter(spectrum.flux, window, polyorder)
        return EmissionSpectrum(
            wavelength=spectrum.wavelength,
            flux=smoothed_data,
            err=spectrum.err,
            **spectrum.meta
        )
    else:
        raise TypeError("Spectrum must be TransmissionSpectrum or EmissionSpectrum.")


def gaussian_smooth(spectrum: BioniumXObject, sigma: float = 2.0):
    """
    Apply a 1D Gaussian filter to smooth a spectrum.

    Parameters
    ----------
    spectrum : BioniumXObject
        The input spectrum.
    sigma : float, optional
        Standard deviation for Gaussian kernel. Default 2.0.

    Returns
    -------
    smoothed : BioniumXObject
        A new spectrum object with smoothed data.
    """
    if isinstance(spectrum, TransmissionSpectrum):
        smoothed_data = gaussian_filter1d(spectrum.transit_depth, sigma)
        return TransmissionSpectrum(
            wavelength=spectrum.wavelength,
            transit_depth=smoothed_data,
            err=spectrum.err,
            **spectrum.meta
        )
    elif isinstance(spectrum, EmissionSpectrum):
        smoothed_data = gaussian_filter1d(spectrum.flux, sigma)
        return EmissionSpectrum(
            wavelength=spectrum.wavelength,
            flux=smoothed_data,
            err=spectrum.err,
            **spectrum.meta
        )
    else:
        raise TypeError("Spectrum must be TransmissionSpectrum or EmissionSpectrum.")
