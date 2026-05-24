"""
TransmissionSpectrum — Core data class for exoplanet transmission spectroscopy.
"""
import numpy as np
from bioniumx.core import BioniumXObject

class TransmissionSpectrum(BioniumXObject):
    """
    Represents a transmission spectrum of an exoplanet atmosphere.

    Stores wavelength grid, transit depth (Rp/Rs)², and uncertainties,
    plus all mission/target metadata.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array in microns.
    transit_depth : array-like
        Transit depth (Rp/Rs)² at each wavelength. Dimensionless.
    err : array-like, optional
        1-sigma uncertainty on transit_depth. If None, assumed uniform 0.
    target_name : str, optional
        Exoplanet identifier (e.g., 'K2-18 b').
    instrument : str, optional
        Observing instrument (e.g., 'JWST/NIRSpec', 'Hubble/WFC3').
    resolution : float, optional
        Spectral resolving power R = λ/Δλ.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength in microns.
    transit_depth : np.ndarray
        Observed transit depth spectrum.
    err : np.ndarray
        Uncertainty array (same shape as transit_depth).
    meta : dict
        Dictionary of all metadata (target_name, instrument, etc.)

    Examples
    --------
    Load from CSV and compute water detection:

    >>> from bioniumx.spectra import TransmissionSpectrum
    >>> from bioniumx.io import load_csv
    >>> wl, depth, err = load_csv("K2-18b_jwst.csv")
    >>> spec = TransmissionSpectrum(wl, depth, err=err, target_name="K2-18 b")
    >>> spec
    <TransmissionSpectrum | {'wavelength': (256,), 'transit_depth': (256,)}>

    References
    ----------
    Madhusudhan et al. (2023), ApJL, 956, L18 — K2-18b carbon detection.
    """

    _required_attrs = ["wavelength", "transit_depth"]
    _metadata_attrs = ["target_name", "instrument", "resolution"]

    def __init__(
        self,
        wavelength,
        transit_depth,
        err=None,
        target_name: str = "Unknown",
        instrument: str = "Unknown",
        resolution: float = None,
        gti: list = None,
    ):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.transit_depth = np.asarray(transit_depth, dtype=float)

        if err is not None:
            self.err = np.asarray(err, dtype=float)
        else:
            self.err = np.zeros_like(self.transit_depth)

        self.meta = {
            "target_name": target_name,
            "instrument": instrument,
            "resolution": resolution,
        }
        self.gti = gti  # Good Wavelength Intervals (analogous to Stingray's GTIs)
        self._validate()

    def apply_wavelength_mask(self, wl_min: float, wl_max: float):
        """Return a masked copy within [wl_min, wl_max] microns."""
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        return TransmissionSpectrum(
            wavelength=self.wavelength[mask],
            transit_depth=self.transit_depth[mask],
            err=self.err[mask],
            **self.meta,
        )

    def rebin(self, factor: int):
        """
        Bin down the wavelength grid by averaging ``factor`` adjacent channels.

        Parameters
        ----------
        factor : int
            Binning factor. Must divide len(wavelength) evenly.

        Returns
        -------
        rebinned : TransmissionSpectrum
            New object with wavelength grid reduced by ``factor``.
        """
        n = len(self.wavelength) // factor * factor
        wl = self.wavelength[:n].reshape(-1, factor).mean(axis=1)
        depth = self.transit_depth[:n].reshape(-1, factor).mean(axis=1)
        # Error propagation: σ_bin = sqrt(sum(σ²)) / factor
        err = np.sqrt((self.err[:n].reshape(-1, factor) ** 2).sum(axis=1)) / factor
        return TransmissionSpectrum(wl, depth, err=err, **self.meta)

    def snr(self) -> np.ndarray:
        """
        Compute signal-to-noise ratio per channel.

        Returns
        -------
        snr : np.ndarray
            SNR = transit_depth / err. Returns inf where err == 0.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.err > 0, self.transit_depth / self.err, np.inf)

    def plot(self, ax=None, **kwargs):
        """
        Quick-look matplotlib plot of the transmission spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        **kwargs
            Passed to matplotlib errorbar.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(
            self.wavelength, self.transit_depth * 1e6,
            yerr=self.err * 1e6, fmt="o", ms=3, **kwargs
        )
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Transit Depth (ppm)")
        ax.set_title(self.meta.get("target_name", ""))
        return ax
