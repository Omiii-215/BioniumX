"""
EmissionSpectrum — Core data class for exoplanet emission spectroscopy.
"""
import numpy as np
from bioniumx.core import BioniumXObject

class EmissionSpectrum(BioniumXObject):
    """
    Represents an emission spectrum of an exoplanet atmosphere.

    Stores wavelength grid, emitted flux (or planet-to-star flux ratio), and uncertainties.

    Parameters
    ----------
    wavelength : array-like
        Wavelength array in microns.
    flux : array-like
        Emitted flux or flux ratio at each wavelength.
    err : array-like, optional
        1-sigma uncertainty on flux. If None, assumed uniform 0.
    target_name : str, optional
        Exoplanet identifier (e.g., 'WASP-43 b').
    instrument : str, optional
        Observing instrument.
    resolution : float, optional
        Spectral resolving power R = λ/Δλ.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength in microns.
    flux : np.ndarray
        Observed emission flux.
    err : np.ndarray
        Uncertainty array.
    meta : dict
        Dictionary of all metadata.
    """

    _required_attrs = ["wavelength", "flux"]
    _metadata_attrs = ["target_name", "instrument", "resolution"]

    def __init__(
        self,
        wavelength,
        flux,
        err=None,
        target_name: str = "Unknown",
        instrument: str = "Unknown",
        resolution: float = None,
        gti: list = None,
    ):
        self.wavelength = np.asarray(wavelength, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

        if err is not None:
            self.err = np.asarray(err, dtype=float)
        else:
            self.err = np.zeros_like(self.flux)

        self.meta = {
            "target_name": target_name,
            "instrument": instrument,
            "resolution": resolution,
        }
        self.gti = gti
        self._validate()

    def apply_wavelength_mask(self, wl_min: float, wl_max: float):
        """Return a masked copy within [wl_min, wl_max] microns."""
        mask = (self.wavelength >= wl_min) & (self.wavelength <= wl_max)
        return EmissionSpectrum(
            wavelength=self.wavelength[mask],
            flux=self.flux[mask],
            err=self.err[mask],
            **self.meta,
        )

    def rebin(self, factor: int):
        n = len(self.wavelength) // factor * factor
        wl = self.wavelength[:n].reshape(-1, factor).mean(axis=1)
        f = self.flux[:n].reshape(-1, factor).mean(axis=1)
        err = np.sqrt((self.err[:n].reshape(-1, factor) ** 2).sum(axis=1)) / factor
        return EmissionSpectrum(wl, f, err=err, **self.meta)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(
            self.wavelength, self.flux,
            yerr=self.err, fmt="o", ms=3, **kwargs
        )
        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Emission Flux")
        ax.set_title(self.meta.get("target_name", ""))
        return ax
