"""
BioniumXObject — Universal base class for all Bionium-X data structures.
Modeled after Stingray's StingrayObject pattern.
"""
import numpy as np
import warnings

class BioniumXObject:
    """
    Base class for all Bionium-X scientific data objects.

    Provides standard I/O, arithmetic, and metadata handling
    for TransmissionSpectrum, EmissionSpectrum, and derived products.

    Parameters
    ----------
    Not instantiated directly. Use subclasses.

    Notes
    -----
    All subclasses must define ``_required_attrs`` listing the array
    attributes that must be set for the object to be valid.
    """

    _required_attrs = []       # Override in subclasses
    _metadata_attrs = []       # Non-array metadata fields

    def __repr__(self):
        name = self.__class__.__name__
        attrs = {a: getattr(self, a, None) for a in self._required_attrs}
        shape_info = {k: (v.shape if hasattr(v, 'shape') else v)
                      for k, v in attrs.items() if v is not None}
        return f"<{name} | {shape_info}>"

    def _validate(self):
        """Check all required attributes are set and consistent."""
        for attr in self._required_attrs:
            val = getattr(self, attr, None)
            if val is None:
                raise ValueError(
                    f"{self.__class__.__name__} requires '{attr}' to be set."
                )

    def write(self, filename: str, fmt: str = "hdf5") -> None:
        """
        Save this object to a file.

        Parameters
        ----------
        filename : str
            Output file path.
        fmt : str, optional
            Format: 'hdf5', 'fits', or 'ascii'. Default 'hdf5'.

        Examples
        --------
        >>> spec.write("K2-18b.h5")
        >>> spec.write("K2-18b.fits", fmt="fits")
        """
        from bioniumx.io import write_object
        write_object(self, filename, fmt=fmt)

    @classmethod
    def read(cls, filename: str, fmt: str = "hdf5"):
        """
        Load an object from a file.

        Parameters
        ----------
        filename : str
            Input file path.
        fmt : str, optional
            Format: 'hdf5', 'fits', or 'ascii'. Default 'hdf5'.

        Returns
        -------
        obj : BioniumXObject subclass
            Reconstructed object.

        Examples
        --------
        >>> spec = TransmissionSpectrum.read("K2-18b.h5")
        """
        from bioniumx.io import read_object
        return read_object(cls, filename, fmt=fmt)

    def apply_wavelength_mask(self, wl_min: float, wl_max: float):
        """
        Return a masked copy of this object within [wl_min, wl_max] microns.

        Parameters
        ----------
        wl_min, wl_max : float
            Wavelength bounds in microns.

        Returns
        -------
        masked : same type as self
        """
        raise NotImplementedError("Subclasses must implement apply_wavelength_mask")
