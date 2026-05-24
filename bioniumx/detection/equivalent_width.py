"""
Equivalent width measurements for spectral lines.
"""
import numpy as np
from bioniumx.core import BioniumXObject


def equivalent_width(
    spectrum: BioniumXObject,
    line_min: float,
    line_max: float,
    continuum: tuple = None,
):
    r"""
    Calculate the equivalent width (EW) of a spectral absorption feature.

    Equivalent width is a measure of the area of a spectral line relative
    to the continuum. In transmission spectroscopy, it correlates with
    atmospheric abundance.

    Parameters
    ----------
    spectrum : BioniumXObject
        The input spectrum (must be continuum-normalized).
    line_min, line_max : float
        Wavelength bounds of the absorption feature (in microns).
    continuum : tuple of (float, float), optional
        Wavelength bounds to compute the local continuum level.
        If None, assumes the spectrum is already normalized to 1.0.

    Returns
    -------
    ew : float
        The equivalent width in microns.
    ew_err : float
        1-sigma uncertainty on the equivalent width.

    Raises
    ------
    ValueError
        If the integration range is invalid.

    Notes
    -----
    Mathematical derivation:
    .. math::
        EW = \int_{\lambda_1}^{\lambda_2} \left(1 - \frac{F(\lambda)}{F_c}\right) d\lambda

    Examples
    --------
    >>> ew, ew_err = equivalent_width(spec_norm, 1.38, 1.42)
    >>> print(f"H2O EW = {ew:.4f} ± {ew_err:.4f} μm")
    """
    if line_min >= line_max:
        raise ValueError("line_min must be less than line_max")

    wl = spectrum.wavelength
    
    if hasattr(spectrum, 'transit_depth'):
        # For transmission, depth usually goes UP for absorption lines.
        # But equivalent width is traditionally defined for flux drops.
        # If the input is normalized transit depth (depth/continuum), it's > 1.
        # We integrate (depth/continuum - 1).
        y = spectrum.transit_depth
        y_err = spectrum.err
        is_transmission = True
    elif hasattr(spectrum, 'flux'):
        y = spectrum.flux
        y_err = spectrum.err
        is_transmission = False
    else:
        raise TypeError("Object must have either 'transit_depth' or 'flux' attribute.")

    # Compute local continuum if requested
    if continuum is not None:
        c_min, c_max = continuum
        c_mask = (wl >= c_min) & (wl <= c_max)
        if c_mask.sum() == 0:
            raise ValueError("No data points in continuum range.")
        fc = np.mean(y[c_mask])
        fc_err = np.std(y[c_mask]) / np.sqrt(c_mask.sum())
    else:
        fc = 1.0
        fc_err = 0.0

    # Integrate over the line
    l_mask = (wl >= line_min) & (wl <= line_max)
    if l_mask.sum() == 0:
        return 0.0, 0.0

    wl_line = wl[l_mask]
    y_line = y[l_mask]
    err_line = y_err[l_mask]

    # Delta lambda for numerical integration (assume uniform or take differences)
    dwl = np.gradient(wl_line) if len(wl_line) > 1 else np.array([0.0])

    if is_transmission:
        # Transit depth increases for absorption
        integrand = (y_line / fc) - 1.0
    else:
        # Flux decreases for absorption
        integrand = 1.0 - (y_line / fc)

    ew = np.sum(integrand * dwl)

    # Uncertainty propagation
    # Var(EW) = sum( (dwl/fc)^2 * err_line^2 ) + (EW/fc)^2 * fc_err^2
    var_ew = np.sum((dwl / fc)**2 * err_line**2) + (ew / fc)**2 * fc_err**2
    ew_err = np.sqrt(var_ew)

    return float(ew), float(ew_err)
