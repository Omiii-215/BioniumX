"""
Template cross-correlation for molecular detection.

The standard technique for detecting molecules in transmission spectra
is to cross-correlate the observed spectrum against a theoretical template
computed from a line list (HITRAN/ExoMol). This implementation follows the
method described in Snellen et al. (2010) and Brogi & Line (2019).
"""
import numpy as np
from scipy.interpolate import interp1d
from bioniumx.spectra import TransmissionSpectrum


def cross_correlate_template(
    spectrum: TransmissionSpectrum,
    template_wavelength: np.ndarray,
    template_depth: np.ndarray,
    velocity_range: tuple = (-150, 150),
    velocity_step: float = 0.5,
) -> dict:
    """
    Cross-correlate a transmission spectrum against a molecular template.

    Shifts the template across a range of radial velocities and computes
    the Pearson correlation coefficient at each shift. A peak in the CCF
    near 0 km/s indicates the molecule is present.

    Parameters
    ----------
    spectrum : TransmissionSpectrum
        Observed spectrum to analyze.
    template_wavelength : array-like
        Template wavelength grid (microns).
    template_depth : array-like
        Template transit depth at each wavelength.
    velocity_range : tuple of (float, float), optional
        Min and max velocity shift to test in km/s. Default (-150, 150).
    velocity_step : float, optional
        Velocity resolution in km/s. Default 0.5.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'velocity' : np.ndarray — velocity axis (km/s)
        - 'ccf'      : np.ndarray — cross-correlation function values
        - 'peak_velocity' : float — velocity of CCF peak (km/s)
        - 'peak_ccf'      : float — peak CCF value
        - 'significance'  : float — peak significance in σ (Gaussian)

    References
    ----------
    Snellen, I. A. G. et al. (2010), Nature, 465, 1049.
    Brogi, M. & Line, M. R. (2019), AJ, 157, 114.

    Examples
    --------
    >>> from bioniumx.molecules import get_template
    >>> wl_t, depth_t = get_template("H2O", resolving_power=100)
    >>> result = cross_correlate_template(spec, wl_t, depth_t)
    >>> print(f"H2O peak at {result['peak_velocity']:.1f} km/s, "
    ...       f"significance={result['significance']:.1f}σ")
    """
    c_kms = 2.998e5  # speed of light in km/s
    velocities = np.arange(*velocity_range, velocity_step)
    template_interp = interp1d(
        template_wavelength, template_depth,
        kind="linear", bounds_error=False, fill_value=0.0
    )

    obs_depth = spectrum.transit_depth - spectrum.transit_depth.mean()
    ccf = np.zeros(len(velocities))

    for i, v in enumerate(velocities):
        # Doppler shift template to velocity v
        shifted_wl = spectrum.wavelength * (1.0 + v / c_kms)
        template_shifted = template_interp(shifted_wl)
        template_shifted -= template_shifted.mean()
        # Pearson cross-correlation
        norm = (np.std(obs_depth) * np.std(template_shifted))
        if norm > 0:
            ccf[i] = np.mean(obs_depth * template_shifted) / norm

    # Significance: peak vs. out-of-peak noise
    peak_idx = np.argmax(np.abs(ccf))
    peak_velocity = velocities[peak_idx]
    peak_ccf = ccf[peak_idx]
    # Estimate noise from CCF values away from the detected peak. Excluding a
    # window around the peak (rather than a fixed |v| > 50 cut) keeps the peak
    # out of its own noise sample, even when it sits at a non-zero velocity.
    # Follows Brogi & Line (2019) recommendation of ~10-20 km/s exclusion.
    exclusion_kms = 15.0
    noise_mask = np.abs(velocities - peak_velocity) > exclusion_kms
    noise_std = np.std(ccf[noise_mask]) if noise_mask.sum() > 5 else 1.0
    significance = abs(peak_ccf) / noise_std if noise_std > 0 else 0.0

    return {
        "velocity": velocities,
        "ccf": ccf,
        "peak_velocity": float(peak_velocity),
        "peak_ccf": float(peak_ccf),
        "significance": float(significance),
    }

def plot_ccf(result: dict, target_molecule: str = "", ax=None):
    """
    Plot the Cross-Correlation Function (CCF) from a cross_correlate_template result.

    Parameters
    ----------
    result : dict
        The output dictionary from `cross_correlate_template`.
    target_molecule : str, optional
        Name of the molecule (e.g., 'CO2') for the title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    v = result["velocity"]
    ccf = result["ccf"]
    peak_v = result["peak_velocity"]
    sig = result["significance"]

    ax.plot(v, ccf, color="#2c3e50", lw=1.5)
    ax.axvline(peak_v, color="#e74c3c", ls="--", lw=1.5, alpha=0.8,
               label=f"Peak: {peak_v:.1f} km/s ({sig:.1f}σ)")

    # Highlight 0 km/s rest frame
    ax.axvline(0, color="gray", ls=":", alpha=0.5)

    title = "Cross-Correlation Function"
    if target_molecule:
        title += f" ({target_molecule})"

    ax.set_title(title, pad=15)
    ax.set_xlabel("Radial Velocity Shift (km/s)")
    ax.set_ylabel("Pearson Correlation Coefficient")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax
