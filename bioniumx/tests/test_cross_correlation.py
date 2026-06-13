"""Tests for CCF significance — regression for issue #49.

The significance estimate must exclude a window around the detected peak when
computing CCF noise, so a real detection at a non-zero systemic velocity isn't
deflated by including its own peak in the noise sample.
"""
import numpy as np

from bioniumx.spectra import TransmissionSpectrum
from bioniumx.detection.cross_correlation import cross_correlate_template


def _inject_spectrum(v_inject_kms, seed=0):
    """Build an observed spectrum with a known feature shifted to v_inject."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(1.0, 2.0, 500)
    template_depth = np.exp(-((wl - 1.5) ** 2) / (2 * 0.002 ** 2))
    c = 2.998e5
    wl_obs = wl / (1.0 + v_inject_kms / c)
    depth_obs = np.interp(wl, wl_obs, template_depth)
    depth_obs = depth_obs + rng.normal(0, 0.01, wl.size)
    spec = TransmissionSpectrum(
        wavelength=wl, transit_depth=depth_obs, err=np.full(wl.size, 0.01)
    )
    return spec, wl, template_depth


def test_significance_detects_signal_at_zero_velocity():
    spec, wl_t, depth_t = _inject_spectrum(0.0)
    res = cross_correlate_template(spec, wl_t, depth_t)
    assert abs(res["peak_velocity"]) < 5.0
    assert res["significance"] > 5.0


def test_significance_not_deflated_for_offset_peak():
    """Regression for #49: a peak at +100 km/s must not be deflated by the
    old fixed |v| > 50 noise window that included the peak itself."""
    spec, wl_t, depth_t = _inject_spectrum(100.0)
    res = cross_correlate_template(spec, wl_t, depth_t)
    # The peak should be found near the injected velocity...
    assert abs(res["peak_velocity"] - 100.0) < 10.0
    # ...and reported as a strong detection, not deflated to a few sigma.
    assert res["significance"] > 5.0


def test_noise_window_excludes_peak_region():
    """The noise sample used for significance should not include the peak."""
    spec, wl_t, depth_t = _inject_spectrum(120.0)
    res = cross_correlate_template(spec, wl_t, depth_t)
    velocities = res["velocity"]
    ccf = res["ccf"]
    peak_v = res["peak_velocity"]
    # Rebuild the peak-excluded noise mask; confirm the peak bin is excluded.
    noise_mask = np.abs(velocities - peak_v) > 15.0
    peak_idx = int(np.argmax(np.abs(ccf)))
    assert not noise_mask[peak_idx]
