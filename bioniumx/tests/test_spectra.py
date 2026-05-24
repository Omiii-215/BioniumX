import numpy as np
import pytest
from bioniumx.spectra import TransmissionSpectrum

class TestTransmissionSpectrum:
    """Tests for TransmissionSpectrum data class."""

    @pytest.fixture
    def simple_spectrum(self):
        wl = np.linspace(0.6, 5.3, 256)
        depth = 0.014 + 1e-4 * np.random.randn(256)
        err = 1e-5 * np.ones(256)
        return TransmissionSpectrum(wl, depth, err=err, target_name="Test b")

    def test_init_basic(self, simple_spectrum):
        assert simple_spectrum.wavelength.shape == (256,)
        assert simple_spectrum.transit_depth.shape == (256,)
        assert simple_spectrum.meta["target_name"] == "Test b"

    def test_missing_wavelength_raises(self):
        with pytest.raises(ValueError, match="requires 'wavelength'"):
            spec = TransmissionSpectrum.__new__(TransmissionSpectrum)
            spec._required_attrs = ["wavelength", "transit_depth"]
            spec._validate()

    def test_apply_wavelength_mask(self, simple_spectrum):
        masked = simple_spectrum.apply_wavelength_mask(1.0, 2.5)
        assert masked.wavelength.min() >= 1.0
        assert masked.wavelength.max() <= 2.5

    def test_rebin_reduces_size(self, simple_spectrum):
        rebinned = simple_spectrum.rebin(4)
        assert len(rebinned.wavelength) == len(simple_spectrum.wavelength) // 4

    def test_rebin_error_propagation(self, simple_spectrum):
        rebinned = simple_spectrum.rebin(4)
        # RMS error in bin = sqrt(4 * sigma²) / 4 = sigma / sqrt(4) = sigma/2
        expected_err = simple_spectrum.err[0] / np.sqrt(4)
        np.testing.assert_allclose(rebinned.err[0], expected_err, rtol=0.01)

    def test_snr_finite_where_err_positive(self, simple_spectrum):
        snr = simple_spectrum.snr()
        assert np.all(np.isfinite(snr))
        assert np.all(snr > 0)

    def test_roundtrip_hdf5(self, simple_spectrum, tmp_path):
        path = str(tmp_path / "test_spec.h5")
        simple_spectrum.write(path)
        loaded = TransmissionSpectrum.read(path)
        np.testing.assert_allclose(loaded.wavelength, simple_spectrum.wavelength)
        np.testing.assert_allclose(loaded.transit_depth, simple_spectrum.transit_depth)
