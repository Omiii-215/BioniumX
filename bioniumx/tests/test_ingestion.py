import pytest
import pandas as pd
import numpy as np
import h5py

# Assuming the module is located at bioniumx/datasets/ingestion.py
from bioniumx.datasets.ingestion import (
    ingest_csv,
    ingest_fits,
    ingest_hdf5,
    load_spectrum,
    HAS_FITS
)

# Fixtures for generating temporary test files
@pytest.fixture
def standard_csv(tmp_path):
    filepath = tmp_path / "standard.csv"
    df = pd.DataFrame({
        'wavelength': [1.0, 2.0, 3.0],
        'flux': [10.5, 11.0, 10.8],
        'noise': [0.1, 0.2, 0.15]
    })
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def standard_csv_no_noise(tmp_path):
    filepath = tmp_path / "no_noise.csv"
    df = pd.DataFrame({
        'wavelength': [1.0, 2.0, 3.0],
        'flux': [10.5, 11.0, 10.8]
    })
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def jwst_csv(tmp_path):
    filepath = tmp_path / "jwst.csv"
    df = pd.DataFrame({
        'wave': [1.0, 'invalid_text', 3.0], # Tests the coercion logic
        'dppm': [1000000.0, 2000000.0, 500000.0],
        'dppm_err': [1000.0, 2000.0, 500.0]
    })
    # Add a comment row to test comment filtering
    with open(filepath, 'w') as f:
        f.write("# This is a JWST comment\n")
    df.to_csv(filepath, mode='a', index=False)
    return str(filepath)

@pytest.fixture
def hdf5_file(tmp_path):
    filepath = tmp_path / "data.h5"
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('wavelength', data=np.array([1.0, 2.0, 3.0]))
        f.create_dataset('flux', data=np.array([10.5, 11.0, 10.8]))
        f.create_dataset('noise', data=np.array([0.1, 0.2, 0.15]))
    return str(filepath)

@pytest.fixture
def fits_file(tmp_path):
    if not HAS_FITS:
        pytest.skip("astropy not installed, skipping FITS file creation")
    
    from astropy.io import fits
    filepath = tmp_path / "data.fits"
    
    # Create a dummy FITS file with a primary HDU and a BinTableHDU
    primary_hdu = fits.PrimaryHDU()
    
    c1 = fits.Column(name='WAVELENGTH', format='D', array=np.array([1.0, 2.0, 3.0]))
    c2 = fits.Column(name='FLUX', format='D', array=np.array([10.5, 11.0, 10.8]))
    c3 = fits.Column(name='ERROR', format='D', array=np.array([0.1, 0.2, 0.15]))
    
    table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3])
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(filepath)
    
    return str(filepath)

# Tests for CSV Ingestion
def test_ingest_csv_standard(standard_csv):
    wave, flux, noise = ingest_csv(standard_csv)
    np.testing.assert_array_equal(wave, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(flux, [10.5, 11.0, 10.8])
    np.testing.assert_array_equal(noise, [0.1, 0.2, 0.15])

def test_ingest_csv_no_noise(standard_csv_no_noise):
    wave, flux, noise = ingest_csv(standard_csv_no_noise)
    np.testing.assert_array_equal(wave, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(flux, [10.5, 11.0, 10.8])
    np.testing.assert_array_equal(noise, [0.0, 0.0, 0.0]) # Should be zeros_like

def test_ingest_csv_jwst(jwst_csv):
    wave, flux, noise = ingest_csv(jwst_csv)
    # The 'invalid_text' row should be dropped by coerce
    np.testing.assert_array_equal(wave, [1.0, 3.0]) 
    # 1_000_000 dppm should equal 1.0 flux
    np.testing.assert_array_equal(flux, [1.0, 0.5]) 
    np.testing.assert_array_equal(noise, [0.001, 0.0005])

def test_ingest_csv_malformed(tmp_path):
    filepath = tmp_path / "bad.csv"
    pd.DataFrame({'a': [1], 'b': [2]}).to_csv(filepath, index=False)
    
    with pytest.raises(ValueError, match="CSV must contain"):
        ingest_csv(str(filepath))

def test_ingest_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_csv("nonexistent_file.csv")

# Tests for HDF5 Ingestion
def test_ingest_hdf5(hdf5_file):
    wave, flux, noise = ingest_hdf5(hdf5_file)
    np.testing.assert_array_equal(wave, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(flux, [10.5, 11.0, 10.8])
    np.testing.assert_array_equal(noise, [0.1, 0.2, 0.15])

# Tests for FITS Ingestion
@pytest.mark.skipif(not HAS_FITS, reason="astropy is not installed")
def test_ingest_fits(fits_file):
    wave, flux, noise = ingest_fits(fits_file)
    np.testing.assert_array_equal(wave, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(flux, [10.5, 11.0, 10.8])
    np.testing.assert_array_equal(noise, [0.1, 0.2, 0.15])

def test_ingest_fits_no_astropy(monkeypatch):
    # Temporarily force HAS_FITS to False to test the fallback
    import bioniumx.datasets.ingestion as ing
    monkeypatch.setattr(ing, 'HAS_FITS', False)
    
    with pytest.raises(ImportError, match="astropy is required"):
        ing.ingest_fits("dummy.fits")

# Tests for load_spectrum (Router)
def test_load_spectrum_routing_csv(standard_csv):
    wave, flux, noise = load_spectrum(standard_csv)
    assert len(wave) == 3

def test_load_spectrum_routing_h5(hdf5_file):
    wave, flux, noise = load_spectrum(hdf5_file)
    assert len(wave) == 3

@pytest.mark.skipif(not HAS_FITS, reason="astropy is not installed")
def test_load_spectrum_routing_fits(fits_file):
    wave, flux, noise = load_spectrum(fits_file)
    assert len(wave) == 3

def test_load_spectrum_unsupported():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_spectrum("data.txt")