import pandas as pd
import numpy as np
import h5py

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False


def ingest_csv(filepath):
    """
    Ingest spectrum data from CSV.
    Expected format: columns 'wavelength', 'flux', 'noise' (optional)
    """
    df = pd.read_csv(filepath, comment='#')
    # Auto-detect JWST ERS column names (e.g. wave, dppm)
    if 'wave' in df.columns and 'dppm' in df.columns:
        # JWST ERS specific format
        # Filter out comments/headers if pandas didn't already
        df = df[pd.to_numeric(df['wave'], errors='coerce').notnull()]
        wavelength = df['wave'].astype(float).values
        # dppm is parts per million, convert to absolute transit depth (0 to 1)
        flux = df['dppm'].astype(float).values / 1_000_000.0
        
        if 'dppm_err' in df.columns:
            noise = df['dppm_err'].astype(float).values / 1_000_000.0
        else:
            noise = np.zeros_like(flux)
    else:
        # Standard format fallback
        if 'wavelength' not in df.columns or 'flux' not in df.columns:
            raise ValueError("CSV must contain 'wavelength'/'flux' OR 'wave'/'dppm' columns.")

        wavelength = df['wavelength'].values
        flux = df['flux'].values
        noise = df['noise'].values if 'noise' in df.columns else np.zeros_like(flux)

    return wavelength, flux, noise


def ingest_fits(filepath):
    """
    Ingest spectrum data from FITS file.
    Assumes standard 1D spectral FITS format.
    """
    if not HAS_FITS:
        raise ImportError("astropy is required to read FITS files.")

    with fits.open(filepath) as hdul:
        data = hdul[1].data
        wavelength = data['WAVELENGTH']
        flux = data['FLUX']
        if 'ERROR' in data.columns.names:
            noise = data['ERROR']
        else:
            noise = np.zeros_like(flux)

    return wavelength, flux, noise


def ingest_hdf5(
        filepath,
        wavelength_key='wavelength',
        flux_key='flux',
        noise_key='noise'):
    """
    Ingest spectrum data from HDF5 file.
    """
    with h5py.File(filepath, 'r') as f:
        wavelength = f[wavelength_key][:]
        flux = f[flux_key][:]
        if noise_key in f:
            noise = f[noise_key][:]
        else:
            noise = np.zeros_like(flux)

    return wavelength, flux, noise


def load_spectrum(filepath):
    """
    Auto-detect format by extension and load.
    """
    if filepath.endswith('.csv'):
        return ingest_csv(filepath)
    elif filepath.endswith('.fits') or filepath.endswith('.fit'):
        return ingest_fits(filepath)
    elif filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        return ingest_hdf5(filepath)
    else:
        raise ValueError(f"Unsupported file format for {filepath}")
