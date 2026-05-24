"""
Generate high-quality static plots for Bionium-X documentation.

This script fetches real JWST WASP-39b data, processes it, 
and generates Matplotlib figures saved to `docs/_static/` 
so they can be embedded in the Sphinx `core_functionality.rst` page.
"""
import os
import matplotlib.pyplot as plt

from bioniumx.datasets.fetch_real import fetch_wasp39b
from bioniumx.datasets.ingestion import load_spectrum
from bioniumx.preprocessing import savitzky_golay
from bioniumx.molecules.catalog import get_template
from bioniumx.detection.cross_correlation import cross_correlate_template, plot_ccf

def main():
    # Ensure docs/_static exists
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "_static"))
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    print("Fetching WASP-39b data...")
    csv_path = fetch_wasp39b()
    wavelength, flux, noise = load_spectrum(csv_path)
    
    from bioniumx.spectra import TransmissionSpectrum
    spec = TransmissionSpectrum(
        wavelength=wavelength,
        transit_depth=flux,
        err=noise,
        target_name="WASP-39b",
        instrument="JWST/NIRISS"
    )
    
    # Plot 1: Raw Spectrum
    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#3498db")
    fig.tight_layout()
    plot_path1 = os.path.join(out_dir, "wasp39b_spectrum.png")
    fig.savefig(plot_path1, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path1}")

    # 2. Smoothing
    print("Smoothing spectrum...")
    spec_smoothed = savitzky_golay(spec, window=15, polyorder=3)
    
    # Plot 2: Smoothed Spectrum overlay
    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#bdc3c7", alpha=0.6, label="Raw Data")
    spec_smoothed.plot(ax=ax, color="#e74c3c", label="Savitzky-Golay Smoothed")
    ax.legend()
    fig.tight_layout()
    plot_path2 = os.path.join(out_dir, "wasp39b_smoothed.png")
    fig.savefig(plot_path2, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path2}")

    # 3. Cross-Correlation
    print("Fetching CO2 template via RADIS...")
    wl_co2, depth_co2 = get_template("CO2", resolving_power=100)
    result = cross_correlate_template(spec_smoothed, wl_co2, depth_co2)
    
    # Plot 3: CCF
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_ccf(result, target_molecule="CO2", ax=ax)
    fig.tight_layout()
    plot_path3 = os.path.join(out_dir, "wasp39b_ccf_co2.png")
    fig.savefig(plot_path3, dpi=200)
    plt.close(fig)
    print(f"Saved {plot_path3}")

if __name__ == "__main__":
    main()
