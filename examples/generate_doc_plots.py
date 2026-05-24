"""
Generate high-quality static plots for Bionium-X documentation.

This script fetches real JWST WASP-39b data, processes it,
and generates Matplotlib figures saved to `docs/_static/`
so they can be embedded in the Sphinx pages.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from bioniumx.datasets.fetch_real import fetch_wasp39b
from bioniumx.datasets.ingestion import load_spectrum
from bioniumx.preprocessing import savitzky_golay, gaussian_smooth, continuum_normalize
from bioniumx.molecules.catalog import get_template
from bioniumx.detection.cross_correlation import cross_correlate_template, plot_ccf
from bioniumx.spectra import TransmissionSpectrum, EmissionSpectrum
from bioniumx.physics.habitability import habitability_score
from bioniumx.molecules.disequilibrium import compute_disequilibrium
from bioniumx.detection.bayesian import bayes_factor

# Set plot style for professional look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'figure.dpi': 200
})

def main():
    # Ensure docs/_static exists
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "_static"))
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching WASP-39b data...")
    csv_path = fetch_wasp39b()
    wavelength, flux, noise = load_spectrum(csv_path)

    spec = TransmissionSpectrum(
        wavelength=wavelength,
        transit_depth=flux,
        err=noise,
        target_name="WASP-39b",
        instrument="JWST/NIRISS"
    )

    # ---------------------------------------------------------
    # Plot 1: Raw Transmission Spectrum
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#3498db")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_spectrum.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 2: Emission Spectrum (Mock Data based on WASP-39b wl)
    # ---------------------------------------------------------
    emission_flux = np.interp(wavelength, [min(wavelength), max(wavelength)], [500, 3000]) # simple blackbody curve mock
    emission_flux += np.random.normal(0, 100, len(wavelength))
    emis_spec = EmissionSpectrum(
        wavelength=wavelength,
        flux=emission_flux,
        err=np.ones_like(wavelength)*100,
        target_name="WASP-39b (Emission)"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    emis_spec.plot(ax=ax, color="#e67e22")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_emission.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 3: Savitzky-Golay Smoothing
    # ---------------------------------------------------------
    spec_smoothed = savitzky_golay(spec, window=15, polyorder=3)
    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#bdc3c7", alpha=0.6, label="Raw Data")
    spec_smoothed.plot(ax=ax, color="#e74c3c", label="Savitzky-Golay")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_smoothed.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 4: Gaussian Smoothing
    # ---------------------------------------------------------
    spec_gauss = gaussian_smooth(spec, sigma=2.0)
    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#bdc3c7", alpha=0.6, label="Raw Data")
    spec_gauss.plot(ax=ax, color="#2ecc71", label="Gaussian Smoothing")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_gaussian.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 5: Continuum Normalization
    # ---------------------------------------------------------
    spec_norm = continuum_normalize(spec_smoothed, method="polynomial", degree=2)
    fig, ax = plt.subplots(figsize=(10, 4))
    spec_norm.plot(ax=ax, color="#9b59b6")
    ax.set_ylabel("Normalized Depth")
    ax.set_title("Continuum Normalized Spectrum")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_normalized.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 6: Cross-Correlation
    # ---------------------------------------------------------
    wl_co2, depth_co2 = get_template("CO2", resolving_power=100)
    result = cross_correlate_template(spec_norm, wl_co2, depth_co2)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_ccf(result, target_molecule="CO2", ax=ax)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "wasp39b_ccf_co2.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 7: Habitability Scores
    # ---------------------------------------------------------
    planets = ["Earth", "Proxima b", "TRAPPIST-1e", "WASP-39b", "Kepler-22b"]
    T_eqs = [255, 234, 251, 1166, 262]
    radii = [1.0, 1.07, 0.92, 1.27 * 11.2, 2.4] # WASP-39 is Jupiter sized (~14 R_earth)

    scores = [habitability_score(t, r) for t, r in zip(T_eqs, radii)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(planets, scores, color=['#27ae60', '#2ecc71', '#2ecc71', '#c0392b', '#f39c12'])
    ax.set_xlabel("Habitability Score (0-1)")
    ax.set_xlim(0, 1)
    ax.set_title("Exoplanet Habitability Comparison")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "habitability_comparison.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 8: Chemical Disequilibrium
    # ---------------------------------------------------------
    # Simulated detection significances
    detections = {"O2": 4.5, "CH4": 3.8, "H2O": 6.2, "CO2": 2.1}
    diseq_res = compute_disequilibrium(detections)

    fig, ax = plt.subplots(figsize=(8, 5))
    molecules = list(detections.keys())
    sigs = list(detections.values())

    ax.bar(molecules, sigs, color='#34495e')
    ax.axhline(2.0, color='r', linestyle='--', label="Detection Threshold (2σ)")

    # Annotate disequilibrium score
    text_str = f"Composite Disequilibrium Score: {diseq_res.disequilibrium_score:.2f}\n"
    text_str += f"Detected Pairs: {', '.join([f'{a}-{b}' for a,b in diseq_res.detected_pairs])}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_ylabel("Detection Significance (σ)")
    ax.legend(loc='upper right')
    ax.set_title("Chemical Disequilibrium Network Analysis")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "disequilibrium_network.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ---------------------------------------------------------
    # Plot 9: Bayesian Evidence
    # ---------------------------------------------------------
    lnZ_no_mol = -155.4
    lnZ_with_mol = [-154.2, -150.1, -145.8, -155.0] # H2O, CO2, CH4, CO
    mols = ["H2O", "CO2", "CH4", "CO"]

    bayes_factors = [bayes_factor(z, lnZ_no_mol) for z in lnZ_with_mol]
    log_K = np.log10(bayes_factors)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['r' if k > 0 else 'gray' for k in log_K]
    ax.bar(mols, log_K, color=colors)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(np.log10(10), color='g', linestyle='--', alpha=0.5, label="Strong Evidence (K > 10)")
    ax.axhline(np.log10(100), color='b', linestyle='--', alpha=0.5, label="Decisive Evidence (K > 100)")

    ax.set_ylabel(r"$\log_{10}(K)$ (Bayes Factor)")
    ax.set_title("Bayesian Model Comparison (vs Null Model)")
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "bayesian_evidence.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
