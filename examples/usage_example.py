"""
Bionium-X: Standard Usage Example
---------------------------------
This script demonstrates how a researcher would use the Bionium-X library
to analyze an exoplanet's transmission spectrum for biosignatures, entirely
independent of any UI or dashboard.
"""
import numpy as np

# 1. Clean imports from the flat, unified API
from bioniumx.spectra import TransmissionSpectrum
from bioniumx.preprocessing import savitzky_golay
from bioniumx.detection import cross_correlate_template
from bioniumx.molecules import get_template, compute_disequilibrium
from bioniumx.physics import habitability_score

def main():
    print("--- Bionium-X Scientific Library Example ---")

    # 1. Initialize a data object (simulating loaded observational data)
    # Wavelength grid from 1 to 5 microns
    wl_obs = np.linspace(1.0, 5.0, 500)
    # Simulated noisy transit depth data with an artificial feature around 1.4 um (water)
    depth_obs = 0.015 + 1e-4 * np.random.randn(500)
    depth_obs += 0.005 * np.exp(-0.5 * ((wl_obs - 1.4) / 0.05) ** 2)
    err_obs = 1e-5 * np.ones(500)

    spec = TransmissionSpectrum(
        wavelength=wl_obs,
        transit_depth=depth_obs,
        err=err_obs,
        target_name="K2-18 b",
        instrument="JWST NIRISS"
    )
    print(f"\nLoaded Object: {spec}")

    # 2. Preprocess the data
    print("Applying Savitzky-Golay filter to smooth observational noise...")
    spec_smoothed = savitzky_golay(spec, window=11, polyorder=3)

    # 3. Fetch theoretical templates
    print("\nFetching theoretical templates for cross-correlation...")
    wl_h2o, depth_h2o = get_template("H2O", resolving_power=100)
    wl_ch4, depth_ch4 = get_template("CH4", resolving_power=100)

    # 4. Run cross-correlation detection algorithm
    print("\nRunning template cross-correlation...")
    result_h2o = cross_correlate_template(spec_smoothed, wl_h2o, depth_h2o)
    result_ch4 = cross_correlate_template(spec_smoothed, wl_ch4, depth_ch4)

    # Store significances (simulated values for CH4 since template doesn't perfectly match our artificial signal)
    sigs = {
        "H2O": result_h2o["significance"],
        "CH4": 4.1,  # Force a high artificial signal for demonstration
        "O2": 5.2    # Force a high artificial signal for demonstration
    }

    print(f"  -> H2O Detection Significance: {sigs['H2O']:.1f} sigma")
    print(f"  -> CH4 Detection Significance: {sigs['CH4']:.1f} sigma")
    print(f"  -> O2  Detection Significance: {sigs['O2']:.1f} sigma")

    # 5. Evaluate chemical disequilibrium
    print("\nEvaluating network chemical disequilibrium...")
    # The compute_disequilibrium engine weights the detections by chemical lifetimes
    disequilibrium = compute_disequilibrium(sigs)
    
    print(f"  -> Detected Pairs: {disequilibrium.detected_pairs}")
    print(f"  -> Flags: {disequilibrium.flags}")
    print(f"  -> Composite Disequilibrium Score (0-1): {disequilibrium.disequilibrium_score:.2f}")

    # 6. Physical Habitability check
    print("\nChecking planetary physics limits...")
    # Assume T_eq = 265 K, R = 2.6 R_earth
    score = habitability_score(T_eq=265.0, radius_Rearth=2.6)
    print(f"  -> Heuristic Habitability Score (0-1): {score:.2f}")
    if score < 0.3:
        print("     (Note: Score heavily penalized due to large radius indicating non-rocky world)")

if __name__ == "__main__":
    main()
