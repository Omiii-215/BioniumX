# Bionium-X

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Documentation Status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](docs/_build/html/index.html)

**Bionium-X** is a rigorous, pure-Python scientific library for high-performance exoplanetary biosignature detection and atmospheric modeling. It provides the foundational mathematical, statistical, and machine-learning frameworks required for analyzing high-resolution transmission and emission spectra.

By parsing highly-dimensional signals from orbital telescopes, the system probabilistically estimates the presence of critical biological precursors—thereby advancing the automated search for extraterrestrial life.

## Key Scientific Features

* **Real Observational Data Ingestion**: Seamlessly fetch and load real JWST (e.g., WASP-39b) and Hubble spectroscopic data from public archives using the internal `datasets` module.
* **Astrobiological Physics & Habitability**: Compute Earth Similarity Indices (ESI) and habitability scores constrained by equilibrium temperatures and planetary radii.
* **Chemical Disequilibrium Detection**: Simultaneously analyze detection significances of multiple reactive molecules (O₂, CH₄, N₂O) to identify highly active, life-supporting atmospheric thermodynamics.
* **Advanced Signal Processing**: Built-in methods for Savitzky-Golay filtering, 1D Gaussian smoothing, and polynomial continuum normalization for accurate feature extraction.
* **Template Cross-Correlation**: Native connection to the Harvard HITRAN API via `radis` to fetch theoretical high-resolution absorption templates and perform Voigt-broadened cross-correlation to confidently detect specific gases.
* **Bayesian Inference**: Compare atmospheric models quantitatively using Bayes Factors to weigh the statistical evidence of biosignatures.
* **Machine Learning Interoperability**: Experimental 1D CNNs, Random Forests, and Spectral Transformers for automated feature extraction and multi-label probability prediction.

## Documentation

Bionium-X comes with a fully automated, comprehensive Sphinx documentation suite featuring visual tutorials and a complete API reference.

To build and view the documentation locally:
```bash
cd docs
make clean && make html
# Open docs/_build/html/index.html in your browser
```

## Installation Instructions

This system is strictly validated for Unix/macOS environments running Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/YourOrg/Bionium-X.git
cd Bionium-X

# Initialize virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quickstart: Detecting CO2 in WASP-39b

Here is a simple example of fetching real JWST data, smoothing it, and using cross-correlation to detect Carbon Dioxide:

```python
import matplotlib.pyplot as plt
from bioniumx.datasets.fetch_real import fetch_wasp39b
from bioniumx.datasets.ingestion import load_spectrum
from bioniumx.spectra import TransmissionSpectrum
from bioniumx.preprocessing import savitzky_golay, continuum_normalize
from bioniumx.molecules.catalog import get_template
from bioniumx.detection.cross_correlation import cross_correlate_template, plot_ccf

# 1. Fetch & Load Real JWST Data
csv_path = fetch_wasp39b()
wavelength, flux, noise = load_spectrum(csv_path)

spec = TransmissionSpectrum(
    wavelength=wavelength, transit_depth=flux, err=noise,
    target_name="WASP-39b", instrument="JWST/NIRISS"
)

# 2. Preprocess: Smooth and Normalize
spec_smoothed = savitzky_golay(spec, window=15, polyorder=3)
spec_norm = continuum_normalize(spec_smoothed, method="polynomial", degree=2)

# 3. Cross-Correlate against HITRAN CO2 Template
wl_co2, depth_co2 = get_template("CO2", resolving_power=100)
result = cross_correlate_template(spec_norm, wl_co2, depth_co2)

# 4. Plot the Peak
fig, ax = plt.subplots(figsize=(8, 4))
plot_ccf(result, target_molecule="CO2", ax=ax)
plt.show()
```

*(For more advanced examples, including Bayesian model comparison and Habitability scoring, please view the `Core Functionality` section of the Sphinx documentation!)*

## Scientific Background

A single atmospheric gas is rarely indicative of life. However, combinations of highly reactive gases that rapidly destroy each other (such as Methane and Oxygen) violate thermodynamic equilibrium. Unless replenished by geologic or continuous biological processes (like photosynthesis and methanogenesis), these combinations would not persist. Bionium-X identifies these disequilibrium thresholds using rigorous statistical models.

## Project Structure

```text
Bionium-X/
│
├── bioniumx/                 # Core scientific library
│   ├── core.py               # Base class definitions
│   ├── spectra/              # Transmission and Emission spectrum objects
│   ├── datasets/             # Fetching and loading routines
│   ├── preprocessing/        # Filters and Normalization
│   ├── physics/              # Habitability, equilibrium, and atmospheric properties
│   ├── molecules/            # HITRAN catalog interfacing & disequilibrium logic
│   ├── detection/            # Cross-correlation and Bayesian factors
│   ├── modeling/             # PyTorch and Scikit-Learn deep learning architectures
│   └── io/                   # HDF5 Input/Output utilities
│
├── docs/                     # Comprehensive Sphinx documentation (build with `make html`)
├── examples/                 # Example scripts and plotting generators
├── tests/                    # Robust unit testing suite
└── requirements.txt          # Strictly versioned environment dependencies
```

## Contribution Guidelines
We welcome collaboration from astrophysicists and deep learning engineers alike! 

Please refer to the `tests/` directory for our `pytest` suite. When adding new modules, ensure you export them in the `__init__.py` files so the Sphinx `autosummary` can automatically generate their API reference pages.

## License
Bionium-X is released under the **MIT License**. See the `LICENSE` file for more details, enabling unrestricted academic and commercial integration with proper attribution.

<div align="center">
  <i>"Ad Astra Per Algorithmos"</i>
</div>
