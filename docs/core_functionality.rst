=========================
Core Bionium-X Functionality
=========================

Here we show how many of the core Bionium-X classes and methods work in practice. We start with basic data constructs for transmission spectra, show how to preprocess the data, and then demonstrate how to compute cross-correlation arrays for biosignature detection.

Working with Transmission Spectra
---------------------------------

### 1. Fetching Real Data

Bionium-X natively supports downloading high-resolution exoplanet spectra from public archives using `pooch`.

.. code-block:: python

    from bioniumx.datasets.fetch_real import fetch_wasp39b
    from bioniumx.datasets.ingestion import load_spectrum
    from bioniumx.spectra import TransmissionSpectrum

    # Download JWST NIRISS observation of WASP-39b
    csv_path = fetch_wasp39b()

    # Parse standard columns ('wave', 'dppm') into pure arrays
    wavelength, flux, noise = load_spectrum(csv_path)

### 2. The TransmissionSpectrum Object

We initialize the primary data structure by passing our arrays and metadata. You can immediately visualize the data using the native `.plot()` method.

.. code-block:: python

    import matplotlib.pyplot as plt

    spec = TransmissionSpectrum(
        wavelength=wavelength,
        transit_depth=flux,
        err=noise,
        target_name="WASP-39b",
        instrument="JWST/NIRISS"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#3498db")
    plt.show()

.. image:: _static/wasp39b_spectrum.png
   :alt: Raw JWST WASP-39b Transmission Spectrum
   :align: center


Preprocessing and Filtering
---------------------------

Observational data often contains high-frequency noise. Bionium-X provides several smoothing algorithms, including the Savitzky-Golay filter, which preserves the shape of the massive absorption lines while minimizing pixel-to-pixel scatter.

.. code-block:: python

    from bioniumx.preprocessing import savitzky_golay

    # Apply Savitzky-Golay with a window length of 15 and 3rd-order polynomial
    spec_smoothed = savitzky_golay(spec, window=15, polyorder=3)

    fig, ax = plt.subplots(figsize=(10, 4))
    spec.plot(ax=ax, color="#bdc3c7", alpha=0.6, label="Raw Data")
    spec_smoothed.plot(ax=ax, color="#e74c3c", label="Savitzky-Golay Smoothed")
    ax.legend()
    plt.show()

.. image:: _static/wasp39b_smoothed.png
   :alt: Savitzky-Golay smoothed spectrum
   :align: center


Template Cross-Correlation
--------------------------

To definitively detect the presence of a molecule (e.g., Carbon Dioxide), we cross-correlate the observed spectrum against a high-resolution theoretical template. 

Bionium-X seamlessly connects to the Harvard HITRAN API via the `radis` library to compute Voigt-broadened quantum cross-sections.

.. code-block:: python

    from bioniumx.molecules.catalog import get_template
    from bioniumx.detection.cross_correlation import cross_correlate_template, plot_ccf

    # Fetch CO2 absorption cross-sections at T=1000K
    wl_co2, depth_co2 = get_template("CO2", resolving_power=100)

    # Correlate across radial velocity shifts from -150 to +150 km/s
    result = cross_correlate_template(spec_smoothed, wl_co2, depth_co2)

    # Visualize the detection significance peak
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_ccf(result, target_molecule="CO2", ax=ax)
    plt.show()

.. image:: _static/wasp39b_ccf_co2.png
   :alt: CO2 Cross-Correlation Function Peak
   :align: center

As shown above, the strong correlation peak at 0 km/s (in the planetary rest frame) confirms a highly significant detection of Carbon Dioxide in the atmosphere of WASP-39b!
