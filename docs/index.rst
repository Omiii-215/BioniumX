Bionium-X: Exoplanetary Biosignature Detection
==============================================

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/Python-3.9+-blue.svg
   :target: https://www.python.org/

**Bionium-X** is a rigorous, pure-Python scientific library for high-performance exoplanetary biosignature detection and atmospheric modeling. 

It provides the foundational mathematical, statistical, and machine-learning frameworks required for analyzing high-resolution transmission and emission spectra.

Why Bionium-X?
--------------
- **Rigorous Data Objects**: Powerful :class:`~bioniumx.spectra.transmission.TransmissionSpectrum` and `EmissionSpectrum` classes that handle masking, error propagation, and physical metadata natively.
- **Advanced Chemistry**: Calculate complex atmospheric chemical disequilibrium scores (weighted by chemical lifetimes) to search for biological imbalances.
- **Physical Limits**: Evaluate heuristic habitability scores based on the Earth Similarity Index, Goldilocks Zones, and atmospheric scale heights.
- **Machine Learning Integration**: Seamlessly interface with deep learning architectures (like 1D CNNs) via lazy-loaded network weights using Pooch.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   core_functionality

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
