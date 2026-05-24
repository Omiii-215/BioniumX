"""
Fetch real observational datasets for Bionium-X examples.
"""
import pooch
import os

# Create a fetcher for real astrophysical data
data_fetcher = pooch.create(
    path=pooch.os_cache("bioniumx_data"),
    base_url="https://raw.githubusercontent.com/afeinstein20/wasp39b_niriss_paper/main/data/ts/",
    registry={
        "MCR-WASP_39b_NIRISS_transmission_spectrum.csv": None, # bypass hash check
    }
)

def fetch_wasp39b() -> str:
    """
    Download the real WASP-39b JWST NIRISS transmission spectrum.

    Returns
    -------
    path : str
        Absolute path to the downloaded CSV file.
    """
    return data_fetcher.fetch("MCR-WASP_39b_NIRISS_transmission_spectrum.csv")
