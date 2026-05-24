import os
import pooch

# Create a master Pooch instance to manage large datasets
GOODCACHE = pooch.create(
    # Use the default cache location for the OS
    path=pooch.os_cache("bioniumx"),
    # Base URL for the data, can be updated later when a data repository is established
    base_url="https://raw.githubusercontent.com/Bionium-X-Collaborative/bioniumx-data/main/",
    # The registry is a dictionary of file names and their SHA256 hashes
    registry=None,
)

def fetch_data(filename: str, url: str = None, known_hash: str = None) -> str:
    """
    Fetch a data file, downloading and caching it locally if it doesn't exist.
    
    Parameters
    ----------
    filename : str
        The name of the file to fetch.
    url : str, optional
        If provided, download from this URL instead of the base_url.
    known_hash : str, optional
        The SHA256 hash of the file. If None, it will download without checking the hash
        (useful during development, but should be set for production data).
        
    Returns
    -------
    str
        The absolute path to the local cached file.
    """
    if url is not None:
        # For one-off downloads that aren't in the central registry
        return pooch.retrieve(
            url=url,
            known_hash=known_hash,
            path=GOODCACHE.path,
            fname=filename
        )
    else:
        # Use the master Pooch instance
        if GOODCACHE.registry is None:
            # If registry isn't populated, we fallback to retrieving from full URL
            raise ValueError("Registry is not populated. Please provide a full 'url'.")
        return GOODCACHE.fetch(filename)
