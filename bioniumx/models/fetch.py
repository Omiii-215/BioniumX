"""
Data fetching using Pooch.

Downloads and caches heavy machine learning models and large templates
so they don't bloat the git repository.
"""
import pooch
import os

# Create the fetcher
fetcher = pooch.create(
    path=pooch.os_cache("bioniumx"),
    base_url="https://github.com/bionium-x-research/bioniumx-data/raw/main/",
    registry={
        "cnn_model.pth": "md5:dummyhash1234567890",
        "rf_model.pkl": "md5:dummyhash0987654321",
        "h2o_template.h5": "md5:dummyhash1111111111",
    }
)

def fetch_model(model_name: str) -> str:
    """
    Download a pre-trained model file and return its local path.

    Parameters
    ----------
    model_name : str
        Name of the model file (e.g., 'cnn_model.pth').

    Returns
    -------
    path : str
        Absolute path to the downloaded (or cached) file.
    """
    # In a real scenario, this fetches from the URL.
    # For this implementation, we simulate it if the file doesn't exist.
    try:
        return fetcher.fetch(model_name)
    except Exception as e:
        # Fallback for local development if the remote URL isn't real
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "..", "saved_models", model_name)
        if os.path.exists(fallback_path):
            return os.path.abspath(fallback_path)
        raise FileNotFoundError(f"Could not download {model_name} and local fallback not found.") from e
