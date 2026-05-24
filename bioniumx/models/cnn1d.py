"""
1D Convolutional Neural Network for Biosignature Detection.

Requires PyTorch (`pip install bioniumx[ml]`).
"""
import warnings

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class BiosignatureCNN:
    """
    1D CNN to classify transmission spectra for specific biosignatures.

    This model takes a normalized transmission spectrum and outputs
    a probability score for the presence of a target molecule (e.g., H2O).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for BiosignatureCNN. Install via `pip install bioniumx[ml]`.")

        self.model = self._build_model(in_channels, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_model(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 64, 128),  # Assuming input length 256
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def load_weights(self, filepath: str):
        """Load pre-trained model weights."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()

    def predict(self, spectrum_array) -> float:
        """
        Predict the probability of the biosignature.

        Parameters
        ----------
        spectrum_array : np.ndarray
            1D array of length 256.

        Returns
        -------
        prob : float
            Probability score [0, 1].
        """
        if spectrum_array.shape[-1] != 256:
            raise ValueError("CNN expects an input array of length 256.")

        x = torch.tensor(spectrum_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            output = self.model(x)

        return output.item()
