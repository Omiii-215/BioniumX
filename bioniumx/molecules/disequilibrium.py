"""
Chemical disequilibrium biosignature detection.

A single molecule is rarely a biosignature. The simultaneous presence
of O₂ + CH₄ (or other highly reactive pairs) indicates an atmosphere
maintained out of equilibrium — the strongest abiotic-defying signature.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


DISEQUILIBRIUM_PAIRS = {
    # (mol_A, mol_B): (timescale_years, description, reference)
    ("O2", "CH4"): (
        1e-3,
        "Photosynthesis + Methanogenesis disequilibrium (Earth analog)",
        "Krissansen-Totton et al. 2018"
    ),
    ("O2", "CO"): (
        1e2,
        "Incomplete combustion / photochemical disequilibrium",
        "Schwieterman et al. 2019"
    ),
    ("N2O", "CH4"): (
        1e1,
        "Denitrification disequilibrium",
        "Seager et al. 2013"
    ),
    ("PH3", "O2"): (
        1e-6,
        "Phosphine in oxidizing atmosphere (Venus controversy)",
        "Greaves et al. 2020"
    ),
}


@dataclass
class DisequilibriumResult:
    """
    Result of a chemical disequilibrium analysis.

    Attributes
    ----------
    detected_pairs : list of tuple
        Molecule pairs simultaneously detected above threshold.
    disequilibrium_score : float
        Composite score 0–1. Higher → stronger disequilibrium signature.
    confidence_intervals : dict
        Per-molecule 1-sigma detection confidence.
    flags : list of str
        Human-readable description of detected signals.
    references : list of str
        Scientific references for each flagged pair.
    """
    detected_pairs: list = field(default_factory=list)
    disequilibrium_score: float = 0.0
    confidence_intervals: Dict[str, tuple] = field(default_factory=dict)
    flags: list = field(default_factory=list)
    references: list = field(default_factory=list)

    def is_significant(self, threshold: float = 0.5) -> bool:
        """Return True if disequilibrium_score exceeds threshold."""
        return self.disequilibrium_score > threshold


def compute_disequilibrium(
    molecule_detections: Dict[str, float],
    detection_threshold: float = 2.0,
) -> DisequilibriumResult:
    """
    Evaluate chemical disequilibrium from per-molecule significance values.

    Parameters
    ----------
    molecule_detections : dict
        Mapping of molecule name → detection significance in σ.
        Example: {"O2": 3.2, "CH4": 2.8, "H2O": 5.1}
    detection_threshold : float, optional
        Minimum σ to consider a molecule "detected". Default 2.0.

    Returns
    -------
    result : DisequilibriumResult

    Examples
    --------
    >>> sigs = {"O2": 3.5, "CH4": 2.9, "H2O": 6.0, "CO2": 4.1}
    >>> result = compute_disequilibrium(sigs)
    >>> print(result.disequilibrium_score)  # e.g. 0.78
    >>> result.is_significant()             # True
    """
    result = DisequilibriumResult()
    detected = {m for m, sig in molecule_detections.items()
                if sig >= detection_threshold}

    for (mol_a, mol_b), (timescale, desc, ref) in DISEQUILIBRIUM_PAIRS.items():
        if mol_a in detected and mol_b in detected:
            result.detected_pairs.append((mol_a, mol_b))
            result.flags.append(desc)
            result.references.append(ref)

    # Score: weighted by inverse chemical lifetime (faster destruction = stronger signal)
    if result.detected_pairs:
        pair_scores = []
        for (mol_a, mol_b) in result.detected_pairs:
            ts = DISEQUILIBRIUM_PAIRS[(mol_a, mol_b)][0]
            sig_a = molecule_detections.get(mol_a, 0.0)
            sig_b = molecule_detections.get(mol_b, 0.0)
            # Geometric mean of significances, weighted by chemical urgency (shorter ts = higher weight)
            weight = (10.0 - np.log10(max(ts, 1e-9))) / 5.0
            pair_score = (sig_a * sig_b) ** 0.5 * weight
            pair_scores.append(pair_score)
        # Normalize to [0, 1]
        raw_score = np.tanh(np.mean(pair_scores) / 5.0)
        result.disequilibrium_score = float(raw_score)

    return result
