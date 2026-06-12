"""
Bayesian model comparison and Bayes factor computation.
"""
import numpy as np


def bayes_factor(evidence_m1: float, evidence_m2: float) -> float:
    """
    Calculate the Bayes Factor (K) comparing two models.

    The Bayes factor is the ratio of the marginal likelihoods (evidences)
    of two competing models. In atmospheric retrieval, M1 might be a model
    containing a specific molecule, and M2 is the model without it.

    Parameters
    ----------
    evidence_m1 : float
        Log-evidence (ln Z) for model 1.
    evidence_m2 : float
        Log-evidence (ln Z) for model 2.

    Returns
    -------
    K : float
        The Bayes factor K = Z1 / Z2. Returns float('inf') if numerical overflow.

    References
    ----------
    Trotta, R. (2008), Bayes in the sky: Bayesian inference and model selection
    in cosmology, Contemp. Phys., 49, 71-104.

    Examples
    --------
    >>> lnZ_with_h2o = -150.2
    >>> lnZ_no_h2o = -155.4
    >>> K = bayes_factor(lnZ_with_h2o, lnZ_no_h2o)
    >>> print(f"Bayes factor for H2O: {K:.1f}")
    """
    delta_lnZ = evidence_m1 - evidence_m2
    result = float(np.exp(delta_lnZ))
    return float('inf') if np.isinf(result) else result
