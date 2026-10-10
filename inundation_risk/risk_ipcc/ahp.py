"""
Analytic Hierarchy Process (AHP) multi-criteria weighting engine (Saaty, 1980).
Computes eigenvector weights, principal eigenvalue, Consistency Index (CI), and Consistency Ratio (CR).
"""

from typing import Dict, List, Tuple
import numpy as np

# Standard Random Inconsistency Index (RI) lookup table (Saaty, 1980)
RANDOM_INDEX = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}


def compute_ahp_weights(
    pairwise_matrix: np.ndarray,
    criteria_names: List[str]
) -> Tuple[Dict[str, float], float, float]:
    """
    Compute priority weights and consistency statistics from an AHP pairwise comparison matrix.

    Parameters:
    -----------
    pairwise_matrix : 2D np.ndarray, shape (n, n)
        Reciprocal square comparison matrix (A[i,j] = 1 / A[j,i]).
    criteria_names : list of str, length n
        Labels for each criterion.

    Returns:
    --------
    weights : dict of str -> float (normalized criteria weights summing to 1.0)
    lambda_max : float (principal eigenvalue)
    consistency_ratio : float (CR, must be < 0.10 for acceptable consistency)
    """
    matrix = np.asarray(pairwise_matrix, dtype=np.float64)
    n = matrix.shape[0]

    if matrix.shape != (n, n):
        raise ValueError(f"Pairwise matrix must be square (n x n). Got shape {matrix.shape}")
    if len(criteria_names) != n:
        raise ValueError(f"Length of criteria_names ({len(criteria_names)}) does not match matrix size ({n}).")

    # Column normalization method to compute priority eigenvector
    col_sums = np.sum(matrix, axis=0)
    norm_matrix = matrix / col_sums
    weights_vec = np.mean(norm_matrix, axis=1)
    weights_vec = weights_vec / np.sum(weights_vec)

    # Principal eigenvalue lambda_max = mean((A @ w) / w)
    weighted_sum = np.dot(matrix, weights_vec)
    lambda_max = float(np.mean(weighted_sum / weights_vec))

    # Consistency Index (CI)
    if n <= 2:
        ci = 0.0
        cr = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        ri = RANDOM_INDEX.get(n, 1.49)
        cr = ci / ri

    weights_dict = {name: float(np.round(w, 4)) for name, w in zip(criteria_names, weights_vec)}
    return weights_dict, float(np.round(lambda_max, 4)), float(np.round(cr, 4))


def verify_ahp_consistency(consistency_ratio: float, threshold: float = 0.10) -> bool:
    """Check whether AHP consistency ratio meets the standard CR < 0.10 threshold."""
    return consistency_ratio < threshold
