"""
Dataset builder for spatial hazard machine learning.
Extracts multi-factor values at presence and pseudo-absence locations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def build_ml_dataset(
    factors: Dict[str, np.ndarray],
    flood_presence_mask: np.ndarray,
    non_flood_mask: Optional[np.ndarray] = None,
    ratio_negative: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Construct a tabular DataFrame from multi-band factor rasters for ML training.

    Parameters:
    -----------
    factors : dict of str -> 2D np.ndarray
        Conditioning factors (e.g., {'slope': arr, 'twi': arr, 'dem': arr}).
    flood_presence_mask : 2D np.ndarray (bool)
        Binary mask of historical flood occurrences (positive class = 1).
    non_flood_mask : 2D np.ndarray (bool), optional
        Mask of confirmed non-flood / low susceptibility zones. If None,
        samples from ~flood_presence_mask where all factors are valid.
    ratio_negative : float
        Ratio of negative samples to positive samples (default: 1.0).
    random_state : int
        Random seed for reproducible sampling.

    Returns:
    --------
    X : pd.DataFrame (feature matrix)
    y : pd.Series (binary target: 1 for flood, 0 for non-flood)
    coords : np.ndarray, shape (n_samples, 2) (row, col indices)
    """
    rng = np.random.default_rng(random_state)

    # Determine global valid mask where none of the factors are NaN
    factor_names = list(factors.keys())
    first_factor = factors[factor_names[0]]
    rows, cols = first_factor.shape

    valid_mask = np.ones((rows, cols), dtype=bool)
    for name, arr in factors.items():
        valid_mask &= ~np.isnan(arr)

    # Positive samples: presence AND valid
    pos_coords = np.argwhere(flood_presence_mask & valid_mask)
    n_pos = len(pos_coords)
    if n_pos == 0:
        raise ValueError("No valid flood presence points found inside the domain.")

    # Negative candidate mask
    if non_flood_mask is not None:
        neg_candidates = np.argwhere(non_flood_mask & valid_mask & (~flood_presence_mask))
    else:
        neg_candidates = np.argwhere(valid_mask & (~flood_presence_mask))

    n_neg = int(n_pos * ratio_negative)
    if len(neg_candidates) < n_neg:
        n_neg = len(neg_candidates)

    neg_choice = rng.choice(len(neg_candidates), size=n_neg, replace=False)
    neg_coords = neg_candidates[neg_choice]

    # Combine
    all_coords = np.vstack([pos_coords, neg_coords])
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

    # Sample factor values
    data_dict = {}
    r_idx = all_coords[:, 0]
    c_idx = all_coords[:, 1]

    for name in factor_names:
        data_dict[name] = factors[name][r_idx, c_idx]

    X = pd.DataFrame(data_dict)
    y = pd.Series(labels, name="target")

    return X, y, all_coords
