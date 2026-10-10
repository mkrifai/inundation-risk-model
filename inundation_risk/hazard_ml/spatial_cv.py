"""
Spatial Block Cross-Validation to eliminate spatial autocorrelation leakage.
Splits geographical coordinates into disjoint spatial blocks for model validation.
"""

from typing import Generator, List, Tuple
import numpy as np
from sklearn.cluster import KMeans


class SpatialBlockKFold:
    """
    K-Fold Cross-Validator with Spatial Block Partitioning.
    Uses K-Means clustering on geospatial coordinates (X, Y) to create
    spatially contiguous folds, ensuring test samples are geographically
    independent of training samples.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coordinates: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate spatial train/test indices.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        coordinates : np.ndarray, shape (n_samples, 2)
            Projected coordinates [x, y] of each sample.

        Yields:
        -------
        train_idx, test_idx : arrays of indices
        """
        coords = np.asarray(coordinates, dtype=np.float64)
        if len(coords) != len(X):
            raise ValueError("Coordinates array length must match number of feature samples.")

        # Cluster coordinates into n_splits spatial clusters
        kmeans = KMeans(
            n_clusters=self.n_splits,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(coords)

        all_indices = np.arange(len(X))
        for cluster_id in range(self.n_splits):
            test_mask = (clusters == cluster_id)
            test_idx = all_indices[test_mask]
            train_idx = all_indices[~test_mask]
            yield train_idx, test_idx
