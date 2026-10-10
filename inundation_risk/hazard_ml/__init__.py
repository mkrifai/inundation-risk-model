"""
Machine learning hazard susceptibility modeling: dataset builders,
spatial block cross-validation, multi-algorithm trainers, and SHAP explainability.
"""

from inundation_risk.hazard_ml.dataset import build_ml_dataset
from inundation_risk.hazard_ml.spatial_cv import SpatialBlockKFold
from inundation_risk.hazard_ml.trainers import HazardModelTrainer
from inundation_risk.hazard_ml.explainability import compute_shap_explanations

__all__ = [
    "build_ml_dataset",
    "SpatialBlockKFold",
    "HazardModelTrainer",
    "compute_shap_explanations",
]
