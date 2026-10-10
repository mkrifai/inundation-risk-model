---
description: Universal standards for academic modeling, scientific software claims, and geospatial-hydrological research workflows.
globs: ["**/*.py", "**/*.ipynb", "**/*.md", "**/*.yaml"]
---

# Scientific Modeling & Academic Rigor Standards

When designing, implementing, or documenting scientific modeling frameworks (especially geospatial, hydroclimatic, or machine learning pipelines):

## 1. Calibrated Academic Phrasing
- **Spatial Autocorrelation**: Do NOT claim that spatial cross-validation (e.g. K-Means coordinate clustering) "eliminates" spatial leakage. State that it *mitigates* or *reduces spatial autocorrelation bias*, and acknowledge that residual spatial dependency can persist.
- **Scenario Outputs**: Do NOT state that adaptation or planning overlays produce "avoided disaster impact" as a factual outcome. Frame them as *projected risk reduction under modeled scenarios* or *exploratory forward simulations*.
- **Machine Learning Performance**: Avoid unverified claims of "high-accuracy" unless benchmarked against independent real-world ground truth or observed flood extents.

## 2. Data Provenance & Algorithmic Attribution
- **Source Specifics**: Always document exact archive names, version numbers (e.g., CHIRPS v3.0 DAILY_RNL; Funk et al., 2026), temporal coverage (e.g., 1981–2025, 45 years), and spatial resolution (e.g., 0.05°, 30m, 100m).
- **Core Algorithm Citations**: Explicitly attribute standard formulas and algorithms in docstrings and docs:
  - Extreme Value Theory: Maximum Likelihood Estimation (MLE), Gumbel (1958), Coles (2001), WMO ETCCDI (Zhang et al., 2011).
  - Terrain Morphometry: Horn (1981) for slope/aspect; Zevenbergen & Thorne (1987) for curvatures.
  - Hydrology: Topographic Wetness Index via Beven & Kirkby (1979); D8 flow routing via O'Callaghan & Mark (1984).
  - Multi-Criteria Decision: Saaty (1980) Analytic Hierarchy Process (AHP).
  - Explainable ML: Lundberg & Lee (2017) for TreeExplainer SHAP.

## 3. Scope Boundaries & Uncertainty
- **Benchmark Nature**: Explicitly distinguish between *code-verification benchmarks* (synthetic/deterministic testing against reference code) and *empirical validation* (testing against observed flood gauges/satellite inundation).
- **Uncertainty Quantification**: Where feasible, provide analytical or empirical uncertainty estimates (e.g., 95% confidence intervals on return periods).

## 4. Public Science Communication & Social Dissemination
- **Modest & Calibrated Tone**: In public dissemination (e.g., LinkedIn, research summaries, conference notes), emphasize workflow reproducibility, modularity, and exploratory potential rather than definitive policy claims.
- **Explicit Disclaimers**: Frame forward applications (e.g., spatial planning or climate adaptation overlays) as intended exploration avenues dependent on local context and ground-truth validation.
- **Minimalist Academic Humility**: Avoid jargon bloating or overselling; maintain a balanced tone acknowledging ongoing development and community validation.
