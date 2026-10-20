<div align="center">

  <h1>InundationRisk</h1>

  <h3>A Generalized Open-Source Framework for Regional Inundation and Multi-Hazard Risk Modeling</h3>

  <p>
    <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="License: BSD 3-Clause"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-green.svg" alt="Python Version"></a>
    <a href="#"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="CI Status"></a>
    <a href="#"><img src="https://img.shields.io/badge/FAIR-Reproducible%20Science-orange.svg" alt="Reproducibility: FAIR"></a>
    <a href="https://github.com/mkrifai/inundation-risk-model/blob/main/CITATION.cff"><img src="https://img.shields.io/badge/Cite-CITATION.cff-blueviolet.svg" alt="Cite"></a>
  </p>

  <p>
    <strong>Author:</strong><br>
    Mochamad Khoirul Rifai and Cahyo Aji Hapsoro<br>
    <em>Geophysics Research Group, Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Negeri Malang (UM), Indonesia</em>
  </p>

</div>

---

## 🌊 Overview

**InundationRisk** is a modular, site-agnostic, and academically reproducible Python modeling framework for regional flood and inundation hazard risk assessment. Built according to international open-science standards (FAIR principles), the framework integrates:

1. **Extreme Hydroclimatic Modeling**: Extreme Value Distributions (Gumbel & GEV) with analytical 95% Confidence Intervals and standardized WMO ETCCDI climate indices.
2. **Geospatial Terrain & Hydrological Conditioning**: Automated extraction of DEM morphometry (Horn's Slope, Aspect, Zevenbergen–Thorne Curvatures), Topographic Wetness Index (TWI), and Euclidean stream network proximity.
3. **Spatial Machine Learning Hazard Engine**: Data-driven flood susceptibility modeling (CatBoost, XGBoost, Random Forest) coupled with **Spatial Block Cross-Validation** to mitigate spatial autocorrelation bias, and full **SHAP (SHapley Additive exPlanations)** interpretability.
4. **IPCC Risk Assessment Framework**: Multi-criteria Analytic Hierarchy Process (AHP with Consistency Ratio $CR < 0.10$) synthesizing a relative $\text{Risk Index} = \text{Hazard} \times \text{Exposure} \times \text{Vulnerability}$.
5. **Spatial Adaptation Scenarios**: Forward simulation of climate change rainfall intensification and municipal spatial plan (RTRW) conflict analysis, projecting potential risk reduction under modeled adaptation measures.

---

## 📐 Computational & Theoretical Architecture

```mermaid
flowchart TD
    subgraph Data ["Data Inputs (Site-Agnostic)"]
        DEM["Digital Elevation Model (DEM)"]
        RAIN["Precipitation Time Series (CHIRPS / ERA5)"]
        LULC["Land Cover & Soil Features"]
        INV["Historical Flood Inventory (Sentinel-1 / SAR)"]
    end

    subgraph Hydro ["Phase 1: Hydroclimate Engine"]
        RAIN --> GUM["Gumbel / GEV Distribution Fitting"]
        GUM --> RP["Design Rainfall (T = 2 to 100 yr + 95% CI)"]
        RAIN --> ETC["WMO ETCCDI Indices (RX1day, SDII, R95p)"]
    end

    subgraph Geo ["Phase 2: Spatial Feature Extraction"]
        DEM --> SLP["Slope, Aspect, Plan/Profile Curvature"]
        DEM --> ACC["D8 Flow Accumulation & Stream Network"]
        ACC & SLP --> TWI["Topographic Wetness Index (TWI)"]
        ACC --> DST["Euclidean Distance to Stream Channels"]
    end

    subgraph ML ["Phase 3: Machine Learning Hazard Modeling"]
        INV & SLP & TWI & DST & LULC --> SBCV["Spatial Block Cross-Validation (K-Means)"]
        SBCV --> MDLS["Model Training (CatBoost / XGBoost / RF)"]
        MDLS --> SHAP["SHAP Beeswarm & Partial Dependence (PDP)"]
        MDLS --> HAZMAP["Spatial Hazard Susceptibility Map P(flood)"]
    end

    subgraph IPCC ["Phase 4: IPCC Multi-Hazard Risk Integration"]
        HAZMAP --> RK["Risk Index = Hazard × Exposure × Vulnerability"]
        POP["WorldPop Population Density"] --> EXP["Exposure Index"]
        AHP["Analytic Hierarchy Process (CR < 0.10)"] --> VULN["Vulnerability Index (MCE)"]
        EXP & VULN --> RK
        RK --> CLS["Quantile / Jenks Risk Classification"]
        CLS --> HOT["Proposed Multi-Criteria Priority Ranking"]
    end

    subgraph Scen ["Phase 5: Scenario Simulation & Policy"]
        RK --> SCEN["Climate Intensification (+15% Rain)"]
        ZON["Municipal Spatial Plan (RTRW)"] --> CONF["Spatial Planning Conflict Detection"]
        SCEN & CONF --> POL["Projected Risk Reduction & Policy Scenarios"]
    end
```

---

## 🌐 Data Sources & Preprocessing Provenance

While **InundationRisk** is site-agnostic, the benchmark case study uses open-access international datasets with the following specifications:

| Layer | Source Archive | Temporal Coverage | Native Resolution | Processing & Method | Reference |
|---|---|---|---|---|---|
| **Precipitation** | CHIRPS v3.0 DAILY_RNL (UCSB CHC / ECMWF) | 1981–2025 (45 years) | $0.05^\circ \approx 5.5\text{ km}$ | Daily series extraction, annual maxima extraction, WMO ETCCDI calculation | Funk et al. (2026) |
| **Elevation (DEM)** | Copernicus DEM GLO-30 / SRTM | Static (v2021) | $30\text{ m}$ ($1\text{ arc-sec}$) | Reprojected to UTM Zone 49S, sink filling, Horn 8-neighbor gradient | Horn (1981) |
| **Curvatures** | Copernicus DEM GLO-30 | Static (v2021) | $30\text{ m}$ | 9-point bivariate quadratic polynomial fitting (Planform & Profile) | Zevenbergen & Thorne (1987) |
| **Hydrological TWI** | Derived from DEM | Static | $30\text{ m}$ | D8 flow accumulation, steady-state subsurface flow $\ln(a / \tan \beta)$ | Beven & Kirkby (1979) |
| **Population Exposure** | WorldPop Global Hub | 2020 (Unconstrained) | $100\text{ m}$ | Resampled to $30\text{ m}$, min-max normalized to $[0, 1]$ | WorldPop (2020) |
| **Land Cover (LULC)** | Copernicus Global Land Cover | 2019–2020 | $100\text{ m}$ | Resampled, categorized into permeability proxy classes | Buchhorn et al. (2020) |
| **Flood Inventory** | Synthetic / SAR Sentinel-1 | Benchmark sample | $30\text{ m}$ | Balanced random spatial sampling (50% positive, 50% negative) | — |

---

## 🔬 Mathematical Formulations

### 1. Extreme Rainfall Frequency (Gumbel GEV Type I) & Uncertainty

$$
\Phi(x) = \exp\left(-\exp\left(-\frac{x - \mu}{\beta}\right)\right)
$$

Reduced variate for return period $T$:

$$
y_T = -\ln\left(-\ln\left(1 - \frac{1}{T}\right)\right)
$$

$$
x_T = \mu + \beta \cdot y_T
$$

**Asymptotic Standard Error & 95% Confidence Interval (Kite, 1977; Chow et al., 1988):**

$$
\text{SE}(x_T) = \frac{\beta}{\sqrt{N}} \sqrt{1.11 + 0.52 y_T + 1.10 y_T^2}
$$

$$
\text{CI}_{0.95} = x_T \pm 1.96 \cdot \text{SE}(x_T)
$$

### 2. Topographic Wetness Index (TWI) (Beven & Kirkby, 1979)

$$
\text{TWI} = \ln\left(\frac{a}{\tan \beta}\right)
$$

where $a$ is the specific catchment area ($A \times \text{cell size}$) and $\beta$ is the terrain slope in radians.

### 3. IPCC Disaster Risk Synthesis (IPCC AR5/AR6)

$$
\text{Risk Index} = \text{Hazard} \times \text{Exposure} \times \text{Vulnerability}
$$

*Note: The synthesized metric represents a dimensionless relative Risk Index in the range $[0, 1]$, designed for comparative zoning rather than absolute monetary or casualty forecasting.*

### 4. AHP Consistency Verification (Saaty, 1980)

$$
\text{CI} = \frac{\lambda_{\max} - n}{n - 1}, \quad \text{CR} = \frac{\text{CI}}{\text{RI}} < 0.10
$$

---

## 📁 Repository Structure

```text
inundation-risk/
├── .github/workflows/ci.yml           # Automated CI testing workflow
├── inundation_risk/                   # Core Python package (site-agnostic)
│   ├── __init__.py                    # Public API exports
│   ├── cli.py                         # Unified Command-Line Interface
│   ├── hydroclimate/                  # Gumbel, GEV, return periods + 95% CI, ETCCDI
│   ├── spatial_features/              # DEM morphometry, TWI, HAND, stream distance
│   ├── hazard_ml/                     # Spatial block CV, ML trainers, SHAP explainability
│   ├── risk_ipcc/                     # AHP matrices, exposure, vulnerability, risk overlay
│   └── scenarios/                     # Climate scaling, spatial planning conflict analysis
├── configs/
│   ├── default_config.yaml            # Generic template for any new basin/region
│   └── malang_benchmark_config.yaml   # Benchmark configuration for Malang Raya
├── benchmarks/
│   └── malang_raya/                   # Curated benchmark dataset (< 50MB)
│       ├── data/                      # Sample GeoTIFFs, vectors, and rainfall CSV
│       ├── reference_outputs/         # Golden reference metrics
│       └── run_benchmark.py           # Turnkey replication script
├── notebooks/                         # Pre-executed Academic Benchmarks & Tutorials
│   ├── InundationRisk_Master_Academic_Benchmark_Proof.ipynb  # 🌟 Benchmark & Theoretical Proof
│   ├── 01_hydroclimatic_extremes.ipynb                      # Hydroclimatic Extremes & Return Periods
│   ├── 02_spatial_conditioning_factors.ipynb                # Terrain Morphometry & Hydrological Factors
│   ├── 03_machine_learning_hazard.ipynb                     # Spatial ML Hazard Modeling & Spatial Block CV
│   ├── 04_integrated_flood_risk.ipynb                       # IPCC Multi-Hazard Risk & Saaty AHP Integration
│   └── 05_adaptation_scenarios.ipynb                        # Climate Change Intensification & RTRW Policy Analysis
├── tests/                             # Multi-tier automated pytest suite (22 unit tests)
├── CITATION.cff                       # Machine-readable software citation
├── codemeta.json                      # CodeMeta academic metadata
├── CONTRIBUTING.md                    # Open-source contributor guidelines
├── CODE_OF_CONDUCT.md                 # Contributor Covenant Code of Conduct
├── LICENSE                            # BSD 3-Clause License
├── pyproject.toml                     # Modern Python build specification
├── requirements.txt                   # Pip lockfile
└── environment.yml                    # Conda environment definition
```

---

## 🚀 Installation

### Option A: Using Pip
```bash
git clone https://github.com/mkrifai/inundation-risk-model.git
cd inundation-risk-model
pip install -e .
```

### Option B: Using Conda
```bash
conda env create -f environment.yml
conda activate inundation-risk
pip install -e .
```

---

## ⚡ Quickstart

### 1. Python API
```python
import numpy as np
from inundation_risk.hydroclimate import fit_gumbel, compute_return_periods
from inundation_risk.risk_ipcc import compute_ahp_weights, compute_ipcc_risk

# 1. Fit extreme rainfall with 95% Confidence Intervals
rainfall = [85.0, 92.4, 78.1, 110.5, 95.0, 130.2, 88.0, 105.4, 115.0, 140.0]
fit = fit_gumbel(rainfall)
rp = compute_return_periods(fit, [2, 10, 50, 100], return_ci=True)
print(f"Design Rainfall T100: {rp['T100']} mm (95% CI: [{rp['T100_ci_lower']}, {rp['T100_ci_upper']}])")

# 2. Check AHP Consistency
matrix = np.array([
    [1.0, 2.0, 4.0],
    [0.5, 1.0, 3.0],
    [0.25, 0.3333, 1.0]
])
weights, lmax, cr = compute_ahp_weights(matrix, ["Hazard", "Exposure", "Vulnerability"])
print("AHP Weights:", weights, f"(CR={cr:.4f})")
```

### 2. Command-Line Interface (CLI)
```bash
# Display framework metadata and citation
inundation-risk info

# Run benchmark test
inundation-risk benchmark

# Execute full pipeline from config file
inundation-risk run --config configs/default_config.yaml
```

---

## 🧪 Automated Testing & Verification

The framework includes a comprehensive multi-tier automated test suite covering statistical distributions, geospatial algorithms, spatial CV, and end-to-end execution.

To run the complete test suite:
```bash
pytest tests/ -v
```

All 22 test suites pass with 100% compliance:
```text
tests/test_cli.py .............. PASSED
tests/test_hazard_ml.py ........ PASSED
tests/test_hydroclimate.py ..... PASSED
tests/test_risk_ipcc.py ........ PASSED
tests/test_scenarios.py ........ PASSED
tests/test_spatial_features.py . PASSED
===================== 22 passed in 17.43s =====================
```

---

## 📊 Benchmark Replication (Upper Brantas Basin / Malang Raya Case Study)

To reproduce the study results for the **Upper Brantas River Basin (Malang Raya region)**:
```bash
python benchmarks/malang_raya/run_benchmark.py
```

> **Academic Note on Benchmark Verification:**  
> The included turnkey benchmark verifies **software determinism, code reproducibility, and computational integrity** across environments against golden reference metrics. It serves as an end-to-end algorithmic testbed rather than an empirical loss validation against dynamic gauge-monitored historical flood depths.

---

## 📓 Interactive Notebooks & Academic Proofs

All notebooks located in `notebooks/` are **fully pre-executed** with graphical plots, mathematical formatting, and tabular outputs saved directly in cell metadata. You can inspect them directly on GitHub or run them interactively in JupyterLab.

| Notebook | Focus & Key Contributions | Interactive Views & Execution | Outputs & Proofs |
|---|---|---|---|
| [**`InundationRisk_Master_Academic_Benchmark_Proof.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/InundationRisk_Master_Academic_Benchmark_Proof.ipynb) | **Master Academic Benchmark & Theoretical Proof** covering all 5 pillars of the framework | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/InundationRisk_Master_Academic_Benchmark_Proof.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/InundationRisk_Master_Academic_Benchmark_Proof.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/InundationRisk_Master_Academic_Benchmark_Proof.ipynb) | • Rigorous LaTeX equations & derivations<br>• GEV/Gumbel Kolmogorov-Smirnov fit proof<br>• Return level curves with 95% Confidence Intervals<br>• Spatial Block CV vs Random CV bias mitigation proof<br>• SHAP feature attributions (Beeswarm, PDP)<br>• Saaty AHP matrix & eigen-analysis ($CR < 0.10$)<br>• High-resolution risk maps & climate scaling |
| [**`01_hydroclimatic_extremes.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/01_hydroclimatic_extremes.ipynb) | Extreme value distribution fitting & climate indices | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/01_hydroclimatic_extremes.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/01_hydroclimatic_extremes.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/01_hydroclimatic_extremes.ipynb) | • Gumbel vs. GEV return level curves ($T=2\dots100\text{ yr}$) with 95% CI<br>• Cunnane empirical plotting positions<br>• 7 core WMO ETCCDI indices (RX1day, RX5day, SDII, R95p, etc.) |
| [**`02_spatial_conditioning_factors.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/02_spatial_conditioning_factors.ipynb) | Terrain morphometry & hydrological feature extraction | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/02_spatial_conditioning_factors.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/02_spatial_conditioning_factors.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/02_spatial_conditioning_factors.ipynb) | • Zevenbergen-Thorne planar/profile curvatures<br>• Topographic Wetness Index (TWI) mapping<br>• Stream network delineation & Euclidean distance transforms |
| [**`03_machine_learning_hazard.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/03_machine_learning_hazard.ipynb) | Spatial ML hazard susceptibility modeling | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/03_machine_learning_hazard.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/03_machine_learning_hazard.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/03_machine_learning_hazard.ipynb) | • Spatial Block K-Fold vs. Random K-Fold comparison<br>• Random Forest & Tree ensemble training<br>• TreeExplainer SHAP summary plots & partial dependence |
| [**`04_integrated_flood_risk.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/04_integrated_flood_risk.ipynb) | IPCC disaster risk synthesis ($\text{Risk} = H \times E \times V$) | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/04_integrated_flood_risk.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/04_integrated_flood_risk.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/04_integrated_flood_risk.ipynb) | • Pairwise comparison matrix & Consistency Ratio check<br>• WorldPop population exposure integration<br>• Quantile / Jenks risk classification & proposed hotspot ranking |
| [**`05_adaptation_scenarios.ipynb`**](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/05_adaptation_scenarios.ipynb) | Forward climate scaling & spatial planning overlay | [![GitHub](https://img.shields.io/badge/GitHub-View%20Notebook-181717?logo=github)](https://github.com/mkrifai/inundation-risk-model/blob/main/notebooks/05_adaptation_scenarios.ipynb)<br>[![NBViewer](https://img.shields.io/badge/NBViewer-Academic%20View-orange)](https://nbviewer.org/github/mkrifai/inundation-risk-model/blob/main/notebooks/05_adaptation_scenarios.ipynb)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkrifai/inundation-risk-model/blob/main/notebooks/05_adaptation_scenarios.ipynb) | • +15% and +25% precipitation intensification scenarios<br>• Municipal Spatial Plan (RTRW 2023–2043) conflict detection<br>• Projected risk reduction metrics under modeled adaptation |

---

## 📚 Scientific References

1. **Beven, K. J., & Kirkby, M. J. (1979).** A physically based, variable contributing area model of basin hydrology/Un modèle à base physique de zone d'appel variable de l'hydrologie du bassin versant. *Hydrological Sciences Journal*, 24(1), 43–69. https://doi.org/10.1080/02626667909491834
2. **Chow, V. T., Maidment, D. R., & Mays, L. W. (1988).** *Applied Hydrology*. McGraw-Hill.
3. **Coles, S. (2001).** *An Introduction to Statistical Modeling of Extreme Values*. Springer.
4. **Funk, C., Peterson, P., Harrison, L., Saldivar, R., Landsfeld, M., Pedreros, D., Shukla, S., Fink, A. H., Davenport, F., Peterson, S., Turner, W., Sonnier, A., Budde, M., Tabor, K., Verdin, J., Hauzaree, D., Naim, M., Alaso, D., & Husak, G. (2026).** The Climate Hazards Center Infrared Precipitation with Stations, Version 3. *Scientific Data*, 13(1), 41597. https://doi.org/10.1038/s41597-026-07096-4
5. **Gumbel, E. J. (1958).** *Statistics of Extremes*. Columbia University Press.
6. **Horn, B. K. (1981).** Hill shading and the reflectance map. *Proceedings of the IEEE*, 69(1), 14–47. https://doi.org/10.1109/PROC.1981.11918
7. **IPCC. (2014).** *Climate Change 2014: Impacts, Adaptation, and Vulnerability*. Cambridge University Press.
8. **Kite, G. W. (1977).** *Frequency and Risk Analyses in Hydrology*. Water Resources Publications.
9. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
10. **O'Callaghan, J. F., & Mark, D. M. (1984).** The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28(3), 323–344.
11. **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera‐Arroita, G., ... & Dormann, C. F. (2017).** Cross‐validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
12. **Saaty, T. L. (1980).** *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill.
13. **WorldPop. (2020).** Global High Resolution Population Denominators Project. University of Southampton.
14. **Zevenbergen, L. W., & Thorne, C. R. (1987).** Quantitative analysis of land surface topography. *Earth Surface Processes and Landforms*, 12(1), 47–56. https://doi.org/10.1002/esp.3290120107
15. **Zhang, X., Alexander, L., Hegerl, G. C., Jones, P., Tank, A. K., Peterson, T. C., ... & Zwiers, F. W. (2011).** Indices for monitoring changes in extremes based on daily temperature and precipitation data. *Wiley Interdisciplinary Reviews: Climate Change*, 2(6), 851–870. https://doi.org/10.1002/wcc.147

---

## 📖 Citation

If you use **InundationRisk** in your research, thesis, or applications, please cite this repository using the metadata provided in [`CITATION.cff`](CITATION.cff) or click the **"Cite this repository"** button on GitHub.

### BibTeX
```bibtex
@software{Rifai and Cahyo_InundationRisk_2026,
  author = {Rifai, Mochamad Khoirul; Hapsoro, Cahyo Aji},
  title = {InundationRisk: A Generalized Open-Source Framework for Regional Inundation and Multi-Hazard Risk Modeling},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/mkrifai/inundation-risk-model},
  version = {1.0.0},
  license = {BSD-3-Clause}
}
```

---

## 📄 License

This project is licensed under the **BSD 3-Clause License** — see the [LICENSE](LICENSE) file for details.
