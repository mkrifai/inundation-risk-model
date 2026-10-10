# Contributing to InundationRisk

Thank you for your interest in contributing to **InundationRisk**! As an academic open-science project, we welcome contributions from hydrologists, geoscientists, data scientists, and software engineers to enhance regional flood hazard and multi-hazard risk modeling.

---

## 🏛️ Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors with respect and professionalism.

---

## 🚀 How to Contribute

### 1. Reporting Issues & Feature Requests
- Check existing GitHub Issues before submitting a new report.
- Clearly describe the bug or feature request, including environment details (OS, Python version), input data formats, and minimum reproducible examples.

### 2. Developing Locally
1. **Fork and Clone**:
   ```bash
   git clone https://github.com/<your-username>/inundation-risk-model.git
   cd inundation-risk-model
   ```
2. **Set up Virtual Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate inundation-risk
   # or with virtualenv:
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```
3. **Branching**:
   Create a descriptive branch for your feature or bugfix:
   ```bash
   git checkout -b feature/improved-spatial-cross-validation
   ```

### 3. Code Quality & Standards
- Follow **PEP 8** style guidelines for Python code.
- Write clear docstrings for all public classes, functions, and methods following Google/NumPy docstring format.
- Maintain academic rigor: cite foundational literature when implementing physical or statistical equations.
- Include unit tests in `tests/` for any new mathematical or geospatial logic.

### 4. Running Tests
Before submitting a pull request, ensure all unit and integration tests pass:
```bash
pytest tests/ -v
python benchmarks/malang_raya/run_benchmark.py
```

### 5. Submitting Pull Requests (PR)
- Reference any relevant Issue numbers in your PR description.
- Summarize changes, theoretical motivation, and verification evidence.
- Ensure all CI tests pass.

---

## 📄 License
By contributing to InundationRisk, you agree that your contributions will be licensed under the project's [BSD 3-Clause License](LICENSE).
