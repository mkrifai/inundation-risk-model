import os
import subprocess
import pytest
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_cli_info():
    cmd = [sys.executable, "-m", "inundation_risk.cli", "info"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "INUNDATION-RISK FRAMEWORK" in result.stdout
    assert "Citation:" in result.stdout


def test_cli_benchmark():
    cmd = [sys.executable, "-m", "inundation_risk.cli", "benchmark"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "Golden Benchmark Verification: PASSED" in result.stdout


def test_cli_run():
    config_path = os.path.join("configs", "default_config.yaml")
    cmd = [sys.executable, "-m", "inundation_risk.cli", "run", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "Execution completed successfully." in result.stdout

