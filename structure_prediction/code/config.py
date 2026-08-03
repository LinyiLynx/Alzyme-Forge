"""Shared paths for neg_fix_v1 top200 Boltz campaign."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "neg_fix_v1_top200"
COMPOUND = CAMPAIGN / "compound_1"

INPUTS = COMPOUND / "inputs"
YAML_DIR = INPUTS / "yaml"
MANIFEST = INPUTS / "manifest.csv"

WORK = COMPOUND / "work"
STRUCTURES = COMPOUND / "structures"
SUMMARY = COMPOUND / "summary.csv"
LOGS = CAMPAIGN / "logs"

BOLTZ_CACHE = ROOT / "cache"
CONDA_ENV = "colabfold_gpu"
