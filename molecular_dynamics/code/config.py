"""Shared paths for neg_fix_v1 GROMACS/OpenMM campaigns."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MDP_DIR = ROOT / "mdp"

CAMPAIGN = ROOT / "neg_fix_v1_top200"
LEGACY_CAMPAIGN = ROOT / "neg_fix_v1_top10"
MANIFEST = CAMPAIGN / "manifest.csv"
LOGS = CAMPAIGN / "logs"

SUMMARY_CSV = (
    Path(__file__).resolve().parents[1]
    / "Alphafold"
    / "neg_fix_v1_top200"
    / "compound_1"
    / "summary.csv"
)
MANIFEST_INPUTS = (
    Path(__file__).resolve().parents[1]
    / "Alphafold"
    / "neg_fix_v1_top200"
    / "compound_1"
    / "inputs"
    / "manifest.csv"
)

CONDA_ENV = "colabfold_gpu"
DEFAULT_PROD_NS = 2.0
DEFAULT_POLL_SEC = 120


def job_dir(job_id: str) -> Path:
    for base in (CAMPAIGN, LEGACY_CAMPAIGN):
        path = base / job_id
        if path.exists():
            return path
    return CAMPAIGN / job_id


def status_path(job_id: str) -> Path:
    for base in (CAMPAIGN, LEGACY_CAMPAIGN):
        path = base / job_id / "md_status.json"
        if path.exists():
            return path
    return CAMPAIGN / job_id / "md_status.json"
