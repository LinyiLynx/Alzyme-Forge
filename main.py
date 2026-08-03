from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "src"
DEFAULT_CONFIG = ROOT / "configs" / "extrapolation.yaml"
DEFAULT_TEST_CSV = ROOT / "data" / "splits" / "test.csv"


def _inject_default_args(argv: list[str]) -> list[str]:
    if len(argv) < 2 or argv[1] not in {"train", "eval", "predict"}:
        return argv

    patched = list(argv)
    command = patched[1]
    insert_at = 2

    if "--config" not in patched:
        patched[insert_at:insert_at] = ["--config", str(DEFAULT_CONFIG)]
        insert_at += 2

    if command in {"eval", "predict"} and "--csv" not in patched:
        patched[insert_at:insert_at] = ["--csv", str(DEFAULT_TEST_CSV)]

    return patched


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(SRC))
    sys.argv = _inject_default_args(sys.argv)

    from eppgt_repro.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
