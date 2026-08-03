#!/usr/bin/env python3
"""Build manifest for top-N jobs (wrapper around prepare_manifest)."""

from prepare_manifest import main

if __name__ == "__main__":
    import sys

    if "--top-n" not in sys.argv:
        sys.argv.extend(["--top-n", "10"])
    if "--out-dir" not in sys.argv:
        sys.argv.extend(["--out-dir", "neg_fix_v1_top10"])
    main()
