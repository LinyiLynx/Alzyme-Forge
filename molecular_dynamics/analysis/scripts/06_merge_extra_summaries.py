#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

base = Path("md_analysis_all")
rows = []

for p in sorted(base.glob("candidates/rank*/extra/extra_summary.csv")):
    df = pd.read_csv(p)
    rows.append(df)

if not rows:
    raise SystemExit("No extra_summary.csv found.")

out_df = pd.concat(rows, ignore_index=True)

out = base / "tables" / "md_2ns_top20_extra_summary.csv"
out_df.to_csv(out, index=False)

print("Wrote:", out)
print(out_df.to_string(index=False))