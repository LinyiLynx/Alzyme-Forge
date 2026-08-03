#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("md_analysis_all")
CAND_DIR = BASE / "candidates"
TABLE_DIR = BASE / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pct_rank_low_good(series):
    return series.rank(method="average", pct=True, ascending=True)


def pct_rank_high_good(series):
    return series.rank(method="average", pct=True, ascending=False)


rows = []

for p in sorted(CAND_DIR.glob("rank*/summary.csv")):
    df = pd.read_csv(p)
    rows.append(df)

if not rows:
    raise SystemExit("No summary.csv files found.")

all_df = pd.concat(rows, ignore_index=True)

num_cols = [
    "ca_rmsd_last20pct_mean_A",
    "ligand_rmsd_last20pct_mean_A",
    "protein_rg_last20pct_std_A",
    "pl_min_distance_last20pct_mean_A",
    "pl_contacts_4A_last20pct_mean",
    "ca_rmsf_mean_A",
    "ca_rmsf_max_A",
]

for col in num_cols:
    if col in all_df.columns:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

score = 0

score += 0.25 * pct_rank_low_good(all_df["ca_rmsd_last20pct_mean_A"])
score += 0.25 * pct_rank_low_good(all_df["ligand_rmsd_last20pct_mean_A"])
score += 0.15 * pct_rank_high_good(all_df["pl_contacts_4A_last20pct_mean"])
score += 0.15 * pct_rank_low_good(all_df["pl_min_distance_last20pct_mean_A"])
score += 0.10 * pct_rank_low_good(all_df["ca_rmsf_mean_A"])
score += 0.10 * pct_rank_low_good(all_df["protein_rg_last20pct_std_A"])

all_df["md_screen_score"] = score
all_df = all_df.sort_values("md_screen_score", ascending=True)
all_df["md_screen_rank"] = np.arange(1, len(all_df) + 1)

all_df["screen_group"] = "lower_priority"
all_df.loc[all_df["md_screen_rank"] <= 20, "screen_group"] = "recommended"
all_df.loc[
    (all_df["md_screen_rank"] > 20) & (all_df["md_screen_rank"] <= 50),
    "screen_group",
] = "backup"

cols_first = [
    "md_screen_rank",
    "screen_group",
    "candidate",
    "md_screen_score",
    "sim_ns",
    "n_frames",
    "ca_rmsd_last20pct_mean_A",
    "ligand_rmsd_last20pct_mean_A",
    "pl_min_distance_last20pct_mean_A",
    "pl_contacts_4A_last20pct_mean",
    "ca_rmsf_mean_A",
    "ca_rmsf_max_A",
    "protein_rg_last20pct_mean_A",
    "protein_rg_last20pct_std_A",
]

cols = [c for c in cols_first if c in all_df.columns]
cols += [c for c in all_df.columns if c not in cols]

ranked = all_df[cols]

all_out = TABLE_DIR / "md_2ns_summary_all.csv"
ranked_out = TABLE_DIR / "md_2ns_ranked_candidates.csv"
top20_out = TABLE_DIR / "md_2ns_top20_candidates.csv"
top50_out = TABLE_DIR / "md_2ns_top50_candidates.csv"

all_df.to_csv(all_out, index=False)
ranked.to_csv(ranked_out, index=False)
ranked.head(20).to_csv(top20_out, index=False)
ranked.head(50).to_csv(top50_out, index=False)

print("Wrote:", all_out)
print("Wrote:", ranked_out)
print("Wrote:", top20_out)
print("Wrote:", top50_out)
print()
print(ranked.head(20).to_string(index=False))