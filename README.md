# AIGT

**AIGT** (AI Glycosyltransferase) screens enzyme–substrate pairs with an ESMC-conditioned EPP-GT model, then validates top hits with structure prediction (Boltz/AlphaFold-style pipeline) and molecular dynamics.

This repository is a cleaned open-source release of the best-AUC ESMC training run (`0601`) together with the downstream structure/MD pipelines and results.

## Repository layout

```text
AIGT_OpenWeighted/
├── main.py                      # CLI entry (train / eval / predict / screen)
├── requirements.txt
├── configs/                     # Training configs (extrapolation.yaml used by best run)
├── src/                         # EPP-GT + ESMC training package (eppgt_repro)
├── data/
│   ├── raw/                     # Original CAZy / GT metadata sources
│   ├── splits/                  # train / val / test used by best AUC run
│   ├── gt_reference/            # Reference GT sequences / annotations
│   └── esmc_embeddings/         # Precomputed ESMC residue embeddings
├── checkpoints/
│   └── esm_best_auc_0601/       # best.pt, last.pt, metrics.csv, resolved config, test eval
├── logs/
│   └── training/train_0601.log  # Full per-epoch / per-step training log
├── screening/                   # Screening inputs + top-ranked candidates
├── structure_prediction/        # Boltz/AlphaFold pipeline code + results
│   ├── code/
│   ├── inputs/
│   ├── logs/
│   └── results/
├── molecular_dynamics/          # OpenMM/Gromacs-style MD pipeline code + results
│   ├── code/
│   ├── mdp/
│   ├── logs/
│   ├── analysis/
│   └── results/                 # top10 + top200 MD jobs
├── figures/                     # AUC / dataset overview figures for the best run
├── scripts/
│   ├── dataset/                 # Negative sampling / clustering helpers
│   └── figures/                 # Plotting scripts
└── docs/                        # Training notes (ESMC, ranking, extrapolation)
```

## Best AUC model (`0601`)

| Item | Value |
|------|--------|
| Run name | `0601` |
| Protein encoder input | ESMC token embeddings (`protein_dim=1152`) |
| Selection metric | external MRR (`best_metric_source: external`) |
| Held-out test AUROC | **0.740** (`checkpoints/esm_best_auc_0601/eval_test.json`) |
| Checkpoints | `best.pt`, `last.pt` |
| Epoch metrics | `checkpoints/esm_best_auc_0601/metrics.csv` |
| Full train log | `logs/training/train_0601.log` |
| Resolved config | `checkpoints/esm_best_auc_0601/resolved_config.yaml` |

## Quick start

```bash
# Install
pip install -r requirements.txt
pip install -e ./src

# Train (same recipe as best run; paths point to this release layout)
python main.py train \
  --config configs/extrapolation.yaml \
  --run-name esm_best_auc_0601 \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --external-val-csv data/splits/test.csv \
  --protein-embedding-manifest data/esmc_embeddings/manifest.csv \
  --protein-embedding-dir data/esmc_embeddings/embeddings \
  --best-metric-source external

# Evaluate released checkpoint
python main.py eval \
  --checkpoint checkpoints/esm_best_auc_0601/best.pt \
  --csv data/splits/test.csv
```

## Structure prediction

Pipeline scripts live in `structure_prediction/code/`. Inputs and Boltz outputs for the screened top candidates are under `structure_prediction/results/`. Model weight cache is **not** redistributed; download via `structure_prediction/code/setup_boltz_cache.sh`.

## Molecular dynamics

Pipeline scripts live in `molecular_dynamics/code/` with MDP templates in `molecular_dynamics/mdp/`. Job outputs:

- `molecular_dynamics/results/top10/`
- `molecular_dynamics/results/top200/`
- Aggregated analysis: `molecular_dynamics/analysis/`

## Notes

- Original working directories were reorganized for release; behavior matches the `0601` ESMC training code and the AF/MD pipelines used to produce the bundled results.
- `data/splits/` are the pre–neg-fix training splits corresponding to the best-AUC ESMC run.
- Screening lists under `screening/` are the candidate sets that fed structure prediction and MD.

## GitHub packaging notes

To stay within GitHub file-size limits, the following local artifacts are **not** tracked in git (see `.gitignore`):

- `data/esmc_embeddings/embeddings/` (~9 GB ESMC tensors; `manifest.csv` is kept)
- `molecular_dynamics/results/` (trajectories / `.dcd` / restart dumps; code, `mdp/`, and `analysis/` are kept)
- Heavy structure-prediction worktrees / MSA caches under `structure_prediction/results/**/work/`
