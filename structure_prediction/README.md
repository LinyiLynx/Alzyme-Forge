# Structure prediction (Boltz)

- `code/` — multi-GPU Boltz scheduling / input preparation / summarization
- `inputs/` — YAML inputs for top screened enzyme–substrate pairs
- `results/neg_fix_v1_top200/` — prediction outputs used in the AIGT pipeline
- `results/legacy_results/` — earlier top200 run outputs
- Model caches/weights are not shipped; use `code/setup_boltz_cache.sh`
