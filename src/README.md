# EPPGT Repro

`EPPGT_repro` is a rebuilt training package for the public EPP-GT code under `/Volumes/Lynx/AIGT/原始数据准备/EPPGT`.

It keeps the original model behavior:

- 3-mer Word2Vec protein embedding
- 46-dim atom features with `Chem.AddHs`
- 2-layer GCN over molecule graphs
- Transformer-style cross-modal attention for binary interaction prediction

## Layout

```text
EPPGT_repro/
├── artifacts/
├── configs/
├── src/eppgt_repro/
└── tests/
```

## Default data source

The default configs point to:

- `/Volumes/Lynx/AIGT/原始数据准备/Project_EPP_GT/phase3/data/processed/train_data_strict.csv`
- `/Volumes/Lynx/AIGT/原始数据准备/Project_EPP_GT/phase3/data/processed/val_data_strict.csv`

Legacy compatibility assets remain optional:

- `/Volumes/Lynx/AIGT/原始数据准备/EPPGT/model/model_save/word2vec_pretrained.model`
- `/Volumes/Lynx/AIGT/原始数据准备/EPPGT/model/model_save/EPPGT_trained.pt`

## Quick start

Train with the default preset:

```bash
PYTHONPATH=src python -m eppgt_repro train --preset default --run-name baseline
```

Train with the legacy preset:

```bash
PYTHONPATH=src python -m eppgt_repro train --preset legacy --run-name legacy_run
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python -m eppgt_repro eval \
  --checkpoint /Volumes/Lynx/AIGT/EPPGT_repro/artifacts/baseline/best.pt \
  --csv /Volumes/Lynx/AIGT/原始数据准备/Project_EPP_GT/phase3/data/processed/val_data_strict.csv
```

Predict scores for a CSV:

```bash
PYTHONPATH=src python -m eppgt_repro predict \
  --checkpoint /Volumes/Lynx/AIGT/EPPGT_repro/artifacts/baseline/best.pt \
  --csv /Volumes/Lynx/AIGT/原始数据准备/EPPGT/data/gt_val.csv
```

## Multi-GPU

The training CLI supports single-process execution and `torchrun` DDP:

```bash
cd /Volumes/Lynx/AIGT/EPPGT_repro
PYTHONPATH=src torchrun --nproc_per_node=4 -m eppgt_repro train --preset default --run-name ddp_run
```
