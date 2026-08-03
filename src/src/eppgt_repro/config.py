from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class PathsConfig:
    train_csv: str
    val_csv: str
    word2vec_path: str | None
    legacy_state_dict_path: str | None
    save_dir: str
    external_val_csv: str | None = None


@dataclass
class ModelConfig:
    protein_dim: int
    atom_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dropout: float


@dataclass
class Word2VecConfig:
    k: int
    vector_size: int
    window: int
    min_count: int
    epochs: int
    workers: int


@dataclass
class ProteinEmbeddingConfig:
    mode: str = "word2vec"
    manifest_path: str | None = None
    embedding_dir: str | None = None
    hash_column: str = "sequence_hash"
    path_column: str = "embedding_path"
    validate_lengths: bool = True
    cache_embeddings: bool = True
    cache_size: int = 256


@dataclass
class TrainConfig:
    epochs: int
    batch_size_per_gpu: int
    optimizer: str
    lr: float
    weight_decay: float
    label_smoothing: float
    num_workers: int
    amp: bool
    seed: int
    grad_clip: float
    scheduler: str
    warmup_epochs: int
    early_stop_patience: int
    classification_loss_weight: float = 0.5
    ranking_loss_weight: float = 1.0
    ranking_margin: float = 0.2
    ranking_max_pairs_per_group: int = 4096
    substrate_batching: bool = True
    best_metric: str = "mrr"
    best_metric_source: str = "val"
    finetune_mode: str = "full"
    eval_before_training: bool = False


@dataclass
class RuntimeConfig:
    ddp: bool
    device: str
    resume: str | None
    log_interval: int
    init_checkpoint: str | None = None


@dataclass
class EPPGTConfig:
    run_name: str
    preset: str
    paths: PathsConfig
    model: ModelConfig
    word2vec: Word2VecConfig
    protein_embedding: ProteinEmbeddingConfig
    train: TrainConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EPPGTConfig":
        protein_embedding_payload = payload.get("protein_embedding", {})
        return cls(
            run_name=payload.get("run_name", ""),
            preset=payload.get("preset", "default"),
            paths=PathsConfig(**payload["paths"]),
            model=ModelConfig(**payload["model"]),
            word2vec=Word2VecConfig(**payload["word2vec"]),
            protein_embedding=ProteinEmbeddingConfig(**protein_embedding_payload),
            train=TrainConfig(**payload["train"]),
            runtime=RuntimeConfig(**payload["runtime"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.preset}_{stamp}"


def default_config_path(preset: str) -> Path:
    path = CONFIG_DIR / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown preset '{preset}': {path}")
    return path


def load_config(path: str | Path) -> EPPGTConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return EPPGTConfig.from_dict(payload)


def save_config(config: EPPGTConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
