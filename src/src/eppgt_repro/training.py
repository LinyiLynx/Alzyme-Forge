from __future__ import annotations

import json
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .config import EPPGTConfig, default_config_path, load_config, save_config
from .data import EsmcEmbeddingStore, PairDataset, collate_pairs, load_pairs_csv, normalize_pair_dataframe
from .gt_screening import GT_OUTPUT_COLUMNS, apply_strict_gt_gate, summarize_gt_gate
from .metrics import compute_binary_metrics, compute_grouped_ranking_metrics
from .model import build_model
from .optim import build_optimizer, build_scheduler
from .word2vec_utils import ensure_word2vec_artifact, load_word2vec_model


@dataclass
class RuntimeState:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_runtime(config: EPPGTConfig) -> RuntimeState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = config.runtime.ddp and world_size > 1

    requested_cuda = config.runtime.device.startswith("cuda")
    use_cuda = requested_cuda and torch.cuda.is_available()

    if distributed and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)

    if use_cuda:
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rank = dist.get_rank() if distributed else 0
    return RuntimeState(
        distributed=distributed,
        rank=rank,
        world_size=world_size if distributed else 1,
        local_rank=local_rank,
        device=device,
    )


def cleanup_runtime(runtime: RuntimeState) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(runtime: RuntimeState) -> None:
    if runtime.distributed:
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


EVAL_METRIC_KEYS = (
    "auroc",
    "auprc",
    "precision",
    "recall",
    "ranking_group_count",
    "ranking_valid_group_count",
    "ranking_mean_group_size",
    "ranking_median_group_size",
    "ranking_mean_positive_count",
    "mean_group_auroc",
    "top1_hit",
    "top5_hit",
    "top10_hit",
    "top50_hit",
    "top100_hit",
    "mrr",
    "ndcg",
    "mean_positive_rank",
)


def prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def select_best_metric(
    val_metrics: dict[str, float],
    external_metrics: dict[str, float] | None,
    config: EPPGTConfig,
) -> tuple[str, float, str]:
    selected_source = config.train.best_metric_source
    selected_metrics = val_metrics if selected_source == "val" else external_metrics
    if selected_metrics is None:
        raise ValueError("best_metric_source='external' was selected but no external validation loader exists")
    best_metric_key = config.train.best_metric
    best_metric_value = float(selected_metrics.get(best_metric_key, math.nan))
    if math.isnan(best_metric_value):
        best_metric_key = "auroc"
        best_metric_value = float(selected_metrics.get(best_metric_key, -math.inf))
    return best_metric_key, best_metric_value, selected_source


def protein_embedding_mode(config: EPPGTConfig) -> str:
    mode = config.protein_embedding.mode.lower()
    if mode not in {"word2vec", "esmc"}:
        raise ValueError("protein_embedding.mode must be 'word2vec' or 'esmc'")
    return mode


def apply_protein_embedding_overrides(
    config: EPPGTConfig,
    mode: str | None = None,
    manifest_path: str | None = None,
    embedding_dir: str | None = None,
    cache_embeddings: bool | None = None,
    cache_size: int | None = None,
) -> None:
    if mode is not None:
        config.protein_embedding.mode = mode
    if manifest_path is not None:
        config.protein_embedding.manifest_path = manifest_path
    if embedding_dir is not None:
        config.protein_embedding.embedding_dir = embedding_dir
    if cache_embeddings is not None:
        config.protein_embedding.cache_embeddings = cache_embeddings
    if cache_size is not None:
        config.protein_embedding.cache_size = cache_size


def infer_esmc_embedding_dim(store: EsmcEmbeddingStore, dataframes: list[pd.DataFrame]) -> int:
    for df in dataframes:
        if df is None or df.empty:
            continue
        tensor = store.get(str(df.iloc[0]["sequence"]))
        return int(tensor.shape[1])
    raise ValueError("Cannot infer ESMC embedding dimension from empty dataframes")


def build_pair_dataset(
    dataframe: pd.DataFrame,
    config: EPPGTConfig,
    word2vec_model=None,
    esmc_store: EsmcEmbeddingStore | None = None,
    include_labels: bool = True,
    include_metadata: bool = False,
) -> PairDataset:
    if protein_embedding_mode(config) == "esmc":
        if esmc_store is None:
            raise ValueError("ESMC mode requires an EsmcEmbeddingStore")
        return PairDataset(
            dataframe,
            include_labels=include_labels,
            include_metadata=include_metadata,
            protein_embedding_store=esmc_store,
        )
    return PairDataset(
        dataframe,
        word2vec_model,
        config.word2vec,
        include_labels=include_labels,
        include_metadata=include_metadata,
    )


def build_metrics_row(
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    external_metrics: dict[str, float] | None,
    config: EPPGTConfig,
) -> dict[str, float | str | int]:
    best_metric_key, best_metric_value, selected_source = select_best_metric(val_metrics, external_metrics, config)
    external_values = (
        prefix_metrics({key: external_metrics.get(key, math.nan) for key in EVAL_METRIC_KEYS}, "external_")
        if external_metrics is not None
        else {f"external_{key}": math.nan for key in EVAL_METRIC_KEYS}
    )
    return {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_classification_loss": train_metrics["classification_loss"],
        "train_ranking_loss": train_metrics["ranking_loss"],
        "train_ranking_pairs": train_metrics["ranking_pairs"],
        "val_loss": val_metrics["loss"],
        **{key: val_metrics.get(key, math.nan) for key in EVAL_METRIC_KEYS},
        "external_loss": external_metrics["loss"] if external_metrics is not None else math.nan,
        **external_values,
        "best_metric": best_metric_key,
        "best_metric_value": best_metric_value,
        "best_metric_source": selected_source,
    }


def apply_finetune_mode(model: nn.Module, mode: str) -> dict[str, Any]:
    if mode == "full":
        for _name, param in model.named_parameters():
            param.requires_grad = True
    elif mode == "head-only":
        trainable_prefixes = ("decoder.dense3.",)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif mode == "mlp-head":
        trainable_prefixes = ("decoder.dense1.", "decoder.dense2.", "decoder.dense3.")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    else:
        raise ValueError("train.finetune_mode must be 'full', 'head-only', or 'mlp-head'")

    total_params = 0
    trainable_params = 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
            trainable_names.append(name)
    if trainable_params == 0:
        raise ValueError(f"finetune_mode={mode!r} left no trainable parameters")
    return {
        "finetune_mode": mode,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
        "trainable_parameter_names": trainable_names,
    }


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise ValueError("Checkpoint does not contain a supported state dict")


def remap_legacy_key(key: str) -> str:
    key = key.removeprefix("module.")
    key = re.sub(r"\bblks\.block(\d+)\b", r"blocks.\1", key)
    key = key.replace(".W_q.", ".w_q.")
    key = key.replace(".W_k.", ".w_k.")
    key = key.replace(".W_v.", ".w_v.")
    key = key.replace(".W_o.", ".w_o.")
    key = key.replace(".ln.", ".layer_norm.")
    return key


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {remap_legacy_key(key): value for key, value in state_dict.items()}


def load_checkpoint_file(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_weights(model: nn.Module, checkpoint_or_path: Any, strict: bool = True) -> dict[str, Any] | None:
    checkpoint = load_checkpoint_file(checkpoint_or_path) if isinstance(checkpoint_or_path, (str, Path)) else checkpoint_or_path
    state_dict = extract_state_dict(checkpoint)
    attempts = [state_dict, remap_legacy_state_dict(state_dict)]
    if any(key.startswith("module.") for key in state_dict):
        attempts.append({key[len("module.") :]: value for key, value in state_dict.items()})
    else:
        attempts.append({f"module.{key}": value for key, value in state_dict.items()})
    last_error = None
    for attempt in attempts:
        try:
            model.load_state_dict(attempt, strict=strict)
            return checkpoint if isinstance(checkpoint, dict) else None
        except RuntimeError as error:
            last_error = error
    raise last_error  # type: ignore[misc]


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    config: EPPGTConfig,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config.to_dict(),
    }
    torch.save(payload, path)


def reduce_scalar(value: float, count: int, runtime: RuntimeState) -> float:
    if not runtime.distributed:
        return value / max(count, 1)
    tensor = torch.tensor([value, count], dtype=torch.float64, device=runtime.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor[0] / tensor[1].clamp(min=1)).item()


def gather_lists(values: list, runtime: RuntimeState) -> list:
    if not runtime.distributed:
        return values
    gathered = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, values)
    merged = []
    for chunk in gathered:
        merged.extend(chunk)
    return merged


def move_batch(batch, device: torch.device):
    moved = []
    for item in batch:
        if hasattr(item, "to"):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def unpack_labeled_batch(batch, device: torch.device):
    moved = move_batch(batch, device)
    if len(moved) == 6:
        compounds, adjs, proteins, labels, atom_lengths, protein_lengths = moved
        metadata = None
    elif len(moved) == 7:
        compounds, adjs, proteins, labels, atom_lengths, protein_lengths, metadata = moved
    else:
        raise ValueError(f"Expected labeled batch with 6 or 7 items, got {len(moved)}")
    return compounds, adjs, proteins, labels, atom_lengths, protein_lengths, metadata


def unpack_unlabeled_batch(batch, device: torch.device):
    moved = move_batch(batch, device)
    if len(moved) == 5:
        compounds, adjs, proteins, atom_lengths, protein_lengths = moved
    elif len(moved) == 6:
        compounds, adjs, proteins, atom_lengths, protein_lengths, _metadata = moved
    else:
        raise ValueError(f"Expected unlabeled batch with 5 or 6 items, got {len(moved)}")
    return compounds, adjs, proteins, atom_lengths, protein_lengths


class SubstrateBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: PairDataset,
        batch_size: int,
        shuffle: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for index, smiles in enumerate(dataset.data["smiles"].astype(str).tolist()):
            groups[smiles].append(index)
        self.groups = groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        group_keys = list(self.groups)
        if self.shuffle:
            rng.shuffle(group_keys)
        for key in group_keys:
            indices = list(self.groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _rank_batches(self) -> list[list[int]]:
        batches = self._build_batches()
        if self.world_size > 1 and batches:
            remainder = len(batches) % self.world_size
            if remainder:
                batches.extend(batches[: self.world_size - remainder])
        return batches[self.rank :: self.world_size]

    def __iter__(self):
        for batch in self._rank_batches():
            yield batch

    def __len__(self) -> int:
        return len(self._rank_batches())


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    runtime: RuntimeState,
    substrate_batching: bool = False,
    seed: int = 0,
):
    if substrate_batching:
        batch_sampler = SubstrateBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            rank=runtime.rank if runtime.distributed else 0,
            world_size=runtime.world_size if runtime.distributed else 1,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_pairs,
            pin_memory=runtime.device.type == "cuda",
        )
    sampler = None
    if runtime.distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=runtime.device.type == "cuda",
    )


def positive_scores(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    metadata: list[dict[str, str]] | None,
    margin: float,
    max_pairs_per_group: int,
) -> tuple[torch.Tensor, int]:
    if metadata is None:
        return scores.new_zeros(()), 0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        groups[str(item["smiles"])].append(index)

    losses = []
    pair_count = 0
    for indices in groups.values():
        index_tensor = torch.tensor(indices, dtype=torch.long, device=scores.device)
        group_scores = scores.index_select(0, index_tensor)
        group_labels = labels.index_select(0, index_tensor)
        pos_scores = group_scores[group_labels == 1]
        neg_scores = group_scores[group_labels == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        pos_grid = pos_scores[:, None].expand(-1, neg_scores.numel()).reshape(-1)
        neg_grid = neg_scores[None, :].expand(pos_scores.numel(), -1).reshape(-1)
        if max_pairs_per_group > 0 and pos_grid.numel() > max_pairs_per_group:
            choice = torch.randperm(pos_grid.numel(), device=scores.device)[:max_pairs_per_group]
            pos_grid = pos_grid.index_select(0, choice)
            neg_grid = neg_grid.index_select(0, choice)
        target = torch.ones_like(pos_grid)
        losses.append(F.margin_ranking_loss(pos_grid, neg_grid, target, margin=margin, reduction="mean"))
        pair_count += int(pos_grid.numel())

    if not losses:
        return scores.new_zeros(()), 0
    return torch.stack(losses).mean(), pair_count


def compute_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: list[dict[str, str]] | None,
    criterion,
    config: EPPGTConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    classification_loss = criterion(logits, labels)
    ranking_loss, ranking_pair_count = pairwise_ranking_loss(
        scores=positive_scores(logits),
        labels=labels,
        metadata=metadata,
        margin=config.train.ranking_margin,
        max_pairs_per_group=config.train.ranking_max_pairs_per_group,
    )
    total_loss = (
        config.train.classification_loss_weight * classification_loss
        + config.train.ranking_loss_weight * ranking_loss
    )
    return total_loss, classification_loss, ranking_loss, ranking_pair_count


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, runtime: RuntimeState, config: EPPGTConfig, epoch: int):
    model.train()
    total_loss = 0.0
    total_classification_loss = 0.0
    total_ranking_loss = 0.0
    total_ranking_pairs = 0
    total_count = 0
    autocast_enabled = scaler is not None
    if runtime.distributed and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(epoch)

    for step, batch in enumerate(loader, start=1):
        compounds, adjs, proteins, labels, atom_lengths, protein_lengths, metadata = unpack_labeled_batch(batch, runtime.device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(proteins, protein_lengths, compounds, adjs, atom_lengths)
            loss, classification_loss, ranking_loss, ranking_pair_count = compute_training_loss(
                logits, labels, metadata, criterion, config
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_classification_loss += classification_loss.detach().item() * batch_size
        total_ranking_loss += ranking_loss.detach().item() * batch_size
        total_ranking_pairs += ranking_pair_count
        total_count += batch_size
        if runtime.is_main and step % config.runtime.log_interval == 0:
            print(
                "[train] "
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.detach().item():.5f} "
                f"cls={classification_loss.detach().item():.5f} "
                f"rank={ranking_loss.detach().item():.5f} "
                f"pairs={ranking_pair_count}",
                flush=True,
            )

    return {
        "loss": reduce_scalar(total_loss, total_count, runtime),
        "classification_loss": reduce_scalar(total_classification_loss, total_count, runtime),
        "ranking_loss": reduce_scalar(total_ranking_loss, total_count, runtime),
        "ranking_pairs": int(sum(gather_lists([total_ranking_pairs], runtime))),
    }


@torch.no_grad()
def evaluate_loader(model, loader, criterion, runtime: RuntimeState):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_labels = []
    all_scores = []
    all_groups = []
    for batch in loader:
        compounds, adjs, proteins, labels, atom_lengths, protein_lengths, metadata = unpack_labeled_batch(batch, runtime.device)
        logits = model(proteins, protein_lengths, compounds, adjs, atom_lengths)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        total_loss += loss.detach().item() * labels.size(0)
        total_count += labels.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_scores.extend(probabilities.detach().cpu().tolist())
        if metadata is not None:
            all_groups.extend(str(item["smiles"]) for item in metadata)

    merged_labels = gather_lists(all_labels, runtime)
    merged_scores = gather_lists(all_scores, runtime)
    merged_groups = gather_lists(all_groups, runtime)
    mean_loss = reduce_scalar(total_loss, total_count, runtime)
    metrics = compute_binary_metrics(merged_labels, merged_scores)
    if merged_groups:
        metrics.update(compute_grouped_ranking_metrics(merged_labels, merged_scores, merged_groups))
    metrics["loss"] = mean_loss
    return metrics


@torch.no_grad()
def predict_loader(model, loader, runtime: RuntimeState) -> tuple[list[int], list[float]]:
    model.eval()
    predictions = []
    scores = []
    for batch in loader:
        compounds, adjs, proteins, atom_lengths, protein_lengths = unpack_unlabeled_batch(batch, runtime.device)
        logits = model(proteins, protein_lengths, compounds, adjs, atom_lengths)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions.extend((probabilities >= 0.5).long().cpu().tolist())
        scores.extend(probabilities.cpu().tolist())
    return gather_lists(predictions, runtime), gather_lists(scores, runtime)


def parse_fasta(path: str | Path) -> pd.DataFrame:
    records = []
    header: str | None = None
    sequence_parts: list[str] = []

    def flush_record() -> None:
        if header is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").replace("*", "").upper()
        if not sequence:
            return
        tokens = header.split()
        protein_id = tokens[0]
        metadata = {}
        for token in tokens[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                metadata[key] = value
        records.append(
            {
                "protein_id": protein_id,
                "sequence": sequence,
                "header": header,
                "gene": metadata.get("Gene", ""),
                "mrna": metadata.get("mRNA", ""),
                "ori_id": metadata.get("OriID", ""),
                "ori_gene_id": metadata.get("OriGeneID", ""),
                "ori_seq_id": metadata.get("OriSeqID", ""),
                "length": len(sequence),
            }
        )

    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_record()
                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line)
    flush_record()
    return pd.DataFrame(records)


def load_smiles_file(path: str | Path) -> pd.DataFrame:
    records = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                first, second = line.split("\t", 1)
            elif "," in line:
                first, second = line.split(",", 1)
            else:
                first, second = f"compound_{index}", line
            first = first.strip()
            second = second.strip()
            if first and second:
                records.append({"compound_id": first, "smiles": second})
    if not records:
        raise ValueError(f"No SMILES records found in {path}")
    return pd.DataFrame(records)


def resolve_word2vec_path(checkpoint_dir: Path, config: EPPGTConfig, override: str | None = None) -> Path:
    if override:
        return Path(override)
    artifact_path = checkpoint_dir / "word2vec.model"
    if artifact_path.exists():
        return artifact_path
    if config.paths.word2vec_path:
        return Path(config.paths.word2vec_path)
    raise FileNotFoundError("Unable to resolve a word2vec model path")


def train_command(
    config: EPPGTConfig,
    run_name: str | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    load_legacy: bool = False,
) -> Path:
    runtime = setup_runtime(config)
    try:
        seed_everything(config.train.seed + runtime.rank)
        run_name = run_name or config.resolve_run_name()
        config.run_name = run_name
        run_dir = ensure_directory(Path(config.paths.save_dir) / run_name)

        train_df = load_pairs_csv(config.paths.train_csv, require_label=True, max_rows=max_train_rows)
        val_df = load_pairs_csv(config.paths.val_csv, require_label=True, max_rows=max_val_rows)
        external_val_df = (
            load_pairs_csv(config.paths.external_val_csv, require_label=True)
            if config.paths.external_val_csv
            else None
        )
        if config.train.best_metric_source not in {"val", "external"}:
            raise ValueError("train.best_metric_source must be 'val' or 'external'")
        if config.train.best_metric_source == "external" and external_val_df is None:
            raise ValueError("train.best_metric_source='external' requires paths.external_val_csv")

        word2vec_model = None
        esmc_store = None
        if protein_embedding_mode(config) == "word2vec":
            word2vec_artifact = run_dir / "word2vec.model"
            if runtime.is_main:
                ensure_word2vec_artifact(
                    sequences=pd.concat([train_df["sequence"], val_df["sequence"]]).tolist(),
                    config=config.word2vec,
                    artifact_path=word2vec_artifact,
                    existing_path=config.paths.word2vec_path,
                )
            barrier(runtime)
            word2vec_model = load_word2vec_model(word2vec_artifact)
        else:
            esmc_store = EsmcEmbeddingStore(config.protein_embedding)
            config.model.protein_dim = infer_esmc_embedding_dim(
                esmc_store,
                [train_df, val_df, external_val_df],
            )
            if runtime.is_main:
                print(f"[protein_embedding] mode=esmc protein_dim={config.model.protein_dim}", flush=True)
        if runtime.is_main:
            save_config(config, run_dir / "resolved_config.yaml")
        barrier(runtime)

        train_dataset = build_pair_dataset(
            train_df,
            config,
            word2vec_model=word2vec_model,
            esmc_store=esmc_store,
            include_labels=True,
            include_metadata=True,
        )
        val_dataset = build_pair_dataset(
            val_df,
            config,
            word2vec_model=word2vec_model,
            esmc_store=esmc_store,
            include_labels=True,
            include_metadata=True,
        )
        external_val_dataset = (
            build_pair_dataset(
                external_val_df,
                config,
                word2vec_model=word2vec_model,
                esmc_store=esmc_store,
                include_labels=True,
                include_metadata=True,
            )
            if external_val_df is not None
            else None
        )
        train_loader = build_dataloader(
            train_dataset,
            config.train.batch_size_per_gpu,
            config.train.num_workers,
            True,
            runtime,
            substrate_batching=config.train.substrate_batching,
            seed=config.train.seed,
        )
        val_loader = build_dataloader(
            val_dataset,
            config.train.batch_size_per_gpu,
            config.train.num_workers,
            False,
            runtime,
            substrate_batching=False,
            seed=config.train.seed,
        )
        external_val_loader = (
            build_dataloader(
                external_val_dataset,
                config.train.batch_size_per_gpu,
                config.train.num_workers,
                False,
                runtime,
                substrate_batching=False,
                seed=config.train.seed,
            )
            if external_val_dataset is not None
            else None
        )

        model = build_model(config.model).to(runtime.device)
        resume_payload = None
        if config.runtime.resume and config.runtime.init_checkpoint:
            raise ValueError("Use either runtime.resume or runtime.init_checkpoint, not both")
        if load_legacy and config.runtime.init_checkpoint:
            raise ValueError("Use either --load-legacy or --init-checkpoint, not both")
        if load_legacy and config.paths.legacy_state_dict_path:
            load_model_weights(model, config.paths.legacy_state_dict_path, strict=True)
        if config.runtime.init_checkpoint:
            load_model_weights(model, config.runtime.init_checkpoint, strict=True)
        if config.runtime.resume:
            resume_payload = load_model_weights(model, config.runtime.resume, strict=True)
        finetune_summary = apply_finetune_mode(model, config.train.finetune_mode)
        if runtime.is_main:
            (run_dir / "finetune_summary.json").write_text(
                json.dumps(finetune_summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[finetune] {json.dumps(finetune_summary, ensure_ascii=False)}", flush=True)

        if runtime.distributed:
            model = DDP(model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None)

        optimizer = build_optimizer(model, config.train)
        scheduler = build_scheduler(optimizer, config.train, max(len(train_loader), 1))
        scaler = torch.cuda.amp.GradScaler(enabled=(config.train.amp and runtime.device.type == "cuda"))
        criterion = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)

        start_epoch = 0
        best_metric = -math.inf
        if resume_payload:
            if resume_payload.get("optimizer") is not None:
                optimizer.load_state_dict(resume_payload["optimizer"])
            if scheduler is not None and resume_payload.get("scheduler") is not None:
                scheduler.load_state_dict(resume_payload["scheduler"])
            if scaler.is_enabled() and resume_payload.get("scaler") is not None:
                scaler.load_state_dict(resume_payload["scaler"])
            start_epoch = int(resume_payload.get("epoch", -1)) + 1
            best_metric = float(resume_payload.get("best_metric", -math.inf))

        metrics_path = run_dir / "metrics.csv"
        if runtime.is_main and not metrics_path.exists():
            metric_columns = [
                "epoch",
                "train_loss",
                "train_classification_loss",
                "train_ranking_loss",
                "train_ranking_pairs",
                "val_loss",
                *EVAL_METRIC_KEYS,
                "external_loss",
                *(f"external_{key}" for key in EVAL_METRIC_KEYS),
                "best_metric",
                "best_metric_value",
                "best_metric_source",
            ]
            pd.DataFrame(
                columns=metric_columns
            ).to_csv(metrics_path, index=False)

        if config.train.eval_before_training and start_epoch == 0:
            val_metrics = evaluate_loader(model, val_loader, criterion, runtime)
            external_metrics = (
                evaluate_loader(model, external_val_loader, criterion, runtime)
                if external_val_loader is not None
                else None
            )
            if runtime.is_main:
                train_metrics = {
                    "loss": math.nan,
                    "classification_loss": math.nan,
                    "ranking_loss": math.nan,
                    "ranking_pairs": 0,
                }
                row = build_metrics_row(-1, train_metrics, val_metrics, external_metrics, config)
                pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=False, index=False)
                print(f"[eval] epoch=-1 {json.dumps(row, ensure_ascii=False)}", flush=True)
                initial_metric_value = float(row["best_metric_value"])
                save_checkpoint(
                    run_dir / "initial.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler if scaler.is_enabled() else None,
                    -1,
                    initial_metric_value,
                    config,
                )
                if initial_metric_value > best_metric:
                    best_metric = initial_metric_value
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler if scaler.is_enabled() else None,
                        -1,
                        best_metric,
                        config,
                    )
            barrier(runtime)

        patience = 0
        for epoch in range(start_epoch, config.train.epochs):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler if scaler.is_enabled() else None,
                criterion,
                runtime,
                config,
                epoch,
            )
            val_metrics = evaluate_loader(model, val_loader, criterion, runtime)
            external_metrics = (
                evaluate_loader(model, external_val_loader, criterion, runtime)
                if external_val_loader is not None
                else None
            )

            if runtime.is_main:
                row = build_metrics_row(epoch, train_metrics, val_metrics, external_metrics, config)
                best_metric_value = float(row["best_metric_value"])
                pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=False, index=False)
                print(f"[eval] epoch={epoch} {json.dumps(row, ensure_ascii=False)}", flush=True)
                save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, scaler if scaler.is_enabled() else None, epoch, best_metric, config)
                if best_metric_value > best_metric:
                    best_metric = best_metric_value
                    save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, scaler if scaler.is_enabled() else None, epoch, best_metric, config)
                    patience = 0
                else:
                    patience += 1
                should_stop = patience >= config.train.early_stop_patience
            else:
                should_stop = False
            if runtime.distributed:
                stop_tensor = torch.tensor(int(should_stop), device=runtime.device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())
                barrier(runtime)
            if should_stop:
                if runtime.is_main:
                    print(f"Early stopping at epoch {epoch}", flush=True)
                break
        return run_dir
    finally:
        cleanup_runtime(runtime)


def load_config_from_checkpoint_or_file(checkpoint_path: str | Path, config_path: str | None = None) -> tuple[EPPGTConfig, dict[str, Any] | None]:
    checkpoint = load_checkpoint_file(checkpoint_path)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        return EPPGTConfig.from_dict(checkpoint["config"]), checkpoint
    if config_path:
        return load_config(config_path), checkpoint if isinstance(checkpoint, dict) else None
    return load_config(default_config_path("default")), checkpoint if isinstance(checkpoint, dict) else None


def eval_command(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    config_path: str | None = None,
    word2vec_path: str | None = None,
    protein_embedding_mode_override: str | None = None,
    protein_embedding_manifest: str | None = None,
    protein_embedding_dir: str | None = None,
    protein_embedding_cache_embeddings: bool | None = None,
    protein_embedding_cache_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_rows: int | None = None,
    device: str | None = None,
    output_path: str | None = None,
) -> Path:
    config, checkpoint = load_config_from_checkpoint_or_file(checkpoint_path, config_path)
    apply_protein_embedding_overrides(
        config,
        mode=protein_embedding_mode_override,
        manifest_path=protein_embedding_manifest,
        embedding_dir=protein_embedding_dir,
        cache_embeddings=protein_embedding_cache_embeddings,
        cache_size=protein_embedding_cache_size,
    )
    if device is not None:
        config.runtime.device = device
    runtime = setup_runtime(config)
    try:
        dataset_df = load_pairs_csv(csv_path, require_label=True, max_rows=max_rows)
        checkpoint_dir = Path(checkpoint_path).resolve().parent
        word2vec_model = None
        esmc_store = None
        if protein_embedding_mode(config) == "word2vec":
            resolved_word2vec = resolve_word2vec_path(checkpoint_dir, config, override=word2vec_path)
            word2vec_model = load_word2vec_model(resolved_word2vec)
        else:
            esmc_store = EsmcEmbeddingStore(config.protein_embedding)
            config.model.protein_dim = infer_esmc_embedding_dim(esmc_store, [dataset_df])
        model = build_model(config.model).to(runtime.device)
        load_model_weights(model, checkpoint or checkpoint_path, strict=True)
        if runtime.distributed:
            model = DDP(model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None)
        dataset = build_pair_dataset(
            dataset_df,
            config,
            word2vec_model=word2vec_model,
            esmc_store=esmc_store,
            include_labels=True,
            include_metadata=True,
        )
        loader = build_dataloader(
            dataset,
            batch_size or config.train.batch_size_per_gpu,
            num_workers if num_workers is not None else config.train.num_workers,
            False,
            runtime,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing)
        metrics = evaluate_loader(model, loader, criterion, runtime)
        target = Path(output_path) if output_path else checkpoint_dir / f"eval_{Path(csv_path).stem}.json"
        if runtime.is_main:
            target.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return target
    finally:
        cleanup_runtime(runtime)


def predict_command(
    checkpoint_path: str | Path,
    csv_path: str | Path,
    config_path: str | None = None,
    word2vec_path: str | None = None,
    protein_embedding_mode_override: str | None = None,
    protein_embedding_manifest: str | None = None,
    protein_embedding_dir: str | None = None,
    protein_embedding_cache_embeddings: bool | None = None,
    protein_embedding_cache_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_rows: int | None = None,
    device: str | None = None,
    output_path: str | None = None,
) -> Path:
    config, checkpoint = load_config_from_checkpoint_or_file(checkpoint_path, config_path)
    apply_protein_embedding_overrides(
        config,
        mode=protein_embedding_mode_override,
        manifest_path=protein_embedding_manifest,
        embedding_dir=protein_embedding_dir,
        cache_embeddings=protein_embedding_cache_embeddings,
        cache_size=protein_embedding_cache_size,
    )
    if device is not None:
        config.runtime.device = device
    runtime = setup_runtime(config)
    try:
        raw_df = pd.read_csv(csv_path)
        normalized = normalize_pair_dataframe(raw_df, require_label=False)
        if max_rows is not None:
            normalized = normalized.head(max_rows).reset_index(drop=True)
            raw_df = raw_df.head(max_rows).reset_index(drop=True)
        checkpoint_dir = Path(checkpoint_path).resolve().parent
        word2vec_model = None
        esmc_store = None
        if protein_embedding_mode(config) == "word2vec":
            resolved_word2vec = resolve_word2vec_path(checkpoint_dir, config, override=word2vec_path)
            word2vec_model = load_word2vec_model(resolved_word2vec)
        else:
            esmc_store = EsmcEmbeddingStore(config.protein_embedding)
            config.model.protein_dim = infer_esmc_embedding_dim(esmc_store, [normalized])
        model = build_model(config.model).to(runtime.device)
        load_model_weights(model, checkpoint or checkpoint_path, strict=True)
        if runtime.distributed:
            model = DDP(model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None)
        dataset = build_pair_dataset(
            normalized,
            config,
            word2vec_model=word2vec_model,
            esmc_store=esmc_store,
            include_labels=False,
        )
        loader = build_dataloader(
            dataset,
            batch_size or config.train.batch_size_per_gpu,
            num_workers if num_workers is not None else config.train.num_workers,
            False,
            runtime,
        )
        predictions, scores = predict_loader(model, loader, runtime)
        target = Path(output_path) if output_path else checkpoint_dir / f"predictions_{Path(csv_path).stem}.csv"
        if runtime.is_main:
            result = raw_df.copy()
            result["prediction"] = predictions
            result["score"] = scores
            result.to_csv(target, index=False)
            print(f"Saved predictions to {target}")
        return target
    finally:
        cleanup_runtime(runtime)


def screen_proteome_command(
    checkpoint_path: str | Path,
    pep_path: str | Path,
    smiles_path: str | Path,
    config_path: str | None = None,
    word2vec_path: str | None = None,
    protein_embedding_mode_override: str | None = None,
    protein_embedding_manifest: str | None = None,
    protein_embedding_dir: str | None = None,
    protein_embedding_cache_embeddings: bool | None = None,
    protein_embedding_cache_size: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_path: str | None = None,
    top_k: int = 50,
    gt_gate: bool = True,
    gt_reference_dir: str | None = "data/gt_reference",
    min_aa: int | None = None,
    max_aa: int | None = None,
    max_proteins: int | None = None,
    device: str | None = None,
) -> Path:
    config, checkpoint = load_config_from_checkpoint_or_file(checkpoint_path, config_path)
    apply_protein_embedding_overrides(
        config,
        mode=protein_embedding_mode_override,
        manifest_path=protein_embedding_manifest,
        embedding_dir=protein_embedding_dir,
        cache_embeddings=protein_embedding_cache_embeddings,
        cache_size=protein_embedding_cache_size,
    )
    if device is not None:
        config.runtime.device = device
    runtime = setup_runtime(config)
    try:
        loaded_proteins = parse_fasta(pep_path)
        proteins = loaded_proteins.reset_index(drop=True)
        if max_proteins is not None:
            proteins = proteins.head(max_proteins).reset_index(drop=True)
        if proteins.empty:
            raise ValueError("No proteins were loaded from the PEP/FASTA file")

        if gt_gate:
            proteins = apply_strict_gt_gate(proteins, gt_reference_dir)
        else:
            proteins = proteins.copy()
            for column in GT_OUTPUT_COLUMNS:
                proteins[column] = ""
            proteins["gt_candidate"] = "1"
            proteins["gt_tier"] = "gate_disabled"
            proteins["gt_reason"] = "strict GT gate disabled"

        score_eligible = pd.Series(True, index=proteins.index)
        zero_reasons = pd.Series("", index=proteins.index, dtype="object")
        if gt_gate:
            gt_candidate_mask = proteins["gt_candidate"].astype(str) == "1"
            score_eligible &= gt_candidate_mask
            zero_reasons.loc[~gt_candidate_mask] = "strict_gt_gate_failed"
        if min_aa is not None:
            min_fail = proteins["length"] < min_aa
            score_eligible &= ~min_fail
            zero_reasons.loc[min_fail & (zero_reasons == "")] = f"protein_length_below_{min_aa}"
        if max_aa is not None:
            max_fail = proteins["length"] > max_aa
            score_eligible &= ~max_fail
            zero_reasons.loc[max_fail & (zero_reasons == "")] = f"protein_length_above_{max_aa}"
        proteins["model_scored"] = score_eligible.astype(int)
        proteins["screen_zero_reason"] = [
            "model_scored" if eligible else (reason or "not_model_scored")
            for eligible, reason in zip(score_eligible.tolist(), zero_reasons.tolist())
        ]

        compounds = load_smiles_file(smiles_path)
        rows = []
        protein_metadata_columns = GT_OUTPUT_COLUMNS + ["model_scored", "screen_zero_reason"]
        for compound in compounds.to_dict("records"):
            for protein in proteins.to_dict("records"):
                row = {
                    "compound_id": compound["compound_id"],
                    "Substrate_SMILES": compound["smiles"],
                    "Protein_ID": protein["protein_id"],
                    "Protein_Sequence": protein["sequence"],
                    "protein_length": protein["length"],
                    "gene": protein["gene"],
                    "mrna": protein["mrna"],
                    "ori_id": protein["ori_id"],
                    "ori_gene_id": protein["ori_gene_id"],
                    "ori_seq_id": protein["ori_seq_id"],
                    "header": protein["header"],
                }
                for column in protein_metadata_columns:
                    row[column] = protein.get(column, "")
                rows.append(row)
        raw_df = pd.DataFrame(rows)
        score_df = raw_df[raw_df["model_scored"].astype(int) == 1].copy()
        result = raw_df.copy()
        result["prediction"] = 0
        result["score"] = 0.0

        if not score_df.empty:
            normalized = normalize_pair_dataframe(score_df, require_label=False)
            checkpoint_dir = Path(checkpoint_path).resolve().parent
            word2vec_model = None
            esmc_store = None
            if protein_embedding_mode(config) == "word2vec":
                resolved_word2vec = resolve_word2vec_path(checkpoint_dir, config, override=word2vec_path)
                word2vec_model = load_word2vec_model(resolved_word2vec)
            else:
                esmc_store = EsmcEmbeddingStore(config.protein_embedding)
                config.model.protein_dim = infer_esmc_embedding_dim(esmc_store, [normalized])
            model = build_model(config.model).to(runtime.device)
            load_model_weights(model, checkpoint or checkpoint_path, strict=True)
            if runtime.distributed:
                model = DDP(model, device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None)
            dataset = build_pair_dataset(
                normalized,
                config,
                word2vec_model=word2vec_model,
                esmc_store=esmc_store,
                include_labels=False,
            )
            loader = build_dataloader(
                dataset,
                batch_size or config.train.batch_size_per_gpu,
                num_workers if num_workers is not None else config.train.num_workers,
                False,
                runtime,
            )
            predictions, scores = predict_loader(model, loader, runtime)
            if len(scores) != len(score_df):
                raise ValueError(f"Prediction count mismatch: got {len(scores)} scores for {len(score_df)} rows")
            result.loc[score_df.index, "prediction"] = predictions
            result.loc[score_df.index, "score"] = scores

        target = Path(output_path) if output_path else Path("predict") / "proteome_predictions.csv"
        if runtime.is_main:
            target.parent.mkdir(parents=True, exist_ok=True)
            result = result.sort_values(["compound_id", "score"], ascending=[True, False]).reset_index(drop=True)
            result.to_csv(target, index=False)

            top_target = target.with_name(f"{target.stem}_top{top_k}{target.suffix}")
            top_result = result.groupby("compound_id", group_keys=False).head(top_k)
            top_result.to_csv(top_target, index=False)
            gt_summary = summarize_gt_gate(proteins)
            print(f"Loaded proteins: {len(loaded_proteins)}")
            print(f"Output proteins: {len(proteins)}")
            print(f"GT gate enabled: {gt_gate}")
            if gt_gate:
                print(f"GT candidates sent to model: {gt_summary['gt_candidates_total']}")
                print(f"GT tier A: {gt_summary['gt_tier_A']}")
                print(f"GT tier B: {gt_summary['gt_tier_B']}")
                print(f"GT tier C: {gt_summary['gt_tier_C']}")
                print(f"Zero-scored proteins by GT gate: {gt_summary['gt_zeroed_total']}")
            print(f"Proteins sent to model after all gates: {int(score_eligible.sum())}")
            print(f"Zero-scored proteins after all gates: {int((~score_eligible).sum())}")
            print(f"Length filter zeroing: min_aa={min_aa}, max_aa={max_aa}")
            print(f"Loaded compounds: {len(compounds)}")
            print(f"Model-scored pairs: {len(score_df)}")
            print(f"Output pairs: {len(result)}")
            print(f"Saved predictions to {target}")
            print(f"Saved top-{top_k} predictions to {top_target}")
        return target
    finally:
        cleanup_runtime(runtime)
