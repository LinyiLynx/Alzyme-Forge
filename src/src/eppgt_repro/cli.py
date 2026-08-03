from __future__ import annotations

import argparse
from pathlib import Path

from .config import default_config_path, load_config
from .training import eval_command, predict_command, screen_proteome_command, train_command


def add_protein_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protein-embedding-mode", choices=["word2vec", "esmc"], default=None)
    parser.add_argument("--protein-embedding-manifest", default=None)
    parser.add_argument("--protein-embedding-dir", default=None)
    parser.add_argument("--disable-protein-embedding-cache", action="store_true")
    parser.add_argument("--protein-embedding-cache-size", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eppgt_repro", description="Rebuilt EPP-GT train/eval/predict CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and validate a model")
    train_parser.add_argument("--preset", choices=["default", "legacy"], default="default")
    train_parser.add_argument("--config", default=None)
    train_parser.add_argument("--run-name", default=None)
    train_parser.add_argument("--train-csv", default=None)
    train_parser.add_argument("--val-csv", default=None)
    train_parser.add_argument("--external-val-csv", default=None)
    train_parser.add_argument("--word2vec-path", default=None)
    add_protein_embedding_args(train_parser)
    train_parser.add_argument("--legacy-state-dict-path", default=None)
    train_parser.add_argument("--save-dir", default=None)
    train_parser.add_argument("--resume", default=None)
    train_parser.add_argument("--init-checkpoint", default=None)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)
    train_parser.add_argument("--dropout", type=float, default=None)
    train_parser.add_argument("--label-smoothing", type=float, default=None)
    train_parser.add_argument("--warmup-epochs", type=int, default=None)
    train_parser.add_argument("--early-stop-patience", type=int, default=None)
    train_parser.add_argument("--num-workers", type=int, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--log-interval", type=int, default=None)
    train_parser.add_argument("--max-train-rows", type=int, default=None)
    train_parser.add_argument("--max-val-rows", type=int, default=None)
    train_parser.add_argument("--classification-loss-weight", type=float, default=None)
    train_parser.add_argument("--ranking-loss-weight", type=float, default=None)
    train_parser.add_argument("--ranking-margin", type=float, default=None)
    train_parser.add_argument("--ranking-max-pairs-per-group", type=int, default=None)
    train_parser.add_argument("--disable-substrate-batching", action="store_true")
    train_parser.add_argument("--best-metric", default=None)
    train_parser.add_argument("--best-metric-source", choices=["val", "external"], default=None)
    train_parser.add_argument("--finetune-mode", choices=["full", "head-only", "mlp-head"], default=None)
    train_parser.add_argument("--train-head-only", action="store_true")
    train_parser.add_argument("--eval-before-training", action="store_true")
    train_parser.add_argument("--load-legacy", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate a checkpoint on labeled CSV")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--csv", required=True)
    eval_parser.add_argument("--config", default=None)
    eval_parser.add_argument("--word2vec-path", default=None)
    add_protein_embedding_args(eval_parser)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--num-workers", type=int, default=None)
    eval_parser.add_argument("--max-rows", type=int, default=None)
    eval_parser.add_argument("--device", default=None)
    eval_parser.add_argument("--output", default=None)

    predict_parser = subparsers.add_parser("predict", help="Predict labels and scores for a CSV")
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--csv", required=True)
    predict_parser.add_argument("--config", default=None)
    predict_parser.add_argument("--word2vec-path", default=None)
    add_protein_embedding_args(predict_parser)
    predict_parser.add_argument("--batch-size", type=int, default=None)
    predict_parser.add_argument("--num-workers", type=int, default=None)
    predict_parser.add_argument("--max-rows", type=int, default=None)
    predict_parser.add_argument("--device", default=None)
    predict_parser.add_argument("--output", default=None)

    screen_parser = subparsers.add_parser("screen", help="Score every protein in a FASTA/PEP file against target SMILES")
    screen_parser.add_argument("--checkpoint", required=True)
    screen_parser.add_argument("--pep", default="predict/input_pep/淫羊藿.pep")
    screen_parser.add_argument("--smiles", default="predict/input_smile/target_smiles.txt")
    screen_parser.add_argument("--config", default=None)
    screen_parser.add_argument("--word2vec-path", default=None)
    add_protein_embedding_args(screen_parser)
    screen_parser.add_argument("--batch-size", type=int, default=None)
    screen_parser.add_argument("--num-workers", type=int, default=None)
    screen_parser.add_argument("--output", default="predict/proteome_predictions.csv")
    screen_parser.add_argument("--top-k", type=int, default=50)
    screen_parser.add_argument("--gt-reference-dir", default="data/gt_reference")
    screen_parser.add_argument("--disable-gt-gate", action="store_true")
    screen_parser.add_argument("--min-aa", type=int, default=None)
    screen_parser.add_argument("--max-aa", type=int, default=None)
    screen_parser.add_argument("--max-proteins", type=int, default=None)
    screen_parser.add_argument("--device", default=None)
    return parser


def load_runtime_config(args) -> tuple:
    config_path = Path(args.config) if args.config else default_config_path(args.preset)
    config = load_config(config_path)
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.train_csv is not None:
        config.paths.train_csv = args.train_csv
    if args.val_csv is not None:
        config.paths.val_csv = args.val_csv
    if getattr(args, "external_val_csv", None) is not None:
        config.paths.external_val_csv = args.external_val_csv
    if args.word2vec_path is not None:
        config.paths.word2vec_path = args.word2vec_path
    if getattr(args, "protein_embedding_mode", None) is not None:
        config.protein_embedding.mode = args.protein_embedding_mode
    if getattr(args, "protein_embedding_manifest", None) is not None:
        config.protein_embedding.manifest_path = args.protein_embedding_manifest
    if getattr(args, "protein_embedding_dir", None) is not None:
        config.protein_embedding.embedding_dir = args.protein_embedding_dir
    if getattr(args, "disable_protein_embedding_cache", False):
        config.protein_embedding.cache_embeddings = False
    if getattr(args, "protein_embedding_cache_size", None) is not None:
        config.protein_embedding.cache_size = args.protein_embedding_cache_size
    if args.legacy_state_dict_path is not None:
        config.paths.legacy_state_dict_path = args.legacy_state_dict_path
    if args.save_dir is not None:
        config.paths.save_dir = args.save_dir
    if args.resume is not None:
        config.runtime.resume = args.resume
    if getattr(args, "init_checkpoint", None) is not None:
        config.runtime.init_checkpoint = args.init_checkpoint
    if args.device is not None:
        config.runtime.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size_per_gpu = args.batch_size
    if getattr(args, "lr", None) is not None:
        config.train.lr = args.lr
    if getattr(args, "weight_decay", None) is not None:
        config.train.weight_decay = args.weight_decay
    if getattr(args, "dropout", None) is not None:
        config.model.dropout = args.dropout
    if getattr(args, "label_smoothing", None) is not None:
        config.train.label_smoothing = args.label_smoothing
    if getattr(args, "warmup_epochs", None) is not None:
        config.train.warmup_epochs = args.warmup_epochs
    if getattr(args, "early_stop_patience", None) is not None:
        config.train.early_stop_patience = args.early_stop_patience
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.seed is not None:
        config.train.seed = args.seed
    if args.log_interval is not None:
        config.runtime.log_interval = args.log_interval
    if getattr(args, "classification_loss_weight", None) is not None:
        config.train.classification_loss_weight = args.classification_loss_weight
    if getattr(args, "ranking_loss_weight", None) is not None:
        config.train.ranking_loss_weight = args.ranking_loss_weight
    if getattr(args, "ranking_margin", None) is not None:
        config.train.ranking_margin = args.ranking_margin
    if getattr(args, "ranking_max_pairs_per_group", None) is not None:
        config.train.ranking_max_pairs_per_group = args.ranking_max_pairs_per_group
    if getattr(args, "disable_substrate_batching", False):
        config.train.substrate_batching = False
    if getattr(args, "best_metric", None) is not None:
        config.train.best_metric = args.best_metric
    if getattr(args, "best_metric_source", None) is not None:
        config.train.best_metric_source = args.best_metric_source
    if getattr(args, "finetune_mode", None) is not None:
        config.train.finetune_mode = args.finetune_mode
    if getattr(args, "train_head_only", False):
        config.train.finetune_mode = "head-only"
    if getattr(args, "eval_before_training", False):
        config.train.eval_before_training = True
    return config, config_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config, _ = load_runtime_config(args)
        train_command(
            config=config,
            run_name=args.run_name,
            max_train_rows=args.max_train_rows,
            max_val_rows=args.max_val_rows,
            load_legacy=args.load_legacy,
        )
        return

    if args.command == "eval":
        eval_command(
            checkpoint_path=args.checkpoint,
            csv_path=args.csv,
            config_path=args.config,
            word2vec_path=args.word2vec_path,
            protein_embedding_mode_override=args.protein_embedding_mode,
            protein_embedding_manifest=args.protein_embedding_manifest,
            protein_embedding_dir=args.protein_embedding_dir,
            protein_embedding_cache_embeddings=False if args.disable_protein_embedding_cache else None,
            protein_embedding_cache_size=args.protein_embedding_cache_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_rows=args.max_rows,
            device=args.device,
            output_path=args.output,
        )
        return

    if args.command == "predict":
        predict_command(
            checkpoint_path=args.checkpoint,
            csv_path=args.csv,
            config_path=args.config,
            word2vec_path=args.word2vec_path,
            protein_embedding_mode_override=args.protein_embedding_mode,
            protein_embedding_manifest=args.protein_embedding_manifest,
            protein_embedding_dir=args.protein_embedding_dir,
            protein_embedding_cache_embeddings=False if args.disable_protein_embedding_cache else None,
            protein_embedding_cache_size=args.protein_embedding_cache_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_rows=args.max_rows,
            device=args.device,
            output_path=args.output,
        )
        return

    if args.command == "screen":
        screen_proteome_command(
            checkpoint_path=args.checkpoint,
            pep_path=args.pep,
            smiles_path=args.smiles,
            config_path=args.config,
            word2vec_path=args.word2vec_path,
            protein_embedding_mode_override=args.protein_embedding_mode,
            protein_embedding_manifest=args.protein_embedding_manifest,
            protein_embedding_dir=args.protein_embedding_dir,
            protein_embedding_cache_embeddings=False if args.disable_protein_embedding_cache else None,
            protein_embedding_cache_size=args.protein_embedding_cache_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_path=args.output,
            top_k=args.top_k,
            gt_gate=not args.disable_gt_gate,
            gt_reference_dir=args.gt_reference_dir,
            min_aa=args.min_aa,
            max_aa=args.max_aa,
            max_proteins=args.max_proteins,
            device=args.device,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")
