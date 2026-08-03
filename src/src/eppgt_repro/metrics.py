from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "auroc": math.nan,
        "auprc": math.nan,
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(labels, scores)
    except ValueError:
        pass
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics["auprc"] = auc(recall, precision)
    except ValueError:
        pass
    return metrics


def compute_grouped_ranking_metrics(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    groups: list[str] | np.ndarray,
    top_ks: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    groups = np.asarray(groups)
    metrics: dict[str, float] = {
        "ranking_group_count": 0.0,
        "ranking_valid_group_count": 0.0,
        "ranking_mean_group_size": math.nan,
        "ranking_median_group_size": math.nan,
        "ranking_mean_positive_count": math.nan,
        "mrr": math.nan,
        "mean_positive_rank": math.nan,
        "mean_group_auroc": math.nan,
        "ndcg": math.nan,
    }
    for k in top_ks:
        metrics[f"top{k}_hit"] = math.nan

    reciprocal_ranks = []
    first_positive_ranks = []
    ndcgs = []
    group_aurocs = []
    group_sizes = []
    positive_counts = []
    top_hits = {k: [] for k in top_ks}
    unique_groups = np.unique(groups)
    metrics["ranking_group_count"] = float(len(unique_groups))

    for group in unique_groups:
        mask = groups == group
        group_labels = labels[mask]
        group_scores = scores[mask]
        positive_count = int(group_labels.sum())
        group_sizes.append(int(mask.sum()))
        positive_counts.append(positive_count)
        if positive_count == 0:
            continue
        order = np.argsort(-group_scores)
        ranked_labels = group_labels[order]
        positive_positions = np.flatnonzero(ranked_labels == 1) + 1
        if len(positive_positions) == 0:
            continue
        metrics["ranking_valid_group_count"] += 1.0
        first_rank = int(positive_positions[0])
        reciprocal_ranks.append(1.0 / first_rank)
        first_positive_ranks.append(float(first_rank))
        for k in top_ks:
            top_hits[k].append(float(np.any(ranked_labels[:k] == 1)))

        gains = ranked_labels.astype(float)
        discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(group_labels)[::-1].astype(float)
        ideal_dcg = float(np.sum(ideal * discounts))
        if ideal_dcg > 0:
            ndcgs.append(dcg / ideal_dcg)
        if len(np.unique(group_labels)) == 2:
            try:
                group_aurocs.append(roc_auc_score(group_labels, group_scores))
            except ValueError:
                pass

    if reciprocal_ranks:
        metrics["mrr"] = float(np.mean(reciprocal_ranks))
        metrics["mean_positive_rank"] = float(np.mean(first_positive_ranks))
        metrics["ndcg"] = float(np.mean(ndcgs)) if ndcgs else math.nan
        for k in top_ks:
            metrics[f"top{k}_hit"] = float(np.mean(top_hits[k])) if top_hits[k] else math.nan
    if group_sizes:
        metrics["ranking_mean_group_size"] = float(np.mean(group_sizes))
        metrics["ranking_median_group_size"] = float(np.median(group_sizes))
        metrics["ranking_mean_positive_count"] = float(np.mean(positive_counts))
    if group_aurocs:
        metrics["mean_group_auroc"] = float(np.mean(group_aurocs))
    return metrics
