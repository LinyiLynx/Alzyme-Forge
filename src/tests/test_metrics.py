import numpy as np

from eppgt_repro.metrics import compute_grouped_ranking_metrics


def test_grouped_ranking_metrics_include_external_screening_columns():
    labels = np.array([1, 0, 0, 1])
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    groups = np.array(["substrate_a", "substrate_a", "substrate_b", "substrate_b"])

    metrics = compute_grouped_ranking_metrics(labels, scores, groups)

    assert metrics["ranking_group_count"] == 2.0
    assert metrics["ranking_valid_group_count"] == 2.0
    assert metrics["ranking_mean_group_size"] == 2.0
    assert metrics["ranking_median_group_size"] == 2.0
    assert metrics["ranking_mean_positive_count"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["top50_hit"] == 1.0
    assert metrics["top100_hit"] == 1.0
