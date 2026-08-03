"""Shared clustering utilities for compound/protein grouping and negative sampling."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
EMBED_DIR = ROOT / "data" / "esmc_embeddings" / "embeddings"


def normalize_sequence_text(sequence: str) -> str:
    return "".join(str(sequence or "").split()).replace("*", "").upper()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(normalize_sequence_text(sequence).encode("utf-8")).hexdigest()


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def choose_kmeans_k(
    matrix: np.ndarray,
    k_min: int = 6,
    k_max: int = 14,
    random_state: int = 42,
) -> int:
    best_k = k_min
    best_score = -1.0
    upper = min(k_max, len(matrix) - 1)
    for k in range(k_min, upper + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            matrix,
            labels,
            sample_size=min(2000, len(matrix)),
            random_state=random_state,
        )
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_proteins(
    matrix: np.ndarray,
    meta: pd.DataFrame,
    k_min: int = 6,
    k_max: int = 14,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[int, np.ndarray]]:
    """L2-normalize ESMC mean-pool vectors, then KMeans (cosine-compatible)."""
    normalized = l2_normalize(matrix.astype(np.float32))
    k = choose_kmeans_k(normalized, k_min=k_min, k_max=k_max, random_state=random_state)
    print(f"[protein clustering] selected k={k}", file=sys.stderr)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(normalized)

    out = meta.copy()
    out["cluster_id"] = labels

    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(set(labels)):
        members = normalized[labels == cid]
        centroids[int(cid)] = l2_normalize(members.mean(axis=0, keepdims=True))[0]
    return out, normalized, centroids


def pca_project(
    matrix: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA projection aligned with KMeans feature space."""
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(matrix)
    return coords, pca.explained_variance_ratio_


def pca_project_2d(
    matrix: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    return pca_project(matrix, n_components=2, random_state=random_state)


def cluster_compounds(
    smiles_list: list[str],
    k_min: int = 8,
    k_max: int = 12,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Cluster compounds with L2-normalized Morgan fingerprints + KMeans."""
    fps = []
    valid_smiles: list[str] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        valid_smiles.append(smi)

    if not fps:
        raise ValueError("No valid SMILES for compound clustering")

    fp_array = np.asarray([np.array(fp) for fp in fps], dtype=np.float32)
    normalized = l2_normalize(fp_array)
    upper = min(k_max, len(normalized) - 1)
    lower = min(k_min, upper)
    k = choose_kmeans_k(normalized, k_min=lower, k_max=upper, random_state=random_state)
    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(normalized)

    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(set(labels)):
        members = normalized[labels == cid]
        centroids[int(cid)] = l2_normalize(members.mean(axis=0, keepdims=True))[0]

    out = pd.DataFrame({"Substrate_SMILES": valid_smiles, "cluster_id": labels})
    print(
        f"[compound clustering] KMeans k={k}, "
        f"{len(valid_smiles)} compounds, {len(set(labels))} clusters",
        file=sys.stderr,
    )
    return out, centroids


def load_protein_embeddings_from_cache(
    cache_path: Path,
    proteins: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        matrix = cached["matrix"]
        meta = pd.DataFrame(cached["meta"].tolist())
        return matrix, meta

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    missing = 0

    for row in proteins.itertuples(index=False):
        seq_hash = sequence_sha256(row.sequence)
        embed_path = EMBED_DIR / f"{seq_hash}.pt"
        if not embed_path.exists():
            missing += 1
            continue
        try:
            payload = torch.load(embed_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(embed_path, map_location="cpu")

        if isinstance(payload, torch.Tensor):
            tensor = payload
        elif isinstance(payload, dict):
            tensor = None
            for key in ("embedding", "token_embedding", "token_embeddings", "representations"):
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
            if tensor is None:
                missing += 1
                continue
        else:
            missing += 1
            continue

        mean_vec = tensor.detach().float().cpu().mean(dim=0).numpy()
        vectors.append(mean_vec)
        rows.append(
            {
                "protein_id": row.protein_id,
                "sequence_hash": seq_hash,
                "sequence_length": len(normalize_sequence_text(row.sequence)),
            }
        )

    if missing:
        print(f"[protein embeddings] skipped {missing} proteins without embeddings", file=sys.stderr)

    matrix = np.vstack(vectors)
    meta = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, matrix=matrix, meta=meta.to_dict(orient="records"))
    return matrix, meta


def cluster_distance_matrix(centroids: dict[int, np.ndarray], use_cosine: bool = True) -> tuple[list[int], np.ndarray]:
    cluster_ids = sorted(centroids)
    vectors = np.stack([centroids[cid] for cid in cluster_ids], axis=0)
    if use_cosine:
        vectors = l2_normalize(vectors)
        sim = vectors @ vectors.T
        dist = 1.0 - sim
    else:
        diff = vectors[:, None, :] - vectors[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 0.0)
    return cluster_ids, dist
