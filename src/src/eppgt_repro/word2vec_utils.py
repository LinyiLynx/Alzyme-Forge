from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from gensim.models import Word2Vec

from .config import Word2VecConfig


def seq_to_kmers(sequence: str, k: int = 3) -> list[str]:
    if len(sequence) < k:
        return [sequence]
    return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]


def load_word2vec_model(path: str | Path) -> Word2Vec:
    return Word2Vec.load(str(path))


def train_word2vec_model(
    sequences: Iterable[str],
    output_path: str | Path,
    config: Word2VecConfig,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = [seq_to_kmers(sequence, config.k) for sequence in sequences]
    model = Word2Vec(
        sentences=corpus,
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=0,
    )
    model.train(corpus, total_examples=len(corpus), epochs=config.epochs)
    model.save(str(output_path))
    return output_path


def ensure_word2vec_artifact(
    sequences: Iterable[str],
    config: Word2VecConfig,
    artifact_path: str | Path,
    existing_path: str | None = None,
) -> Path:
    artifact_path = Path(artifact_path)
    if existing_path:
        model = load_word2vec_model(existing_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(artifact_path))
        return artifact_path
    unique_sequences = list(dict.fromkeys(sequences))
    return train_word2vec_model(unique_sequences, artifact_path, config)


def embed_sequence(model: Word2Vec, sequence: str, k: int, vector_size: int) -> np.ndarray:
    kmers = seq_to_kmers(sequence, k)
    embedding = np.zeros((len(kmers), vector_size), dtype=np.float32)
    for index, token in enumerate(kmers):
        if token in model.wv:
            embedding[index] = model.wv[token]
    return embedding
