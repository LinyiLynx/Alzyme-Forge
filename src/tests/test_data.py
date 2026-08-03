from pathlib import Path

import pandas as pd
import torch

from eppgt_repro.config import Word2VecConfig
from eppgt_repro.data import PairDataset, collate_pairs, normalize_pair_dataframe
from eppgt_repro.word2vec_utils import train_word2vec_model


def make_word2vec(tmp_path: Path):
    config = Word2VecConfig(k=3, vector_size=100, window=5, min_count=1, epochs=2, workers=1)
    path = tmp_path / "toy.model"
    train_word2vec_model(["MKTAAA", "MKTBBB", "MKTVVV"], path, config)
    return config, path


def test_normalize_new_schema():
    df = pd.DataFrame(
        {
            "Protein_ID": ["P1"],
            "Protein_Sequence": ["MKTAAA"],
            "Substrate_SMILES": ["CCO"],
            "Label": [1],
        }
    )
    normalized = normalize_pair_dataframe(df)
    assert normalized.columns.tolist() == ["protein_id", "sequence", "smiles", "label"]
    assert normalized.iloc[0].to_dict() == {"protein_id": "P1", "sequence": "MKTAAA", "smiles": "CCO", "label": 1}


def test_normalize_legacy_schema():
    df = pd.DataFrame({"com": ["CCO"], "seq": ["MKTAAA"], "label": [0]})
    normalized = normalize_pair_dataframe(df)
    assert normalized.iloc[0]["sequence"] == "MKTAAA"
    assert normalized.iloc[0]["smiles"] == "CCO"
    assert normalized.iloc[0]["label"] == 0


def test_pair_dataset_and_collate_shapes(tmp_path: Path):
    config, path = make_word2vec(tmp_path)
    model = __import__("gensim").models.Word2Vec.load(str(path))
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "sequence": "MKTAAA", "smiles": "CCO", "label": 1},
            {"protein_id": "P2", "sequence": "MKTVVV", "smiles": "CCN", "label": 0},
        ]
    )
    dataset = PairDataset(df, model, config, include_labels=True)
    batch = collate_pairs([dataset[0], dataset[1]])
    compounds, adjs, proteins, labels, atom_lengths, protein_lengths = batch
    assert compounds.shape[0] == 2
    assert compounds.shape[2] == 46
    assert proteins.shape[2] == 100
    assert labels.tolist() == [1, 0]
    assert torch.equal(torch.diagonal(adjs[0], dim1=0, dim2=1)[: atom_lengths[0]], torch.ones(atom_lengths[0]))
