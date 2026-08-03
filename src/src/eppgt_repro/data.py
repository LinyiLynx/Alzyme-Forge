from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from .config import ProteinEmbeddingConfig, Word2VecConfig
from .word2vec_utils import embed_sequence


NEW_STYLE_COLUMNS = {"Protein_Sequence", "Substrate_SMILES"}
LEGACY_COLUMNS = {"com", "seq"}
NORMALIZED_COLUMNS = {"sequence", "smiles"}


def normalize_sequence_text(sequence: str) -> str:
    return "".join(str(sequence or "").split()).replace("*", "").upper()


def sequence_sha256(sequence: str) -> str:
    normalized = normalize_sequence_text(sequence)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_pair_dataframe(df: pd.DataFrame, require_label: bool = True) -> pd.DataFrame:
    if NEW_STYLE_COLUMNS.issubset(df.columns):
        normalized = pd.DataFrame(
            {
                "protein_id": df["Protein_ID"] if "Protein_ID" in df.columns else [f"protein_{idx}" for idx in range(len(df))],
                "sequence": df["Protein_Sequence"],
                "smiles": df["Substrate_SMILES"],
            }
        )
        label_column = "Label" if "Label" in df.columns else None
    elif LEGACY_COLUMNS.issubset(df.columns):
        normalized = pd.DataFrame(
            {
                "protein_id": df["protein_id"] if "protein_id" in df.columns else [f"protein_{idx}" for idx in range(len(df))],
                "sequence": df["seq"],
                "smiles": df["com"],
            }
        )
        label_column = "label" if "label" in df.columns else None
    elif NORMALIZED_COLUMNS.issubset(df.columns):
        normalized = pd.DataFrame(
            {
                "protein_id": df["protein_id"] if "protein_id" in df.columns else [f"protein_{idx}" for idx in range(len(df))],
                "sequence": df["sequence"],
                "smiles": df["smiles"],
            }
        )
        label_column = "label" if "label" in df.columns else None
    else:
        raise ValueError(f"Unsupported columns: {list(df.columns)}")

    if label_column is not None:
        normalized["label"] = df[label_column].astype(int)
    elif require_label:
        raise ValueError("Labels are required for this operation")

    subset = ["protein_id", "sequence", "smiles"] + (["label"] if "label" in normalized.columns else [])
    normalized = normalized[subset].dropna().reset_index(drop=True)
    normalized["sequence"] = normalized["sequence"].map(normalize_sequence_text)
    return normalized


def load_pairs_csv(path: str | Path, require_label: bool = True, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    normalized = normalize_pair_dataframe(df, require_label=require_label)
    if max_rows is not None:
        normalized = normalized.head(max_rows).reset_index(drop=True)
    return normalized


class AtomFeatures:
    atom_dim = 46

    def atom_features(self, atom, explicit_h: bool = False, use_chirality: bool = True):
        symbol = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other"]
        degree = [0, 1, 2, 3, 4, 5, 6]
        hybridization_type = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            "other",
        ]
        result = (
            self.one_of_k_encoding_unk(atom.GetSymbol(), symbol)
            + self.one_of_k_encoding(atom.GetDegree(), degree)
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + self.one_of_k_encoding_unk(atom.GetHybridization(), hybridization_type)
            + self.one_of_k_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
            + self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
            + [atom.GetIsAromatic()]
        )
        if not explicit_h:
            result = result + self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                result = result + self.one_of_k_encoding_unk(atom.GetProp("_CIPCode"), ["R", "S"]) + [
                    atom.HasProp("_ChiralityPossible")
                ]
            except Exception:
                result = result + [False, False, atom.HasProp("_ChiralityPossible")]
        return result

    def one_of_k_encoding(self, value, allowable_set):
        if value not in allowable_set:
            raise ValueError(f"input {value} not in allowable set {allowable_set}")
        return [value == item for item in allowable_set]

    def one_of_k_encoding_unk(self, value, allowable_set):
        if value not in allowable_set:
            value = allowable_set[-1]
        return [value == item for item in allowable_set]

    def adjacent_matrix(self, mol):
        return np.array(Chem.GetAdjacencyMatrix(mol))

    def mol_features(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        atom_feature = np.zeros((mol.GetNumAtoms(), self.atom_dim), dtype=np.float32)
        for atom in mol.GetAtoms():
            atom_feature[atom.GetIdx(), :] = self.atom_features(atom)
        adj_matrix = self.adjacent_matrix(mol).astype(np.float32)
        return atom_feature, adj_matrix


class EsmcEmbeddingStore:
    def __init__(self, config: ProteinEmbeddingConfig) -> None:
        self.config = config
        self.manifest_path = Path(config.manifest_path).resolve() if config.manifest_path else None
        self.embedding_dir = Path(config.embedding_dir).resolve() if config.embedding_dir else None
        self._path_by_hash: dict[str, Path] = {}
        self._cache: OrderedDict[str, torch.FloatTensor] = OrderedDict()
        if self.manifest_path is not None:
            self._load_manifest(self.manifest_path)
        if self.manifest_path is None and self.embedding_dir is None:
            raise ValueError("ESMC embedding mode requires protein_embedding.manifest_path or embedding_dir")

    def _load_manifest(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"ESMC manifest not found: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        if self.config.hash_column not in manifest.columns:
            raise ValueError(
                f"ESMC manifest missing hash column {self.config.hash_column!r}: {manifest_path}"
            )
        if self.config.path_column not in manifest.columns and self.embedding_dir is None:
            raise ValueError(
                f"ESMC manifest missing path column {self.config.path_column!r} and no embedding_dir was provided"
            )
        for row in manifest.itertuples(index=False):
            row_dict = row._asdict()
            seq_hash = str(row_dict[self.config.hash_column]).strip().lower()
            if not seq_hash:
                continue
            path_value = str(row_dict.get(self.config.path_column, "") or "").strip()
            if path_value:
                path = Path(path_value)
                if not path.is_absolute():
                    path = manifest_path.parent / path
            elif self.embedding_dir is not None:
                path = self.embedding_dir / f"{seq_hash}.pt"
            else:
                continue
            self._path_by_hash[seq_hash] = path

    def path_for_hash(self, seq_hash: str) -> Path:
        seq_hash = seq_hash.lower()
        if seq_hash in self._path_by_hash:
            return self._path_by_hash[seq_hash]
        if self.embedding_dir is not None:
            return self.embedding_dir / f"{seq_hash}.pt"
        if self.manifest_path is not None:
            return self.manifest_path.parent / "embeddings" / f"{seq_hash}.pt"
        raise KeyError(seq_hash)

    def _load_tensor(self, path: Path) -> torch.FloatTensor:
        if not path.exists():
            raise FileNotFoundError(f"ESMC embedding file not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
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
                raise ValueError(f"No tensor-like ESMC embedding found in {path}")
        else:
            raise ValueError(f"Unsupported ESMC embedding payload in {path}: {type(payload)}")
        if tensor.dim() != 2:
            raise ValueError(f"ESMC embedding must be 2D [length, dim], got {tuple(tensor.shape)} from {path}")
        return tensor.detach().float().cpu()

    def get(self, sequence: str) -> torch.FloatTensor:
        normalized = normalize_sequence_text(sequence)
        seq_hash = sequence_sha256(normalized)
        if self.config.cache_embeddings and seq_hash in self._cache:
            self._cache.move_to_end(seq_hash)
            return self._cache[seq_hash]
        tensor = self._load_tensor(self.path_for_hash(seq_hash))
        if self.config.validate_lengths and tensor.shape[0] != len(normalized):
            raise ValueError(
                "ESMC embedding length mismatch for hash "
                f"{seq_hash}: embedding length={tensor.shape[0]}, sequence length={len(normalized)}"
            )
        if self.config.cache_embeddings and self.config.cache_size != 0:
            self._cache[seq_hash] = tensor
            self._cache.move_to_end(seq_hash)
            if self.config.cache_size > 0:
                while len(self._cache) > self.config.cache_size:
                    self._cache.popitem(last=False)
        return tensor


class PairDataset(Dataset, AtomFeatures):
    def __init__(
        self,
        data: pd.DataFrame,
        word2vec_model=None,
        word2vec_config: Word2VecConfig | None = None,
        include_labels: bool = True,
        include_metadata: bool = False,
        protein_embedding_store: EsmcEmbeddingStore | None = None,
    ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.word2vec_model = word2vec_model
        self.word2vec_config = word2vec_config
        self.protein_embedding_store = protein_embedding_store
        self.include_labels = include_labels
        self.include_metadata = include_metadata
        self._sequence_cache: dict[str, torch.FloatTensor] = {}
        self._smiles_cache: dict[str, tuple[torch.FloatTensor, torch.FloatTensor]] = {}
        if self.protein_embedding_store is None and (self.word2vec_model is None or self.word2vec_config is None):
            raise ValueError("PairDataset requires either Word2Vec inputs or an ESMC embedding store")

    def sequence_embedding(self, sequence: str) -> torch.FloatTensor:
        if self.protein_embedding_store is not None:
            return self.protein_embedding_store.get(sequence)
        if sequence not in self._sequence_cache:
            embedding = embed_sequence(
                self.word2vec_model,
                sequence,
                self.word2vec_config.k,  # type: ignore[union-attr]
                self.word2vec_config.vector_size,  # type: ignore[union-attr]
            )
            self._sequence_cache[sequence] = torch.tensor(embedding, dtype=torch.float32)
        return self._sequence_cache[sequence]

    def smiles_embedding(self, smiles: str) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if smiles not in self._smiles_cache:
            atom_feature, adj = self.mol_features(smiles)
            self._smiles_cache[smiles] = (
                torch.tensor(atom_feature, dtype=torch.float32),
                torch.tensor(adj, dtype=torch.float32),
            )
        return self._smiles_cache[smiles]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        compound, adj = self.smiles_embedding(row["smiles"])
        protein = self.sequence_embedding(row["sequence"])
        metadata = {
            "protein_id": str(row["protein_id"]),
            "smiles": str(row["smiles"]),
            "sequence_hash": sequence_sha256(str(row["sequence"])),
        }
        if not self.include_labels:
            if self.include_metadata:
                return compound, adj, protein, metadata
            return compound, adj, protein
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.include_metadata:
            return compound, adj, protein, label, metadata
        return compound, adj, protein, label


def collate_pairs(batch):
    metadata = None
    if len(batch[0]) == 3:
        compounds, adjs, proteins = zip(*batch)
        labels = None
    elif len(batch[0]) == 4 and isinstance(batch[0][-1], dict):
        compounds, adjs, proteins, metadata = zip(*batch)
        labels = None
    elif len(batch[0]) == 5:
        compounds, adjs, proteins, labels, metadata = zip(*batch)
    else:
        compounds, adjs, proteins, labels = zip(*batch)

    device = torch.device("cpu")
    atom_dim = compounds[0].shape[1]
    protein_dim = proteins[0].shape[1]
    batch_size = len(compounds)

    atom_lengths = [compound.shape[0] for compound in compounds]
    protein_lengths = [protein.shape[0] for protein in proteins]
    max_atoms = max(atom_lengths)
    max_proteins = max(protein_lengths)

    padded_compounds = torch.zeros((batch_size, max_atoms, atom_dim), dtype=torch.float32, device=device)
    padded_adjs = torch.zeros((batch_size, max_atoms, max_atoms), dtype=torch.float32, device=device)
    padded_proteins = torch.zeros((batch_size, max_proteins, protein_dim), dtype=torch.float32, device=device)

    for index, compound in enumerate(compounds):
        atom_count = compound.shape[0]
        padded_compounds[index, :atom_count, :] = compound
        adj = adjs[index]
        padded_adjs[index, :atom_count, :atom_count] = adj + torch.eye(atom_count, dtype=torch.float32)

    for index, protein in enumerate(proteins):
        protein_count = protein.shape[0]
        padded_proteins[index, :protein_count, :] = protein

    atom_lengths_tensor = torch.tensor(atom_lengths, dtype=torch.int64)
    protein_lengths_tensor = torch.tensor(protein_lengths, dtype=torch.int64)
    if labels is None:
        if metadata is not None:
            return padded_compounds, padded_adjs, padded_proteins, atom_lengths_tensor, protein_lengths_tensor, list(metadata)
        return padded_compounds, padded_adjs, padded_proteins, atom_lengths_tensor, protein_lengths_tensor
    labels_tensor = torch.stack(labels)
    if metadata is not None:
        return (
            padded_compounds,
            padded_adjs,
            padded_proteins,
            labels_tensor,
            atom_lengths_tensor,
            protein_lengths_tensor,
            list(metadata),
        )
    return padded_compounds, padded_adjs, padded_proteins, labels_tensor, atom_lengths_tensor, protein_lengths_tensor
