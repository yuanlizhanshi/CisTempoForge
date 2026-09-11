from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import CisTempoForgeConfig, DataConfig


REQUIRED_INPUT_KEYS = (
    "dna", "atac", "target", "peak_mask", "gene_peak_index",
    "atac_channel_mean", "atac_channel_std",
)
REQUIRED_STRUCTURE_KEYS = ("features", "masks")


@dataclass(frozen=True)
class Normalization:
    target_mean: float
    target_std: float

    def to_dict(self) -> dict[str, float]:
        return {"target_mean": self.target_mean, "target_std": self.target_std}

    @classmethod
    def from_dict(cls, value: Mapping[str, float]) -> "Normalization":
        return cls(float(value["target_mean"]), max(float(value["target_std"]), 1e-6))


@dataclass
class PreparedData:
    arrays: dict[str, np.ndarray]
    distance: np.ndarray
    metadata: pd.DataFrame
    structure_features: np.ndarray
    structure_masks: np.ndarray
    normalization: Normalization
    data_config: DataConfig

    def indexes(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.metadata["split"].to_numpy() == split)

    def dataset(self, split: str) -> "CisTempoForgeDataset":
        indexes = self.indexes(split)
        if not len(indexes):
            raise ValueError(f"split has no genes: {split}")
        return CisTempoForgeDataset(
            self.arrays, self.distance, self.metadata, indexes, self.normalization,
            self.structure_features, self.structure_masks, self.data_config.day_indices,
        )


def _load_npz(path: str | Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(Path(path).expanduser(), allow_pickle=False) as loaded:
        missing = set(required) - set(loaded.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: loaded[key] for key in required}


def compute_train_normalization(
    target: np.ndarray, metadata: pd.DataFrame, train_split: str,
    day_indices: list[int],
) -> Normalization:
    train_ids = np.flatnonzero(metadata["split"].to_numpy() == train_split)
    if not len(train_ids):
        raise ValueError(f"training split has no genes: {train_split}")
    values = np.asarray(target[np.ix_(train_ids, day_indices)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("training targets contain non-finite values")
    return Normalization(float(values.mean()), max(float(values.std()), 1e-6))


def load_data(config: CisTempoForgeConfig) -> PreparedData:
    config.validate(check_paths=True)
    data = config.data
    arrays = _load_npz(data.model_inputs, REQUIRED_INPUT_KEYS)
    structure = _load_npz(data.structure_inputs, REQUIRED_STRUCTURE_KEYS)
    metadata = pd.read_parquet(Path(data.metadata).expanduser()).reset_index(drop=True)
    distance = np.load(Path(data.distance).expanduser(), allow_pickle=False)
    n_genes = len(metadata)
    if "split" not in metadata or "n_selected_promoter_peaks" not in metadata:
        raise ValueError("metadata requires split and n_selected_promoter_peaks columns")
    if arrays["target"].ndim != 2 or arrays["target"].shape[0] != n_genes:
        raise ValueError("target must have shape [metadata genes, days]")
    if max(data.day_indices) >= arrays["target"].shape[1]:
        raise ValueError("day_indices exceed target time dimension")
    if arrays["atac"].ndim != 3 or max(data.day_indices) >= arrays["atac"].shape[1]:
        raise ValueError("day_indices exceed ATAC time dimension")
    if arrays["peak_mask"].shape != arrays["gene_peak_index"].shape:
        raise ValueError("peak_mask and gene_peak_index shapes differ")
    if arrays["peak_mask"].shape[0] != n_genes or distance.shape != arrays["peak_mask"].shape:
        raise ValueError("gene/peak distance shapes do not align with metadata")
    features = np.asarray(structure["features"], dtype=np.float32)
    masks = np.asarray(structure["masks"], dtype=np.float32)
    if features.shape != (n_genes, 8) or masks.shape != (n_genes, 8):
        raise ValueError("structure features and masks must each be [genes, 8]")
    split_values = set(metadata["split"].astype(str))
    required_splits = {data.train_split, data.validation_split, data.test_split}
    missing_splits = required_splits - split_values
    if missing_splits:
        raise ValueError(f"metadata is missing configured splits: {sorted(missing_splits)}")
    normalization = compute_train_normalization(
        arrays["target"], metadata, data.train_split, data.day_indices
    )
    return PreparedData(
        arrays, distance, metadata, features, masks, normalization, data,
    )


class CisTempoForgeDataset(Dataset):
    def __init__(self, arrays: Mapping[str, np.ndarray], distance: np.ndarray,
                 metadata: pd.DataFrame, indexes: np.ndarray,
                 normalization: Normalization | Mapping[str, float],
                 structure_features: np.ndarray, structure_masks: np.ndarray,
                 day_indices: list[int]):
        self.arrays = arrays
        self.distance = np.asarray(distance)
        self.metadata = metadata
        self.indexes = np.asarray(indexes, dtype=np.int64)
        self.normalization = (
            normalization if isinstance(normalization, Normalization)
            else Normalization.from_dict(normalization)
        )
        self.structure_features = np.asarray(structure_features, dtype=np.float32)
        self.structure_masks = np.asarray(structure_masks, dtype=np.float32)
        self.day_indices = np.asarray(day_indices, dtype=np.int64)
        n = len(metadata)
        if self.structure_features.shape != (n, 8) or self.structure_masks.shape != (n, 8):
            raise ValueError("structure arrays must be [genes, 8]")
        if np.any(self.indexes < 0) or np.any(self.indexes >= n):
            raise IndexError("dataset gene indexes are out of range")

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        gene = int(self.indexes[item])
        rows = self.arrays["gene_peak_index"][gene]
        mask = self.arrays["peak_mask"][gene].astype(bool, copy=True)
        dna = self.arrays["dna"][rows].astype(np.float32)
        pooled = self.arrays["atac"][rows][:, self.day_indices].astype(np.float32)
        pooled = pooled.transpose(1, 0, 2)
        mean = float(self.arrays["atac_channel_mean"][0])
        std = max(float(self.arrays["atac_channel_std"][0]), 1e-6)
        atac = ((pooled - mean) / std)[:, :, None]
        dna[~mask] = 0
        atac[:, ~mask] = 0
        gene_distance = self.distance[gene].astype(np.float32, copy=True)
        gene_distance[~mask] = 0
        promoter_count = int(self.metadata.iloc[gene]["n_selected_promoter_peaks"])
        promoter_mask = np.zeros_like(mask)
        promoter_mask[:promoter_count] = mask[:promoter_count]
        target = self.arrays["target"][gene, self.day_indices].astype(np.float32)
        target = (target - self.normalization.target_mean) / self.normalization.target_std
        level = np.float32(target.mean())
        return {
            "dna": torch.from_numpy(dna), "atac": torch.from_numpy(atac),
            "peak_mask": torch.from_numpy(mask),
            "promoter_mask": torch.from_numpy(promoter_mask),
            "mean_absolute_tss_distance": torch.from_numpy(gene_distance),
            "structure_features": torch.from_numpy(self.structure_features[gene]),
            "structure_mask": torch.from_numpy(self.structure_masks[gene]),
            "target": torch.from_numpy(target), "level_target": torch.tensor(level),
            "trend_target": torch.from_numpy(target - level),
            "gene_index": torch.tensor(gene, dtype=torch.long),
        }


def make_loader(dataset: Dataset, *, shuffle: bool, batch_size: int,
                num_workers: int = 0, pin_memory: bool | None = None) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )


def denormalize(values: np.ndarray, normalization: Normalization) -> np.ndarray:
    return values * normalization.target_std + normalization.target_mean
