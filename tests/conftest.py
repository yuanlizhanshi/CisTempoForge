from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data(tmp_path):
    rng = np.random.default_rng(7)
    n_genes, n_days, max_peaks, length = 7, 7, 2, 40
    n_peaks = n_genes * max_peaks
    dna = rng.normal(size=(n_peaks, 4, length)).astype(np.float32)
    atac = rng.normal(size=(n_peaks, n_days, length)).astype(np.float32)
    target = np.stack([
        np.linspace(i * 0.1, i * 0.1 + (i + 1) * 0.2, n_days)
        for i in range(n_genes)
    ]).astype(np.float32)
    peak_index = np.arange(n_peaks).reshape(n_genes, max_peaks)
    peak_mask = np.ones((n_genes, max_peaks), dtype=bool)
    peak_mask[-1, -1] = False
    model_inputs = tmp_path / "model_inputs.npz"
    np.savez_compressed(
        model_inputs, dna=dna, atac=atac, target=target,
        peak_mask=peak_mask, gene_peak_index=peak_index,
        atac_channel_mean=np.array([atac.mean()], np.float32),
        atac_channel_std=np.array([atac.std()], np.float32),
    )
    metadata = pd.DataFrame({
        "gene": [f"g{i}" for i in range(n_genes)],
        "split": ["train"] * 4 + ["validation"] * 2 + ["test"],
        "n_selected_promoter_peaks": [1] * n_genes,
    })
    metadata_path = tmp_path / "metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    distance_path = tmp_path / "distance.npy"
    np.save(distance_path, rng.random((n_genes, max_peaks)).astype(np.float32))
    structure_path = tmp_path / "structure.npz"
    np.savez_compressed(
        structure_path,
        features=rng.normal(size=(n_genes, 8)).astype(np.float32),
        masks=np.ones((n_genes, 8), np.float32),
    )
    return {
        "model_inputs": str(model_inputs), "metadata": str(metadata_path),
        "distance": str(distance_path), "structure_inputs": str(structure_path),
        "target": target,
    }
