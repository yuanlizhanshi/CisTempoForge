"""Training with no held-out split, so every gene contributes to training.

`validation_split=None` exists so a final model can be fit on the whole dataset.
The trade is that nothing can be early-stopped or selected on, so `fit` runs the
full budget and retains the last epoch; these tests pin that behaviour and the
config rules that make it legal.
"""

import json

import numpy as np
import pandas as pd
import pytest

from cistempoforge import config_from_dict, fit, load_checkpoint
from cistempoforge.config import CisTempoForgeConfig


def all_gene_config(data, epochs=2, **training_overrides):
    """A config whose metadata has every gene under one label."""
    metadata = pd.read_parquet(data["metadata"])
    metadata["split"] = "train"
    metadata_path = data["metadata"].replace("metadata.parquet", "metadata_all.parquet")
    metadata.to_parquet(metadata_path, index=False)

    training = {
        "epochs": epochs, "patience": 2, "batch_size": 2,
        "gradient_accumulation": 2, "num_workers": 0,
        "reverse_complement_probability": 0.5, "device": "cpu",
        "mixed_precision": "none", "show_progress": False,
    }
    training.update(training_overrides)
    return config_from_dict({
        "data": {
            "model_inputs": data["model_inputs"], "metadata": metadata_path,
            "distance": data["distance"], "structure_inputs": data["structure_inputs"],
            "day_indices": [0, 1, 2, 3, 4, 5],
            "validation_split": None, "test_split": None,
        },
        "model": {
            "n_days": 6, "variant": "independent_structure", "fusion": "concat",
            "hidden_dim": 8, "motif_filters": 2, "motif_width": 19, "dropout": 0,
        },
        "training": training,
    })


def test_validation_split_may_be_none():
    config = CisTempoForgeConfig(data=config_from_dict({
        "data": {
            "model_inputs": "a", "metadata": "b", "distance": "c",
            "structure_inputs": "d", "validation_split": None, "test_split": None,
        },
    }).data)
    config.validate()  # must not raise


def test_duplicate_split_names_are_still_rejected():
    """Relaxing validation must not relax the distinctness rule."""
    with pytest.raises(ValueError, match="distinct"):
        config_from_dict({
            "data": {
                "model_inputs": "a", "metadata": "b", "distance": "c",
                "structure_inputs": "d",
                "train_split": "train", "validation_split": "train",
                "test_split": None,
            },
        })


def test_empty_split_name_is_still_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        config_from_dict({
            "data": {
                "model_inputs": "a", "metadata": "b", "distance": "c",
                "structure_inputs": "d", "validation_split": "",
            },
        })


def test_fit_without_validation_trains_on_every_gene(synthetic_data, tmp_path):
    config = all_gene_config(synthetic_data, epochs=3)
    output = tmp_path / "all_genes"
    result = fit(config, output)

    # Every gene is in the training set; none is held out.
    assert result.validation_metrics is None
    assert len(result.history) == 3
    assert result.best_epoch == 3, "without a validation split the final epoch is kept"

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["validation_metrics"] is None
    assert manifest["validation_metrics_normalized"] is None
    assert manifest["validation_genes"] == 0
    assert "no validation split" in manifest["selection"]["rule"]
    assert manifest["best_epoch"] == 3

    # All 7 synthetic genes were trained on, not the 4 of the "train" label.
    assert manifest["training_genes"] == 7


def test_history_records_train_metrics_only(synthetic_data, tmp_path):
    config = all_gene_config(synthetic_data, epochs=2)
    result = fit(config, tmp_path / "history")

    columns = list(result.history.columns)
    assert "train_total" in columns
    assert "learning_rate" in columns
    assert not any(name.startswith("validation") for name in columns), columns
    assert result.history["train_total"].notna().all()


def test_every_epoch_checkpoint_is_kept(synthetic_data, tmp_path):
    """Selecting an epoch after the fact depends on these surviving."""
    config = all_gene_config(synthetic_data, epochs=3)
    output = tmp_path / "checkpoints"
    fit(config, output)

    for epoch in (1, 2, 3):
        path = output / ".epoch_checkpoints" / f"epoch_{epoch:03d}.pt"
        assert path.exists(), f"epoch {epoch} checkpoint missing"
        assert load_checkpoint(path, map_location="cpu")["epoch"] == epoch


def test_no_validation_run_is_reproducible(synthetic_data, tmp_path):
    """Two identical all-gene runs must agree bitwise, as with validation."""
    import torch

    config = all_gene_config(synthetic_data, epochs=2)
    first = fit(config, tmp_path / "run_a")
    second = fit(config, tmp_path / "run_b")

    state_a = {k: v.clone() for k, v in first.model.state_dict().items()}
    state_b = second.model.state_dict()
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    assert np.array_equal(
        first.history["train_total"].to_numpy(),
        second.history["train_total"].to_numpy(),
    )
