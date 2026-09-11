import numpy as np
import pandas as pd

from cistempoforge import config_from_dict, load_data


def make_config(data):
    return config_from_dict({
        "data": {
            "model_inputs": data["model_inputs"], "metadata": data["metadata"],
            "distance": data["distance"], "structure_inputs": data["structure_inputs"],
            "day_indices": [0, 2, 4, 6],
        },
        "model": {"n_days": 4, "hidden_dim": 8, "motif_filters": 2},
    })


def test_data_slicing_masks_and_train_only_normalization(synthetic_data):
    config = make_config(synthetic_data)
    prepared = load_data(config)
    expected = synthetic_data["target"][:4][:, [0, 2, 4, 6]]
    assert np.isclose(prepared.normalization.target_mean, expected.mean())
    assert np.isclose(prepared.normalization.target_std, expected.std())
    item = prepared.dataset("test")[0]
    assert item["atac"].shape == (4, 2, 1, 40)
    assert item["target"].shape == (4,)
    assert not item["peak_mask"][-1]
    assert item["dna"][-1].count_nonzero() == 0


def test_split_indexes_do_not_overlap(synthetic_data):
    prepared = load_data(make_config(synthetic_data))
    train = set(prepared.indexes("train"))
    validation = set(prepared.indexes("validation"))
    test = set(prepared.indexes("test"))
    assert not train & validation
    assert not train & test
    assert not validation & test


def test_load_data_without_test_split(synthetic_data, tmp_path):
    metadata = pd.read_parquet(synthetic_data["metadata"])
    metadata.loc[metadata["split"] == "test", "split"] = "train"
    metadata_path = tmp_path / "cv_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)
    values = {
        "model_inputs": synthetic_data["model_inputs"],
        "metadata": str(metadata_path),
        "distance": synthetic_data["distance"],
        "structure_inputs": synthetic_data["structure_inputs"],
        "day_indices": [0, 2, 4, 6],
        "test_split": None,
    }
    config = config_from_dict({
        "data": values,
        "model": {"n_days": 4, "hidden_dim": 8, "motif_filters": 2},
    })
    prepared = load_data(config)
    assert len(prepared.dataset("train")) == 5
    assert len(prepared.dataset("validation")) == 2
