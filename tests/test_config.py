import json

import pytest

from cistempoforge import config_from_dict, load_config


def minimal(data):
    return {"data": {key: data[key] for key in (
        "model_inputs", "metadata", "distance", "structure_inputs"
    )}}


def test_typed_json_round_trip(synthetic_data, tmp_path):
    config = config_from_dict(minimal(synthetic_data))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    loaded = load_config(path)
    assert loaded == config
    assert loaded.model.n_days == 6


def test_unknown_and_inconsistent_fields_are_rejected(synthetic_data):
    values = minimal(synthetic_data)
    values["unexpected"] = True
    with pytest.raises(ValueError, match="unknown top-level"):
        config_from_dict(values)
    values = minimal(synthetic_data)
    values["model"] = {"n_days": 7}
    with pytest.raises(ValueError, match="day_indices length"):
        config_from_dict(values)


def test_missing_paths_are_reported(synthetic_data):
    config = config_from_dict(minimal(synthetic_data))
    config.data.model_inputs = "/definitely/missing/model_inputs.npz"
    with pytest.raises(FileNotFoundError, match="model_inputs"):
        config.validate(check_paths=True)


def test_optional_test_split_is_valid(synthetic_data):
    values = minimal(synthetic_data)
    values["data"]["test_split"] = None
    config = config_from_dict(values)
    assert config.data.test_split is None


def test_non_null_split_names_remain_distinct(synthetic_data):
    values = minimal(synthetic_data)
    values["data"].update({"validation_split": "train", "test_split": None})
    with pytest.raises(ValueError, match="must be distinct"):
        config_from_dict(values)


def test_progress_setting_must_be_boolean(synthetic_data):
    values = minimal(synthetic_data)
    values["training"] = {"show_progress": "yes"}
    with pytest.raises(TypeError, match="show_progress"):
        config_from_dict(values)
