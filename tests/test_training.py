import numpy as np
import pytest
import torch

from cistempoforge import (
    CisTempoForgeModel,
    TrainingConfig,
    config_from_dict,
    fit,
    load_checkpoint,
    load_model,
    select_best_epoch,
    simple_huber_loss,
)
from cistempoforge.trainer import _restore_rng, _rng_state


def training_config(data, epochs=2, **training_overrides):
    training = {
        "epochs": epochs, "patience": 2, "batch_size": 2,
        "gradient_accumulation": 2, "num_workers": 0,
        "reverse_complement_probability": 0, "device": "cpu",
        "mixed_precision": "none",
    }
    training.update(training_overrides)
    return config_from_dict({
        "data": {
            "model_inputs": data["model_inputs"], "metadata": data["metadata"],
            "distance": data["distance"], "structure_inputs": data["structure_inputs"],
            "day_indices": [0, 1, 2, 3, 4, 5],
        },
        "model": {
            "n_days": 6, "variant": "independent_structure", "fusion": "concat",
            "hidden_dim": 8, "motif_filters": 2, "motif_width": 19, "dropout": 0,
        },
        "training": training,
    })


def test_loss_weights_are_applied():
    output = {
        "trend": torch.ones(2, 3), "level": torch.ones(2),
        "local_trend": torch.ones(2, 3),
        "level_encoder_trend": torch.ones(2, 3),
    }
    batch = {
        "trend_target": torch.zeros(2, 3), "level_target": torch.zeros(2),
    }
    config = TrainingConfig(
        trend_weight=1, level_weight=0.1, local_trend_weight=0.05,
        level_encoder_trend_weight=0.1,
    )
    total, pieces = simple_huber_loss(output, batch, config)
    expected = sum(value for name, value in pieces.items() if name.startswith("weighted_"))
    assert torch.equal(total, expected)


def test_selection_rule():
    history = [
        {"epoch": 1, "validation_centered_rmse": 1.0, "validation_median_trajectory_pearson": 0.2},
        {"epoch": 2, "validation_centered_rmse": 1.004, "validation_median_trajectory_pearson": 0.8},
        {"epoch": 3, "validation_centered_rmse": 1.006, "validation_median_trajectory_pearson": 0.9},
    ]
    assert select_best_epoch(history, 0.005)["epoch"] == 2


def test_fit_checkpoint_reload_and_resume(synthetic_data, tmp_path):
    config = training_config(synthetic_data)
    output = tmp_path / "run"
    result = fit(config, output)
    assert result.best_checkpoint.exists() and result.last_checkpoint.exists()
    assert (output / "manifest.json").exists()
    assert (output / ".epoch_checkpoints").is_dir()
    model, state = load_model(result.best_checkpoint)
    assert model.n_days == 6 and state["format_version"] == 1
    before = {name: value.clone() for name, value in model.state_dict().items()}
    reloaded, _ = load_model(result.best_checkpoint)
    assert all(torch.equal(before[name], value) for name, value in reloaded.state_dict().items())
    resumed = fit(config, output, resume_from=result.last_checkpoint)
    assert resumed.best_epoch == result.best_epoch
    assert len(resumed.history) == len(result.history)
    assert load_checkpoint(result.last_checkpoint)["normalization"] == state["normalization"]


def test_fit_progress_reports_each_epoch(synthetic_data, tmp_path, capsys):
    config = training_config(synthetic_data, epochs=1)
    config.training.show_progress = True
    fit(config, tmp_path / "progress_run")
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Epoch 01/01" in combined
    assert "centered RMSE" in combined


def test_resume_accepts_legacy_checkpoint_without_progress_setting(synthetic_data, tmp_path):
    config = training_config(synthetic_data, epochs=1)
    config.training.show_progress = False
    output = tmp_path / "legacy_resume"
    first = fit(config, output)
    state = torch.load(first.last_checkpoint, map_location="cpu", weights_only=False)
    state["config"]["training"].pop("show_progress")
    legacy = output / "legacy_last.pt"
    torch.save(state, legacy)
    config.training.show_progress = True
    resumed = fit(config, output, resume_from=legacy)
    assert len(resumed.history) == 1


def test_learning_rate_schedule_is_applied_and_recorded(synthetic_data, tmp_path):
    config = training_config(
        synthetic_data, epochs=6, patience=6,
        lr_schedule="cosine", min_lr_ratio=0.1,
    )
    result = fit(config, tmp_path / "scheduled")
    rates = result.history["learning_rate"]
    assert rates.iloc[0] == pytest.approx(config.training.learning_rate)
    assert rates.iloc[-1] == pytest.approx(config.training.learning_rate * 0.1)
    assert rates.is_monotonic_decreasing, "no warmup configured, so decay is monotone"


def test_constant_rate_is_unchanged_when_no_schedule_is_requested(synthetic_data, tmp_path):
    """The default path must still train at a flat rate."""
    config = training_config(synthetic_data, epochs=3, patience=3)
    result = fit(config, tmp_path / "flat")
    rates = result.history["learning_rate"].unique()
    assert rates.tolist() == [config.training.learning_rate]


def test_scheduler_state_is_checkpointed(synthetic_data, tmp_path):
    config = training_config(
        synthetic_data, epochs=3, patience=3, lr_schedule="cosine", warmup_fraction=0.34,
    )
    result = fit(config, tmp_path / "stateful")
    state = load_checkpoint(result.last_checkpoint, map_location="cpu")
    assert "scheduler_state" in state
    assert state["scheduler_state"]["last_epoch"] == 3


def test_generator_state_round_trips():
    """Resume must restore the dedicated generators, not just the global RNG.

    Without this the resumed run would replay permutations and augmentation
    draws the interrupted run had already consumed.
    """
    loader_generator = torch.Generator().manual_seed(3)
    torch.rand(5, generator=loader_generator)  # advance past the checkpoint
    state = _rng_state({"loader": loader_generator})
    expected = torch.rand(3, generator=loader_generator)

    restored = torch.Generator().manual_seed(99)
    _restore_rng(state, {"loader": restored})
    assert torch.equal(expected, torch.rand(3, generator=restored))


def test_restore_rng_tolerates_checkpoints_without_generator_state():
    """Checkpoints written before 0.3.0 carry no generator snapshot."""
    state = _rng_state()
    state.pop("generators")
    _restore_rng(state, {"loader": torch.Generator().manual_seed(1)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_restore_rng_accepts_cuda_mapped_checkpoint_state():
    state = _rng_state()
    state["torch"] = state["torch"].cuda()
    if state["cuda"] is not None:
        state["cuda"] = [item.cuda() for item in state["cuda"]]
    _restore_rng(state)
