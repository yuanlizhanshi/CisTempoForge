import math

import pytest
import torch

from cistempoforge import TrainingConfig, build_lr_scheduler


def optimizer_for(lr=1.0):
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.zeros(1)
    return torch.optim.SGD([parameter], lr=lr)


def rates(config, steps):
    """Return the learning rate used by each of ``steps`` epochs."""
    optimizer = optimizer_for()
    scheduler = build_lr_scheduler(optimizer, config)
    used = []
    for _ in range(steps):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # real loops step the optimizer before the schedule
        if scheduler is not None:
            scheduler.step()
    return used


def test_default_configuration_has_no_scheduler():
    """The default must reproduce the constant-rate behaviour exactly."""
    assert build_lr_scheduler(optimizer_for(), TrainingConfig()) is None


def test_warmup_reaches_peak_then_cosine_decays_to_the_floor():
    config = TrainingConfig(
        lr_schedule="cosine", warmup_fraction=0.1, min_lr_ratio=0.05, epochs=40,
    )
    used = rates(config, 40)

    assert used[0] == pytest.approx(0.25), "first warmup epoch should be a quarter in"
    assert used[3] == pytest.approx(1.0), "warmup should reach the peak"
    # Decay runs over epochs 4..39, so epoch 20 sits at progress 16/35.
    assert used[20] == pytest.approx(0.589, abs=0.005)
    assert used[-1] == pytest.approx(0.05, abs=1e-9), "final epoch should hit the floor"
    assert all(later <= earlier + 1e-9 for earlier, later in zip(used[3:], used[4:]))


def test_linear_decay_is_monotonic_and_never_below_the_floor():
    config = TrainingConfig(
        lr_schedule="linear", warmup_fraction=0.0, min_lr_ratio=0.1, epochs=10,
    )
    used = rates(config, 10)
    assert used[0] == pytest.approx(1.0)
    assert used[-1] == pytest.approx(0.1, abs=1e-6)
    assert all(later < earlier for earlier, later in zip(used, used[1:]))
    assert min(used) >= 0.1 - 1e-9


def test_warmup_without_decay_holds_the_peak_afterwards():
    config = TrainingConfig(lr_schedule="none", warmup_fraction=0.5, epochs=8)
    used = rates(config, 8)
    assert used[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert used[4:] == pytest.approx([1.0] * 4)


def test_scheduler_epochs_pins_the_horizon_for_short_runs():
    """A short run must see the same prefix of the curve as a long one.

    Budget-based comparisons (successive halving) rely on this: a 10-epoch run
    scored mid-schedule stays comparable to the 40-epoch run's first 10 epochs.
    """
    short = TrainingConfig(lr_schedule="cosine", epochs=10, scheduler_epochs=40)
    long = TrainingConfig(lr_schedule="cosine", epochs=40, scheduler_epochs=40)
    assert rates(short, 10) == pytest.approx(rates(long, 40)[:10])


def test_warmup_floor_gives_a_nonzero_first_rate():
    config = TrainingConfig(lr_schedule="cosine", warmup_fraction=0.25, epochs=40)
    used = rates(config, 40)
    assert all(rate > 0 for rate in used)
    assert used[9] == pytest.approx(1.0)


def _validate(training: TrainingConfig):
    """Route a TrainingConfig through the same validation real configs hit."""
    from cistempoforge import config_from_dict

    return config_from_dict({
        "data": {
            "model_inputs": "a", "metadata": "b", "distance": "c",
            "structure_inputs": "d",
        },
        "training": training.__dict__,
    })


@pytest.mark.parametrize("training, message", [
    (TrainingConfig(lr_schedule="exponential"), "lr_schedule"),
    (TrainingConfig(warmup_fraction=1.5), "warmup_fraction"),
    (TrainingConfig(min_lr_ratio=-0.1), "min_lr_ratio"),
    (TrainingConfig(huber_beta=0), "huber_beta"),
    (TrainingConfig(adam_beta2=1.0), "adam_beta1 and adam_beta2"),
    (TrainingConfig(adam_eps=0.0), "adam_eps"),
    (TrainingConfig(scheduler_epochs=0), "scheduler_epochs"),
])
def test_invalid_schedule_settings_are_rejected(training, message):
    with pytest.raises(ValueError, match=message):
        _validate(training)


def test_cosine_midpoint_matches_the_closed_form():
    """Sanity-check the curve itself, independent of the scheduler wrapper."""
    horizon, floor = 40, 0.05
    for epoch in (0, 7, 19, 33, 39):
        progress = epoch / (horizon - 1)
        expected = floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        config = TrainingConfig(
            lr_schedule="cosine", min_lr_ratio=floor, epochs=horizon,
            scheduler_epochs=horizon,
        )
        used = rates(config, horizon)
        assert used[epoch] == pytest.approx(expected, abs=0.02)
