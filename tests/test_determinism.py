"""Guards for the RNG isolation that makes hyperparameter comparisons valid.

Model construction draws from the global torch RNG, so a configuration that
changes how much it draws (a wider hidden layer, more filters, a longer motif)
used to shift the training shuffle by an amount that grew with the size of the
model.  Two configurations then differed by batch order as well as by
hyperparameter, and the measured difference was uninterpretable.  These tests
pin that fix on the real `fit()` path.
"""

import torch

from cistempoforge import CisTempoForgeModel, config_from_dict, fit, load_data
from cistempoforge import engine
from cistempoforge.data import make_loader
from cistempoforge.engine import reverse_complement, set_seed


def training_config(data, epochs=1, **model_overrides):
    model = {
        "n_days": 6, "variant": "independent_structure", "fusion": "concat",
        "hidden_dim": 8, "motif_filters": 2, "motif_width": 19, "dropout": 0,
    }
    model.update(model_overrides)
    return config_from_dict({
        "data": {
            "model_inputs": data["model_inputs"], "metadata": data["metadata"],
            "distance": data["distance"], "structure_inputs": data["structure_inputs"],
            "day_indices": [0, 1, 2, 3, 4, 5],
        },
        "model": model,
        "training": {
            "epochs": epochs, "patience": 2, "batch_size": 2,
            "gradient_accumulation": 1, "num_workers": 0,
            "reverse_complement_probability": 0.5, "device": "cpu",
            "mixed_precision": "none",
        },
    })


def first_batch_order(monkeypatch, config, output):
    """Run fit() and report the gene indices the training loop saw first.

    `train_epoch` calls `reverse_complement` before touching the model, so this
    observes the true first batch without perturbing the loader.
    """
    captured = []
    original = engine.reverse_complement

    def spy(batch, probability, **kwargs):
        if not captured:
            captured.append(batch["gene_index"].clone())
        return original(batch, probability, **kwargs)

    monkeypatch.setattr(engine, "reverse_complement", spy)
    try:
        fit(config, output)
    finally:
        monkeypatch.setattr(engine, "reverse_complement", original)
    assert captured, "training ran no steps"
    return captured[0]


def test_batch_order_is_independent_of_model_architecture(synthetic_data, tmp_path, monkeypatch):
    """Widths and kernel sizes must not move the shuffle.

    Under the old ordering these configurations produced different first
    permutations, because each built a differently sized model between the
    seeding call and the first batch.
    """
    variants = [
        {},
        {"hidden_dim": 16},
        {"motif_filters": 6},
        {"motif_width": 21},
        {"dropout": 0.5},
    ]
    orders = [
        first_batch_order(
            monkeypatch, training_config(synthetic_data, **override),
            tmp_path / f"order_{index}",
        )
        for index, override in enumerate(variants)
    ]
    for index, order in enumerate(orders[1:], start=1):
        assert torch.equal(orders[0], order), (
            f"variant {index} {variants[index]} saw a different batch order "
            f"than the baseline: {order.tolist()} vs {orders[0].tolist()}"
        )


def test_reverse_complement_is_independent_of_global_rng():
    """The augmentation must be driven by its own generator.

    Dropout consumes the global accelerator stream before the first
    augmentation draw, so routing reverse_complement through the global RNG
    would let any architecture difference desynchronise the augmentation.
    """
    # Shapes as produced by the dataset: dna is [genes, peaks, 4, length] and
    # atac is [genes, days, peaks, 1, length].
    genes, peaks, days, length = 2, 3, 4, 8
    batch = {
        "dna": torch.zeros(genes, peaks, 4, length, dtype=torch.long),
        "atac": torch.arange(genes * days * peaks * length, dtype=torch.float32).reshape(
            genes, days, peaks, 1, length
        ),
        "peak_mask": torch.ones(genes, peaks, dtype=torch.bool),
    }

    def choices(global_draws: int):
        generator = torch.Generator().manual_seed(1234)
        torch.manual_seed(9999)
        for _ in range(global_draws):
            torch.rand(1)  # stand-in for the dropout masks model size perturbs
        return reverse_complement(batch, 0.5, generator=generator)["atac"]

    assert torch.equal(choices(0), choices(17))


def test_make_loader_generator_reproduces_the_same_order(synthetic_data):
    config = training_config(synthetic_data)
    dataset = load_data(config).dataset(config.data.train_split)

    def order():
        generator = torch.Generator().manual_seed(config.training.seed)
        loader = make_loader(
            dataset, shuffle=True, batch_size=2, num_workers=0, generator=generator,
        )
        return [batch["gene_index"] for batch in loader]

    first, second = order(), order()
    assert all(torch.equal(a, b) for a, b in zip(first, second))
