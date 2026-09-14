# CisTempoForge

CisTempoForge is an installable Python package for temporal cis-regulatory
expression prediction. It turns the reusable data, model, SIMPLE_HUBER loss,
training, checkpoint selection, prediction, and checkpoint logic from the D0–D5
research code into a standalone library suitable for notebooks and independent
training projects.

## Installation

```bash
python -m pip install -e '.[test]'
```

For CUDA training, install a PyTorch build compatible with the host NVIDIA
driver. CisTempoForge supports Python 3.10 and later.

## Quick start

```python
from cistempoforge import fit, load_config, load_model

config = load_config("config.json")
config.training.learning_rate = 1e-4
config.training.epochs = 40

result = fit(config, output_dir="runs/d0_d5_seed42")
model, checkpoint = load_model(result.best_checkpoint, device="cuda")
```

`fit` displays Notebook-aware progress bars for the training and validation
batches in every epoch, followed by a persistent epoch summary.
Set `config.training.show_progress = False` for quiet batch execution.

`fit` constructs only the configured training and validation datasets. The test
split is never used for training, early stopping, or checkpoint selection. After
model selection, create the test dataset explicitly and call `predict` to run
test inference.

### Learning-rate schedule

`learning_rate` is constant unless a schedule is requested. Set
`lr_schedule` to `"cosine"` or `"linear"` to decay from the peak, optionally
after a warmup:

```python
config.training.lr_schedule = "cosine"
config.training.warmup_fraction = 0.05   # fraction of the horizon
config.training.min_lr_ratio = 0.05      # floor, as a fraction of learning_rate
config.training.scheduler_epochs = 40    # horizon; defaults to training.epochs
```

The schedule advances once per epoch and is laid out over `scheduler_epochs`
rather than over the epochs actually run, so early stopping cannot leave the
rate unannealed. Pin `scheduler_epochs` explicitly when comparing runs of
different lengths — a short run then sees the same prefix of the curve as a
long one. The first epoch of the horizon reaches the floor exactly.

Other training knobs: `adam_beta1`, `adam_beta2`, `adam_eps`, and `huber_beta`
(the transition point of the SmoothL1 terms on trend heads; the level head
stays MSE). Note that changing `huber_beta` rescales `validation_loss_*`, so
compare runs on prediction metrics such as `centered_rmse`, never on loss.

### Reproducibility

Everything that affects a run's trajectory is derived from
`training.seed` alone. The shuffle order and the reverse-complement
augmentation use dedicated generators seeded from it, and the global streams
are re-seeded after model construction. Two configurations that differ only in
hyperparameters therefore see an identical sequence of batches and
augmentations, which is what makes A/B comparison meaningful. Changing the seed
changes the shuffle, the augmentation, the initialization, and dropout.

By default, training refuses to overwrite an output directory containing
`best.pt`, `last.pt`, `history.csv`, or `manifest.json`. Resume an interrupted
run with:

```python
result = fit(
    config,
    "runs/d0_d5_seed42",
    resume_from="runs/d0_d5_seed42/last.pt",
)
```

See `examples/train_from_python.py`, `examples/notebook_quickstart.ipynb`, and
`examples/d0_d5_config.json` for complete examples.

## Input data contract

All paths in the configuration are explicit paths resolved by the calling
process. The package does not depend on a particular repository layout.

- `model_inputs.npz` must contain `dna [peaks,4,length]`,
  `atac [peaks,days,length]`, `target [genes,days]`,
  `peak_mask [genes,max_peaks]`, a matching `gene_peak_index` array,
  `atac_channel_mean`, and `atac_channel_std`.
- `metadata.parquet` must contain one row per gene and include at least `split`
  and `n_selected_promoter_peaks` columns.
- The distance `.npy` file must have shape `[genes,max_peaks]`.
- The structure `.npz` file must contain `features` and `masks`, both with shape
  `[genes,8]`.

Target normalization is estimated exclusively from the training split and the
selected `day_indices`. `n_days` must equal the number of day indices. Six-day,
seven-day, and other windows containing at least two time points are supported.

## Stable public API

The primary high-level entry points are `fit`, `load_config`, and `load_model`.
For custom training loops, the package also exposes `CisTempoForgeModel`,
`CisTempoForgeDataset`, `load_data`, `make_loader`, `simple_huber_loss`,
`train_epoch`, `evaluate`, `predict`, `save_checkpoint`, and
`trajectory_metrics`.

The checkpoint format is versioned starting with CisTempoForge `0.1.0`.
Checkpoints produced by the historical D0–D5 scripts use a different format and
are not guaranteed to load with this package.

For cross-validation workflows that use every sample in either training or
validation, set `config.data.test_split = None`. The default remains `"test"`
for conventional train/validation/test experiments.

## Training outputs

The caller owns `output_dir`. CisTempoForge writes:

- `best.pt`: the checkpoint selected by the preregistered selection rule;
- `last.pt`: the latest complete training state for recovery;
- `.epoch_checkpoints/`: per-epoch states required for exact selection and
  interrupted-run recovery;
- `history.csv` and `manifest.json`.

Checkpoint selection first retains epochs whose centered RMSE is no greater
than `1 + tolerance` times the minimum centered RMSE. It then selects the epoch
with the highest median trajectory Pearson correlation, using the earliest
epoch as the final tie-break. The default tolerance is 0.5%.

## License

MIT License. Copyright (c) 2026 yuanlizhanshi.
