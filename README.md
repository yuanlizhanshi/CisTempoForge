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

`fit` displays scPrinter-style, Notebook-aware progress bars for the training
and validation batches in every epoch, followed by a persistent epoch summary.
Set `config.training.show_progress = False` for quiet batch execution.

`fit` constructs only the configured training and validation datasets. The test
split is never used for training, early stopping, or checkpoint selection. After
model selection, create the test dataset explicitly and call `predict` to run
test inference.

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
