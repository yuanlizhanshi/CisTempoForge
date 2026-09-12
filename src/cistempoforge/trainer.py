from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .checkpoint import checkpoint_payload, load_checkpoint, load_model, save_checkpoint
from .config import CisTempoForgeConfig, resolve_config
from .data import Normalization, denormalize, load_data, make_loader
from .engine import (
    build_lr_scheduler,
    evaluate,
    predict,
    resolve_amp,
    set_seed,
    train_epoch,
)
from .metrics import select_best_epoch, trajectory_metrics
from .model import CisTempoForgeModel
from .version import __version__


@dataclass(frozen=True)
class TrainingResult:
    model: CisTempoForgeModel
    best_checkpoint: Path
    last_checkpoint: Path
    history: pd.DataFrame
    # None when the run had no validation split and so nothing to measure.
    validation_metrics: dict[str, object] | None
    best_epoch: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state(generators: Mapping[str, torch.Generator] | None = None) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # Dedicated generators are independent of the global streams above, so
        # they need their own snapshot or a resumed run would replay the
        # shuffles and augmentations it has already consumed.
        "generators": {
            name: item.get_state() for name, item in (generators or {}).items()
        },
    }


def _restore_rng(state: dict, generators: Mapping[str, torch.Generator] | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor.  A checkpoint loaded with
    # map_location="cuda" also moves this small bookkeeping tensor to CUDA.
    torch.set_rng_state(state["torch"].detach().cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in state["cuda"]])
    saved = state.get("generators") or {}
    for name, item in (generators or {}).items():
        if name in saved:
            item.set_state(saved[name].detach().cpu())


def _write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(history).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _validate_output(output_dir: Path, resume_from: str | Path | None) -> None:
    managed = [output_dir / name for name in ("best.pt", "last.pt", "history.csv", "manifest.json")]
    if resume_from is None and any(path.exists() for path in managed):
        raise FileExistsError(f"output directory already contains a training run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _device(config: CisTempoForgeConfig) -> torch.device:
    device = torch.device(config.training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    return device


def _resume_signature(config_or_values) -> dict:
    """Return the numerically relevant configuration for resume checks."""
    values = resolve_config(config_or_values).to_dict()
    values["training"].pop("show_progress", None)
    return values


def fit(config_or_path, output_dir: str | Path, *, resume_from: str | Path | None = None) -> TrainingResult:
    """Train a model and write its checkpoints, history, and manifest.

    Set ``data.validation_split`` to ``None`` to train on every gene.  There is
    then nothing to early-stop or select on, so the run uses the full
    ``training.epochs`` budget and keeps the final epoch as ``best.pt``.  Fix
    that budget in advance from an independent estimate of the right training
    length -- a held-out split elsewhere, or a cross-validation study -- because
    the training loss alone cannot indicate where generalisation peaks.  Every
    epoch is checkpointed under ``.epoch_checkpoints/``, so a different epoch can
    be promoted to ``best.pt`` afterwards without retraining: runs are
    reproducible, so epoch K here is the epoch K a shorter run would end on.

    With a validation split the behaviour is unchanged: early stopping applies
    and ``best.pt`` is chosen by the configured selection rule.
    """
    config = resolve_config(config_or_path)
    config.validate(check_paths=True)
    output_dir = Path(output_dir)
    _validate_output(output_dir, resume_from)
    device = _device(config)
    amp_dtype, use_scaler = resolve_amp(device, config.training.mixed_precision)
    set_seed(config.training.seed)
    prepared = load_data(config)
    train_dataset = prepared.dataset(config.data.train_split)
    # A run with no validation split trains on every gene and has nothing to
    # early-stop or select on, so it runs the full budget and keeps its final
    # epoch.  Callers in that mode must fix `epochs` in advance from an
    # independent estimate of the right training length.
    validation_dataset = (
        prepared.dataset(config.data.validation_split)
        if config.data.validation_split is not None else None
    )
    # The shuffle order must depend only on the configured seed.  RandomSampler
    # otherwise draws its permutation from the global torch RNG, which model
    # construction has already perturbed by an amount that varies with the
    # architecture -- so two configurations would not see the same batches.
    loader_generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = make_loader(
        train_dataset, shuffle=True, batch_size=config.training.batch_size,
        num_workers=config.training.num_workers, generator=loader_generator,
    )
    validation_loader = (
        make_loader(
            validation_dataset, shuffle=False, batch_size=config.training.batch_size,
            num_workers=0,
        )
        if validation_dataset is not None else None
    )
    model = CisTempoForgeModel(**config.model.__dict__).to(device)
    # Re-seed after construction for the same reason: everything the training
    # loop draws from the global streams (dropout masks, and any op not routed
    # through a dedicated generator) then depends on the seed alone.
    set_seed(config.training.seed)
    # Reverse-complement augmentation runs on the accelerator, so it needs its
    # own generator: dropout consumes the global CUDA stream first and would
    # otherwise drag the augmentation choices apart across configurations.
    augmentation_generator = torch.Generator(device=device.type).manual_seed(
        config.training.seed
    )
    generators = {"loader": loader_generator, "augmentation": augmentation_generator}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_eps,
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_lr_scheduler(optimizer, config.training)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    history: list[dict] = []
    start_epoch, best_rmse, stale = 1, float("inf"), 0
    epoch_dir = output_dir / ".epoch_checkpoints"
    epoch_dir.mkdir(exist_ok=True)
    normalization = prepared.normalization.to_dict()

    if resume_from is not None:
        # Keep RNG and optimizer bookkeeping on CPU while deserializing. Model
        # and optimizer state loaders move tensors to their parameter devices.
        resume = load_checkpoint(resume_from, map_location="cpu")
        if _resume_signature(resume["config"]) != _resume_signature(config):
            raise ValueError("resume checkpoint configuration does not match requested configuration")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        if scaler is not None and "scaler_state" in resume:
            scaler.load_state_dict(resume["scaler_state"])
        # Checkpoints written before the schedule existed have no scheduler
        # state; the schedule then restarts from its first epoch, which is only
        # correct when resuming a run that never got past epoch one.
        if scheduler is not None and "scheduler_state" in resume:
            scheduler.load_state_dict(resume["scheduler_state"])
        if resume["normalization"] != normalization:
            raise ValueError("resume checkpoint normalization does not match current training data")
        history = list(resume.get("history", []))
        training_state = resume.get("training_state", {})
        best_rmse = float(training_state.get("best_rmse", float("inf")))
        stale = int(training_state.get("stale", 0))
        start_epoch = int(resume["epoch"]) + 1
        if "rng_state" in training_state:
            _restore_rng(training_state["rng_state"], generators)

    started = time.time()
    last_path = output_dir / "last.pt"
    for epoch in range(start_epoch, config.training.epochs + 1):
        epoch_label = f"Epoch {epoch:02d}/{config.training.epochs:02d}"
        train_loss = train_epoch(
            model, train_loader, optimizer, device, config.training,
            scaler=scaler, amp_dtype=amp_dtype,
            progress=config.training.show_progress,
            description=f"{epoch_label} - Training",
            augmentation_generator=augmentation_generator,
        )
        # Capture the rate this epoch actually trained at, then advance.  The
        # step lands before the checkpoint is written, so the stored
        # ``last_epoch`` always matches the epoch stored alongside it.
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            **{f"train_{key}": value for key, value in train_loss.items()},
            "elapsed_seconds": time.time() - started,
        }
        if validation_loader is not None:
            validation_loss = evaluate(
                model, validation_loader, device, config.training, amp_dtype=amp_dtype,
                progress=config.training.show_progress,
                description=f"{epoch_label} - Validation",
            )
            true, predicted, _ = predict(model, validation_loader, device, amp_dtype=amp_dtype)
            values = trajectory_metrics(true, predicted)
            record.update({
                **{f"validation_loss_{key}": value for key, value in validation_loss.items()},
                "validation_centered_rmse": values["centered_rmse"],
                "validation_median_trajectory_pearson": values["median_trajectory_pearson"],
                "validation_absolute_rmse": values["absolute_rmse"],
            })
        history.append(record)
        if validation_loader is not None:
            current_rmse = float(values["centered_rmse"])
            if current_rmse < best_rmse - 1e-7:
                best_rmse, stale = current_rmse, 0
            else:
                stale += 1
        training_state = {"best_rmse": best_rmse, "stale": stale, "rng_state": _rng_state(generators)}
        payload = checkpoint_payload(
            model=model, config=config, epoch=epoch, normalization=normalization,
            metrics=values if validation_loader is not None else {},
            package_version=__version__, optimizer=optimizer,
            scaler=scaler, scheduler=scheduler, history=history,
            training_state=training_state,
        )
        epoch_path = epoch_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(epoch_path, payload, overwrite=True)
        save_checkpoint(last_path, payload, overwrite=True)
        _write_history(output_dir / "history.csv", history)
        if config.training.show_progress:
            summary = f"{epoch_label} | train loss {train_loss['total']:.4f} | "
            if validation_loader is not None:
                summary += (
                    f"validation loss {validation_loss['total']:.4f} | "
                    f"centered RMSE {values['centered_rmse']:.4f} | "
                    f"median trajectory Pearson {values['median_trajectory_pearson']:.4f} | "
                    f"patience {stale}/{config.training.patience}"
                )
            else:
                summary += f"lr {epoch_lr:.2e} | no validation split"
            tqdm.write(summary)
        if validation_loader is not None and stale >= config.training.patience:
            if config.training.show_progress:
                tqdm.write(f"Early stopping after {epoch_label.lower()}.")
            break

    if not history:
        raise RuntimeError("no epochs were run; increase training.epochs or use an earlier checkpoint")
    if validation_loader is not None:
        selected = select_best_epoch(history, config.selection.rmse_tolerance_fraction)
        best_epoch = int(selected["epoch"])
        selection = {
            "best_epoch": best_epoch,
            "rule": "within configured tolerance of minimum validation centered RMSE; "
                    "highest median trajectory Pearson; earliest epoch final tie-break",
        }
    else:
        # Nothing was held out, so there is no evidence on which to prefer an
        # epoch.  Every epoch checkpoint is on disk; the caller is expected to
        # pick one from a budget fixed in advance and re-point `best.pt` at it.
        best_epoch = int(history[-1]["epoch"])
        selection = {
            "best_epoch": best_epoch,
            "rule": "no validation split; the final epoch is retained",
        }
    selected_path = epoch_dir / f"epoch_{best_epoch:03d}.pt"
    if not selected_path.exists():
        raise FileNotFoundError(
            "selected epoch checkpoint is unavailable; resume from the original output directory"
        )
    selected_state = load_checkpoint(selected_path, map_location="cpu")
    selected_state["selection"] = selection
    best_path = output_dir / "best.pt"
    save_checkpoint(best_path, selected_state, overwrite=True)
    best_model, _ = load_model(best_path, device=device)
    if validation_loader is not None:
        true, predicted, _ = predict(best_model, validation_loader, device, amp_dtype=amp_dtype)
        normalized_metrics = trajectory_metrics(true, predicted)
        final_metrics = trajectory_metrics(
            denormalize(true, prepared.normalization),
            denormalize(predicted, prepared.normalization),
        )
    else:
        normalized_metrics = None
        final_metrics = None
    manifest = {
        "package_version": __version__, "best_epoch": best_epoch,
        "epochs_run": len(history), "checkpoint": "best.pt",
        "checkpoint_sha256": _sha256(best_path),
        "training_genes": len(train_dataset),
        "validation_genes": len(validation_dataset) if validation_dataset is not None else 0,
        "test_loader_created": False,
        "normalization": normalization,
        "selection": selection,
        "validation_metrics_normalized": normalized_metrics,
        "validation_metrics": final_metrics,
        "device": str(device), "mixed_precision": str(amp_dtype),
    }
    temporary_manifest = output_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, output_dir / "manifest.json")
    return TrainingResult(
        best_model, best_path, last_path, pd.DataFrame(history),
        final_metrics, best_epoch,
    )
