from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .checkpoint import checkpoint_payload, load_checkpoint, load_model, save_checkpoint
from .config import CisTempoForgeConfig, resolve_config
from .data import Normalization, denormalize, load_data, make_loader
from .engine import evaluate, predict, resolve_amp, set_seed, train_epoch
from .metrics import select_best_epoch, trajectory_metrics
from .model import CisTempoForgeModel
from .version import __version__


@dataclass(frozen=True)
class TrainingResult:
    model: CisTempoForgeModel
    best_checkpoint: Path
    last_checkpoint: Path
    history: pd.DataFrame
    validation_metrics: dict[str, object]
    best_epoch: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


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


def fit(config_or_path, output_dir: str | Path, *, resume_from: str | Path | None = None) -> TrainingResult:
    config = resolve_config(config_or_path)
    config.validate(check_paths=True)
    output_dir = Path(output_dir)
    _validate_output(output_dir, resume_from)
    device = _device(config)
    amp_dtype, use_scaler = resolve_amp(device, config.training.mixed_precision)
    prepared = load_data(config)
    train_dataset = prepared.dataset(config.data.train_split)
    validation_dataset = prepared.dataset(config.data.validation_split)
    train_loader = make_loader(
        train_dataset, shuffle=True, batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )
    validation_loader = make_loader(
        validation_dataset, shuffle=False, batch_size=config.training.batch_size,
        num_workers=0,
    )
    set_seed(config.training.seed)
    model = CisTempoForgeModel(**config.model.__dict__).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else None
    history: list[dict] = []
    start_epoch, best_rmse, stale = 1, float("inf"), 0
    epoch_dir = output_dir / ".epoch_checkpoints"
    epoch_dir.mkdir(exist_ok=True)
    normalization = prepared.normalization.to_dict()

    if resume_from is not None:
        resume = load_checkpoint(resume_from, map_location=device)
        if resume["config"] != config.to_dict():
            raise ValueError("resume checkpoint configuration does not match requested configuration")
        model.load_state_dict(resume["model_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        if scaler is not None and "scaler_state" in resume:
            scaler.load_state_dict(resume["scaler_state"])
        if resume["normalization"] != normalization:
            raise ValueError("resume checkpoint normalization does not match current training data")
        history = list(resume.get("history", []))
        training_state = resume.get("training_state", {})
        best_rmse = float(training_state.get("best_rmse", float("inf")))
        stale = int(training_state.get("stale", 0))
        start_epoch = int(resume["epoch"]) + 1
        if "rng_state" in training_state:
            _restore_rng(training_state["rng_state"])

    started = time.time()
    last_path = output_dir / "last.pt"
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, config.training,
            scaler=scaler, amp_dtype=amp_dtype,
        )
        validation_loss = evaluate(
            model, validation_loader, device, config.training, amp_dtype=amp_dtype
        )
        true, predicted, _ = predict(model, validation_loader, device, amp_dtype=amp_dtype)
        values = trajectory_metrics(true, predicted)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_loss.items()},
            **{f"validation_loss_{key}": value for key, value in validation_loss.items()},
            "validation_centered_rmse": values["centered_rmse"],
            "validation_median_trajectory_pearson": values["median_trajectory_pearson"],
            "validation_absolute_rmse": values["absolute_rmse"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        current_rmse = float(values["centered_rmse"])
        if current_rmse < best_rmse - 1e-7:
            best_rmse, stale = current_rmse, 0
        else:
            stale += 1
        training_state = {"best_rmse": best_rmse, "stale": stale, "rng_state": _rng_state()}
        payload = checkpoint_payload(
            model=model, config=config, epoch=epoch, normalization=normalization,
            metrics=values, package_version=__version__, optimizer=optimizer,
            scaler=scaler, history=history, training_state=training_state,
        )
        epoch_path = epoch_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(epoch_path, payload, overwrite=True)
        save_checkpoint(last_path, payload, overwrite=True)
        _write_history(output_dir / "history.csv", history)
        if stale >= config.training.patience:
            break

    if not history:
        raise RuntimeError("no epochs were run; increase training.epochs or use an earlier checkpoint")
    selected = select_best_epoch(history, config.selection.rmse_tolerance_fraction)
    best_epoch = int(selected["epoch"])
    selected_path = epoch_dir / f"epoch_{best_epoch:03d}.pt"
    if not selected_path.exists():
        raise FileNotFoundError(
            "selected epoch checkpoint is unavailable; resume from the original output directory"
        )
    selected_state = load_checkpoint(selected_path, map_location="cpu")
    selected_state["selection"] = {
        "best_epoch": best_epoch,
        "rule": "within configured tolerance of minimum validation centered RMSE; "
                "highest median trajectory Pearson; earliest epoch final tie-break",
    }
    best_path = output_dir / "best.pt"
    save_checkpoint(best_path, selected_state, overwrite=True)
    best_model, _ = load_model(best_path, device=device)
    true, predicted, _ = predict(best_model, validation_loader, device, amp_dtype=amp_dtype)
    normalized_metrics = trajectory_metrics(true, predicted)
    final_metrics = trajectory_metrics(
        denormalize(true, prepared.normalization),
        denormalize(predicted, prepared.normalization),
    )
    manifest = {
        "package_version": __version__, "best_epoch": best_epoch,
        "epochs_run": len(history), "checkpoint": "best.pt",
        "checkpoint_sha256": _sha256(best_path),
        "training_genes": len(train_dataset),
        "validation_genes": len(validation_dataset),
        "test_loader_created": False,
        "normalization": normalization,
        "selection": selected_state["selection"],
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
