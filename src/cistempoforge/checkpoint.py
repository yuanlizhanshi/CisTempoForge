from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import CisTempoForgeConfig, config_from_dict
from .model import ARCHITECTURE, CisTempoForgeModel


CHECKPOINT_FORMAT_VERSION = 1


def _model_options(config: CisTempoForgeConfig) -> dict[str, Any]:
    return config.model.__dict__.copy()


def checkpoint_payload(*, model: CisTempoForgeModel, config: CisTempoForgeConfig,
                       epoch: int, normalization: Mapping[str, float],
                       metrics: Mapping[str, Any], package_version: str,
                       optimizer: torch.optim.Optimizer | None = None,
                       scaler: torch.amp.GradScaler | None = None,
                       scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                       history: list[dict] | None = None,
                       training_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "package_version": package_version,
        "config": config.to_dict(),
        "model_options": _model_options(config),
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "normalization": dict(normalization),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    return payload


def save_checkpoint(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported CisTempoForge checkpoint format")
    if state.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint architecture is incompatible")
    config = config_from_dict(state["config"])
    if state.get("model_options") != config.model.__dict__:
        raise ValueError("checkpoint model options do not match its configuration")
    return state


def load_model(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[CisTempoForgeModel, dict[str, Any]]:
    device = torch.device(device)
    state = load_checkpoint(path, map_location=device)
    model = CisTempoForgeModel(**state["model_options"]).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, state
