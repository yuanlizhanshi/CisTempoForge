from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch

from .config import TrainingConfig
from .losses import simple_huber_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_amp(device: torch.device, policy: str) -> tuple[torch.dtype | None, bool]:
    if device.type != "cuda" or policy == "none":
        return None, False
    if policy == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is not supported by this GPU")
        return torch.bfloat16, False
    if policy == "fp16":
        return torch.float16, True
    if policy == "auto":
        return (torch.bfloat16, False) if torch.cuda.is_bf16_supported() else (torch.float16, True)
    raise ValueError(f"unknown mixed precision policy: {policy}")


def _autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def reverse_complement(batch: dict[str, torch.Tensor], probability: float) -> dict[str, torch.Tensor]:
    if probability <= 0:
        return batch
    result = dict(batch)
    dna, atac = result["dna"], result["atac"]
    choose = (
        torch.rand(dna.shape[:2], device=dna.device) < probability
    ) & result["peak_mask"].bool()
    rc = dna[:, :, [3, 2, 1, 0]].flip(-1)
    result["dna"] = torch.where(choose[:, :, None, None], rc, dna)
    result["atac"] = torch.where(choose[:, None, :, None, None], atac.flip(-1), atac)
    return result


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(model: torch.nn.Module, loader: Iterable, optimizer: torch.optim.Optimizer,
                device: str | torch.device, config: TrainingConfig,
                *, scaler: torch.amp.GradScaler | None = None,
                amp_dtype: torch.dtype | None = None) -> dict[str, float]:
    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    count = 0
    accumulation = config.gradient_accumulation
    total_steps = len(loader)  # type: ignore[arg-type]
    for step, cpu_batch in enumerate(loader, 1):
        batch = reverse_complement(
            _to_device(cpu_batch, device), config.reverse_complement_probability
        )
        with _autocast(device, amp_dtype):
            output = model(batch)
            total, pieces = simple_huber_loss(output, batch, config)
        scaled_loss = total / accumulation
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if step % accumulation == 0 or step == total_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        size = len(batch["target"])
        values = {"total": total, **pieces}
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * size
        count += size
    if count == 0:
        raise ValueError("training loader is empty")
    return {name: value / count for name, value in sums.items()}


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: Iterable, device: str | torch.device,
             config: TrainingConfig, *, amp_dtype: torch.dtype | None = None) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast(device, amp_dtype):
            output = model(batch)
            total, pieces = simple_huber_loss(output, batch, config)
        size = len(batch["target"])
        for name, value in {"total": total, **pieces}.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * size
        count += size
    if count == 0:
        raise ValueError("evaluation loader is empty")
    return {name: value / count for name, value in sums.items()}


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: Iterable, device: str | torch.device,
            *, amp_dtype: torch.dtype | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device(device)
    model.eval()
    true, predicted, ids = [], [], []
    for cpu_batch in loader:
        batch = _to_device(cpu_batch, device)
        with _autocast(device, amp_dtype):
            output = model(batch)
        true.append(batch["target"].float().cpu().numpy())
        predicted.append(output["absolute"].float().cpu().numpy())
        ids.append(batch["gene_index"].cpu().numpy())
    if not true:
        raise ValueError("prediction loader is empty")
    return np.concatenate(true), np.concatenate(predicted), np.concatenate(ids)
