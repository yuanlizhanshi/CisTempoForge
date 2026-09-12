from __future__ import annotations

import torch
from torch.nn import functional as F

from .config import TrainingConfig


def simple_huber_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                      config: TrainingConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Predictions and targets are z-scored, so beta is expressed in units of the
    # target standard deviation: values below 1 keep the loss quadratic over a
    # wider band, above 1 make it robust to a wider tail.
    beta = float(config.huber_beta)
    raw = {
        "trend": F.smooth_l1_loss(output["trend"], batch["trend_target"], beta=beta),
        "level": F.mse_loss(output["level"], batch["level_target"]),
        "local_trend": F.smooth_l1_loss(output["local_trend"], batch["trend_target"], beta=beta),
    }
    if output.get("level_encoder_trend") is not None:
        raw["level_encoder_trend"] = F.smooth_l1_loss(
            output["level_encoder_trend"], batch["trend_target"], beta=beta
        )
    weights = {
        "trend": config.trend_weight,
        "level": config.level_weight,
        "local_trend": config.local_trend_weight,
        "level_encoder_trend": config.level_encoder_trend_weight,
    }
    weighted = {name: value * float(weights[name]) for name, value in raw.items()}
    total = sum(weighted.values())
    pieces = {f"raw_{name}": value for name, value in raw.items()}
    pieces.update({f"weighted_{name}": value for name, value in weighted.items()})
    return total, pieces
