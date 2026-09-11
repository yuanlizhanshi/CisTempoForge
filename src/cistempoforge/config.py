from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar


@dataclass
class DataConfig:
    model_inputs: str
    metadata: str
    distance: str
    structure_inputs: str
    day_indices: list[int] = field(default_factory=lambda: list(range(6)))
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str | None = "test"


@dataclass
class ModelConfig:
    n_days: int = 6
    variant: str = "independent_structure"
    fusion: str = "concat"
    hidden_dim: int = 128
    motif_filters: int = 32
    motif_width: int = 19
    dropout: float = 0.2
    detach_level_state: bool = False


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 24
    gradient_accumulation: int = 2
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    trend_weight: float = 1.0
    level_weight: float = 0.10
    local_trend_weight: float = 0.05
    level_encoder_trend_weight: float = 0.10
    reverse_complement_probability: float = 0.5
    num_workers: int = 4
    gradient_clip_norm: float = 5.0
    device: str = "cuda"
    mixed_precision: str = "auto"


@dataclass
class SelectionConfig:
    rmse_tolerance_fraction: float = 0.005


@dataclass
class CisTempoForgeConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def validate(self, *, check_paths: bool = False) -> None:
        if not isinstance(self.model.n_days, int) or isinstance(self.model.n_days, bool):
            raise TypeError("model.n_days must be an integer")
        if self.model.n_days < 2:
            raise ValueError("model.n_days must be at least 2")
        if len(self.data.day_indices) != self.model.n_days:
            raise ValueError("data.day_indices length must equal model.n_days")
        if len(set(self.data.day_indices)) != len(self.data.day_indices):
            raise ValueError("data.day_indices must be unique")
        if any(not isinstance(day, int) or day < 0 for day in self.data.day_indices):
            raise ValueError("data.day_indices must contain non-negative integers")
        split_names = [self.data.train_split, self.data.validation_split]
        if self.data.test_split is not None:
            split_names.append(self.data.test_split)
        if any(not isinstance(name, str) or not name for name in split_names):
            raise ValueError("configured split names must be non-empty strings")
        if len(set(split_names)) != len(split_names):
            raise ValueError("configured split names must be distinct")
        if not isinstance(self.model.hidden_dim, int) or isinstance(self.model.hidden_dim, bool):
            raise TypeError("model.hidden_dim must be an integer")
        if self.model.hidden_dim <= 0 or self.model.hidden_dim % 4:
            raise ValueError("model.hidden_dim must be positive and divisible by 4")
        if self.model.motif_width <= 0 or self.model.motif_width % 2 == 0:
            raise ValueError("model.motif_width must be a positive odd integer")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("model.dropout must be in [0, 1)")
        t = self.training
        for name in ("batch_size", "gradient_accumulation", "epochs", "patience"):
            if not isinstance(getattr(t, name), int) or isinstance(getattr(t, name), bool):
                raise TypeError(f"training.{name} must be an integer")
            if getattr(t, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if t.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")
        if t.learning_rate <= 0 or t.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0 <= t.reverse_complement_probability <= 1:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        if t.mixed_precision not in {"auto", "bf16", "fp16", "none"}:
            raise ValueError("mixed_precision must be auto, bf16, fp16, or none")
        if self.selection.rmse_tolerance_fraction < 0:
            raise ValueError("selection.rmse_tolerance_fraction cannot be negative")
        if check_paths:
            for name in ("model_inputs", "metadata", "distance", "structure_inputs"):
                path = Path(getattr(self.data, name)).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"data.{name} does not exist: {path}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: Mapping[str, Any], section: str) -> T:
    if not isinstance(values, Mapping):
        raise TypeError(f"{section} must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {section} fields: {sorted(unknown)}")
    try:
        return cls(**dict(values))
    except TypeError as exc:
        raise ValueError(f"invalid {section} configuration: {exc}") from exc


def config_from_dict(values: Mapping[str, Any]) -> CisTempoForgeConfig:
    if not isinstance(values, Mapping):
        raise TypeError("configuration must be an object")
    allowed = {"data", "model", "training", "selection"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown top-level configuration fields: {sorted(unknown)}")
    if "data" not in values:
        raise ValueError("configuration requires a data section")
    config = CisTempoForgeConfig(
        data=_strict_dataclass(DataConfig, values["data"], "data"),
        model=_strict_dataclass(ModelConfig, values.get("model", {}), "model"),
        training=_strict_dataclass(TrainingConfig, values.get("training", {}), "training"),
        selection=_strict_dataclass(SelectionConfig, values.get("selection", {}), "selection"),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> CisTempoForgeConfig:
    path = Path(path)
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration {path}: {exc}") from exc
    return config_from_dict(values)


def resolve_config(value: CisTempoForgeConfig | Mapping[str, Any] | str | Path) -> CisTempoForgeConfig:
    if isinstance(value, CisTempoForgeConfig):
        value.validate()
        return value
    if isinstance(value, Mapping):
        return config_from_dict(value)
    return load_config(value)
