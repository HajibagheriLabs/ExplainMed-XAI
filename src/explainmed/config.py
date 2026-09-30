"""Experiment configuration loaded from YAML into frozen dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class PathsConfig:
    metadata_csv: Path
    images_dir: Path
    cache_dir: Path
    splits_file: Path
    reports_dir: Path
    runs_dir: Path
    mlflow_dir: Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    decode_batch_size: int
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]
    val_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        for name in ("image_size", "decode_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"data.{name} must be positive, got {getattr(self, name)}"
                )
        for name in ("normalize_mean", "normalize_std"):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"data.{name} must have one value per rgb channel")
        if min(self.normalize_std) <= 0:
            raise ValueError("data.normalize_std values must be positive")
        for name in ("val_fraction", "test_fraction"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"data.{name} must be in (0, 1), got {value}")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                "data.val_fraction + data.test_fraction must be below 1 to leave a training set"
            )


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"train.{name} must be positive, got {getattr(self, name)}"
                )
        if self.lr <= 0:
            raise ValueError(f"train.lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(
                f"train.weight_decay must be non-negative, got {self.weight_decay}"
            )


@dataclass(frozen=True)
class LeakageDemoConfig:
    repeats: int
    epochs: int
    batch_size: int
    lr: float

    def __post_init__(self) -> None:
        for name in ("repeats", "epochs", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"leakage_demo.{name} must be positive, got {getattr(self, name)}"
                )
        if self.lr <= 0:
            raise ValueError(f"leakage_demo.lr must be positive, got {self.lr}")


@dataclass(frozen=True)
class Config:
    seed: int
    device: str
    cpu_threads: int
    paths: PathsConfig
    data: DataConfig
    train: TrainConfig
    leakage_demo: LeakageDemoConfig

    def __post_init__(self) -> None:
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.cpu_threads <= 0:
            raise ValueError(f"cpu_threads must be positive, got {self.cpu_threads}")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _build(Config, raw, "config")


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = {field.name for field in dataclasses.fields(cls)}
    missing, unknown = names - raw.keys(), raw.keys() - names
    if missing or unknown:
        raise ValueError(
            f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}"
        )
    return cls(
        **{name: _coerce(hints[name], raw[name], f"{where}.{name}") for name in names}
    )


def _coerce(expected: type, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(expected):
        return _build(expected, value, where)
    if get_origin(expected) is tuple and isinstance(value, list):
        item_type = get_args(expected)[0]
        return tuple(
            _coerce(item_type, item, f"{where}[{i}]") for i, item in enumerate(value)
        )
    if expected is Path and isinstance(value, str):
        return Path(value)
    # bool subclasses int, so without this `epochs: true` would pass as 1
    if isinstance(value, bool) and expected is not bool:
        raise TypeError(f"{where}: expected {expected.__name__}, got bool")
    if expected is float and isinstance(value, int):
        return float(value)
    if not isinstance(value, expected):
        # pyyaml reads `1e-4` as a string, so this is usually a missing decimal point
        raise TypeError(
            f"{where}: expected {expected.__name__}, got {type(value).__name__} {value!r}"
        )
    return value
