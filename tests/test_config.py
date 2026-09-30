from pathlib import Path

import pytest
import yaml

from explainmed.config import load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _write_variant(tmp_path: Path, section: str, key: str, value: object) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "variant.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads_with_typed_fields() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert isinstance(cfg.seed, int)
    assert isinstance(cfg.paths.metadata_csv, Path)
    assert isinstance(cfg.train.lr, float)
    assert cfg.data.val_fraction + cfg.data.test_fraction < 1.0


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = _write_variant(tmp_path, "train", "learning_rate", 0.1)
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leave a training set"):
        load_config(_write_variant(tmp_path, "data", "val_fraction", 0.9))
    with pytest.raises(TypeError, match="expected float"):
        load_config(_write_variant(tmp_path, "train", "lr", "1e-4"))
