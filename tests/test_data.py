import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from explainmed.config import load_config
from explainmed.data import (
    CLASSES,
    SPLITS,
    assign_splits,
    eval_transform,
    make_splits,
    train_transform,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def _synthetic_metadata() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for class_index, dx in enumerate(CLASSES):
        for lesion in range(40 * (class_index + 1)):
            lesion_id = f"lesion_{dx}_{lesion:04d}"
            for image in range(rng.integers(1, 4)):
                image_id = f"{lesion_id}_{image}"
                rows.append({"lesion_id": lesion_id, "image_id": image_id, "dx": dx})
    return pd.DataFrame(rows)


def test_no_lesion_spans_two_splits() -> None:
    metadata = _synthetic_metadata()
    generated = make_splits(metadata, VAL_FRACTION, TEST_FRACTION, seed=0)
    committed = json.loads((REPO_ROOT / "configs" / "splits.json").read_text())
    for splits in (generated, committed):
        train, val, test = (set(splits[name]) for name in SPLITS)
        assert not (train & val or train & test or val & test)

    assigned = assign_splits(metadata, generated)
    assert assigned.groupby(metadata["lesion_id"]).nunique().max() == 1


def test_splits_are_stratified_by_diagnosis() -> None:
    metadata = _synthetic_metadata()
    splits = make_splits(metadata, VAL_FRACTION, TEST_FRACTION, seed=0)
    lesion_dx = metadata.groupby("lesion_id")["dx"].first()
    overall = lesion_dx.value_counts(normalize=True)
    for name in SPLITS:
        share = lesion_dx.loc[splits[name]].value_counts(normalize=True)
        assert (share - overall).abs().max() < 0.01
    assert abs(len(splits["val"]) / len(lesion_dx) - VAL_FRACTION) < 0.01
    assert abs(len(splits["test"]) / len(lesion_dx) - TEST_FRACTION) < 0.01


def test_splits_are_deterministic_under_a_fixed_seed() -> None:
    metadata = _synthetic_metadata()
    first = make_splits(metadata, VAL_FRACTION, TEST_FRACTION, seed=7)
    shuffled = metadata.sample(frac=1.0, random_state=1)
    assert make_splits(shuffled, VAL_FRACTION, TEST_FRACTION, seed=7) == first
    assert make_splits(metadata, VAL_FRACTION, TEST_FRACTION, seed=8) != first


def test_transforms_keep_shape_and_eval_is_deterministic() -> None:
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").data
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(
        0, 256, (4, 3, 16, 16), dtype=torch.uint8, generator=generator
    )

    evaluated = eval_transform(images, cfg)
    assert evaluated.shape == images.shape
    assert evaluated.dtype == torch.float32
    assert torch.equal(evaluated, eval_transform(images, cfg))

    augmented = train_transform(images, cfg, generator)
    assert augmented.shape == images.shape
    assert augmented.dtype == torch.float32
    # flips and transposes only move pixels, so each channel keeps its set of values
    assert torch.equal(
        augmented.flatten(2).sort(dim=2).values, evaluated.flatten(2).sort(dim=2).values
    )
