"""HAM10000 metadata loading, lesion-grouped splits, and image transforms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision.io import ImageReadMode, decode_jpeg, read_file

from explainmed.config import Config, DataConfig

CLASS_NAMES = {
    "akiec": "actinic keratosis / intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesion",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}
CLASSES = tuple(CLASS_NAMES)
SPLITS = ("train", "val", "test")
METADATA_COLUMNS = (
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
)


def load_metadata(metadata_csv: Path, images_dir: Path) -> pd.DataFrame:
    """One row per image, sorted by image_id, with an integer `label` and a file `path`."""
    metadata = pd.read_csv(metadata_csv)
    missing = set(METADATA_COLUMNS) - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_csv} is missing columns {sorted(missing)}")
    unknown = set(metadata["dx"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown diagnosis classes {sorted(unknown)}")
    if metadata["image_id"].duplicated().any():
        raise ValueError("image_id values are not unique")
    # stratifying lesions by diagnosis only makes sense if a lesion has exactly one
    if (metadata.groupby("lesion_id")["dx"].nunique() > 1).any():
        raise ValueError("a lesion_id maps to more than one diagnosis")

    metadata = metadata.sort_values("image_id", ignore_index=True)
    metadata["label"] = metadata["dx"].map(CLASSES.index)
    metadata["path"] = [str(images_dir / f"{i}.jpg") for i in metadata["image_id"]]
    absent = [path for path in metadata["path"] if not Path(path).exists()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} images listed in the metadata are missing, e.g. {absent[0]}"
        )
    return metadata


def make_splits(
    metadata: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[str]]:
    """Lesion ids per split, stratified by diagnosis; a lesion never spans two splits."""
    lesions = metadata.groupby("lesion_id")["dx"].first().sort_index()
    rest, test = train_test_split(
        lesions.index.to_numpy(),
        test_size=test_fraction,
        stratify=lesions.to_numpy(),
        random_state=seed,
    )
    train, val = train_test_split(
        rest,
        test_size=val_fraction / (1.0 - test_fraction),
        stratify=lesions.loc[rest].to_numpy(),
        random_state=seed,
    )
    return {
        "train": sorted(map(str, train)),
        "val": sorted(map(str, val)),
        "test": sorted(map(str, test)),
    }


def assign_splits(metadata: pd.DataFrame, splits: dict[str, list[str]]) -> pd.Series:
    """Split name per image; raises unless every lesion belongs to exactly one split."""
    owner: dict[str, str] = {}
    for name in SPLITS:
        for lesion in splits[name]:
            if lesion in owner:
                raise ValueError(
                    f"lesion {lesion} is in both {owner[lesion]} and {name}"
                )
            owner[lesion] = name
    lesions = set(metadata["lesion_id"])
    if lesions != set(owner):
        raise ValueError(
            f"splits do not match the metadata: {len(lesions - set(owner))} lesions "
            f"unassigned, {len(set(owner) - lesions)} not in the metadata"
        )
    return metadata["lesion_id"].map(owner).rename("split")


def split_rows(
    metadata: pd.DataFrame, cfg: Config, device: torch.device
) -> dict[str, torch.Tensor]:
    """Row indices of each committed split, as tensors on `device`."""
    split = assign_splits(metadata, load_splits(cfg)).to_numpy()
    return {
        name: torch.tensor(np.flatnonzero(split == name), device=device)
        for name in SPLITS
    }


def save_splits(splits: dict[str, list[str]], cfg: Config) -> None:
    payload = {**_split_provenance(cfg), **splits}
    cfg.paths.splits_file.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def load_splits(cfg: Config) -> dict[str, list[str]]:
    """Committed lesion ids per split; raises if they were generated under another config."""
    payload = json.loads(cfg.paths.splits_file.read_text(encoding="utf-8"))
    expected = _split_provenance(cfg)
    found = {key: payload.get(key) for key in expected}
    if found != expected:
        raise ValueError(
            f"{cfg.paths.splits_file} was written for {found}, config says "
            f"{expected}; regenerate it with `make data`"
        )
    return {name: payload[name] for name in SPLITS}


def _split_provenance(cfg: Config) -> dict[str, float | int]:
    return {
        "seed": cfg.seed,
        "val_fraction": cfg.data.val_fraction,
        "test_fraction": cfg.data.test_fraction,
    }


def load_images(
    metadata: pd.DataFrame, cfg: Config, device: torch.device
) -> torch.Tensor:
    """Every image as one uint8 tensor [N, 3, S, S] on `device`, in metadata row order."""
    size = cfg.data.image_size
    cache_file = cfg.paths.cache_dir / f"ham10000_{size}.pt"
    image_ids = metadata["image_id"].tolist()
    if cache_file.exists():
        cached = torch.load(cache_file)
        if cached["image_ids"] != image_ids:
            raise ValueError(f"{cache_file} does not match the metadata; delete it")
        return cached["images"].to(device)

    paths = metadata["path"].tolist()
    images = torch.empty((len(paths), 3, size, size), dtype=torch.uint8, device=device)
    step = cfg.data.decode_batch_size
    for start in range(0, len(paths), step):
        encoded = [read_file(path) for path in paths[start : start + step]]
        decoded = decode_jpeg(encoded, device=device, mode=ImageReadMode.RGB)
        # the full frame is squashed to a square so no part of the lesion is cropped away
        resized = F.interpolate(
            torch.stack(decoded).float(),
            size=(size, size),
            mode="bilinear",
            antialias=True,
        )
        images[start : start + step] = resized.round().clamp(0, 255).to(torch.uint8)

    cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"image_ids": image_ids, "images": images.cpu()}, cache_file)
    return images


def eval_transform(images: torch.Tensor, cfg: DataConfig) -> torch.Tensor:
    """Deterministic scaling and normalisation of uint8 images to float32."""
    mean = torch.tensor(cfg.normalize_mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.normalize_std, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def train_transform(
    images: torch.Tensor, cfg: DataConfig, generator: torch.Generator
) -> torch.Tensor:
    """Random flips and transposes followed by the eval transform."""
    # dermatoscopy has no canonical orientation, so all 8 square symmetries keep labels
    draws = torch.rand((3, images.shape[0]), generator=generator, device=images.device)
    transpose, flip_rows, flip_cols = (draws < 0.5).view(3, -1, 1, 1, 1)
    images = torch.where(transpose, images.transpose(2, 3), images)
    images = torch.where(flip_rows, images.flip(2), images)
    images = torch.where(flip_cols, images.flip(3), images)
    return eval_transform(images, cfg)
