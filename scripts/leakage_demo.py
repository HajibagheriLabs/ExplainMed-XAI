"""Train one model under a lesion-grouped and a naive image-level split to expose leakage."""

import argparse
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18

from explainmed.config import Config, load_config
from explainmed.data import (
    CLASSES,
    assign_splits,
    eval_transform,
    load_images,
    load_metadata,
    make_splits,
    train_transform,
)
from explainmed.train import resolve_device, seed_everything

VALUE_COLUMNS = [
    "n_train",
    "n_test",
    "test_images_with_lesion_in_train",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "accuracy_single_image_lesions",
    "accuracy_multi_image_lesions",
]


def naive_image_split(
    metadata: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> pd.Series:
    """Split name per image, stratified by diagnosis but blind to lesion_id."""
    rows = np.arange(len(metadata))
    dx = metadata["dx"].to_numpy()
    rest, test = train_test_split(
        rows, test_size=test_fraction, stratify=dx, random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_fraction / (1.0 - test_fraction),
        stratify=dx[rest],
        random_state=seed,
    )
    split = np.empty(len(metadata), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return pd.Series(split, index=metadata.index, name="split")


def fit_and_predict(
    images: torch.Tensor,
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    test_rows: torch.Tensor,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    demo = cfg.leakage_demo
    device = images.device
    use_amp = device.type == "cuda"
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=demo.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = -(-len(train_rows) // demo.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=demo.lr, total_steps=demo.epochs * steps_per_epoch
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for _ in range(demo.epochs):
        order = torch.randperm(len(train_rows), generator=generator, device=device)
        for batch in train_rows[order].split(demo.batch_size):
            inputs = train_transform(images[batch], cfg.data, generator)
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.cross_entropy(model(inputs), labels[batch])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    model.eval()
    predicted = []
    with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
        for batch in test_rows.split(demo.batch_size):
            logits = model(eval_transform(images[batch], cfg.data))
            predicted.append(logits.argmax(dim=1))
    return torch.cat(predicted).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)

    run_dir = cfg.paths.runs_dir / "leakage_demo"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    images = load_images(metadata, cfg, device)
    labels = torch.tensor(metadata["label"].to_numpy(), device=device)
    fractions = (cfg.data.val_fraction, cfg.data.test_fraction)
    # only lesions with several images can leak, so single-image ones are the control
    lesion_sizes = metadata.groupby("lesion_id")["image_id"].transform("size")
    multi_image = (lesion_sizes > 1).to_numpy()

    runs = []
    for repeat in range(cfg.leakage_demo.repeats):
        seed = cfg.seed + repeat
        strategies = {
            "lesion_grouped": assign_splits(
                metadata, make_splits(metadata, *fractions, seed)
            ),
            "image_naive": naive_image_split(metadata, *fractions, seed),
        }
        for name, split in strategies.items():
            train = (split == "train").to_numpy()
            test = (split == "test").to_numpy()
            train_lesions = set(metadata.loc[train, "lesion_id"])
            leaked = metadata.loc[test, "lesion_id"].isin(train_lesions).to_numpy()

            predicted = fit_and_predict(
                images,
                labels,
                torch.tensor(np.flatnonzero(train), device=device),
                torch.tensor(np.flatnonzero(test), device=device),
                cfg,
                seed,
            )
            truth = metadata.loc[test, "label"].to_numpy()
            correct = predicted == truth
            runs.append(
                {
                    "split": name,
                    "repeat": repeat,
                    "seed": seed,
                    "n_train": train.sum(),
                    "n_test": test.sum(),
                    "test_images_with_lesion_in_train": leaked.sum(),
                    "accuracy": accuracy_score(truth, predicted),
                    "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                    "macro_f1": f1_score(truth, predicted, average="macro"),
                    "accuracy_single_image_lesions": correct[~multi_image[test]].mean(),
                    "accuracy_multi_image_lesions": correct[multi_image[test]].mean(),
                }
            )
            print(runs[-1], flush=True)

    runs = pd.DataFrame(runs)
    # both splits share a seed per repeat, so the gap is a paired difference
    by_repeat = runs.pivot(index="repeat", columns="split", values=VALUE_COLUMNS)
    by_repeat = by_repeat.swaplevel(axis=1)
    gap = by_repeat["image_naive"] - by_repeat["lesion_grouped"]
    gap = gap.astype(float).assign(split="naive_minus_grouped")
    summary = pd.concat([runs, gap]).groupby("split", sort=False)[VALUE_COLUMNS]
    summary = summary.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "repeats", cfg.leakage_demo.repeats)

    reports_dir = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    runs.to_csv(reports_dir / "leakage_demo_runs.csv", index=False, **csv_options)
    summary.to_csv(reports_dir / "leakage_demo.csv", **csv_options)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
