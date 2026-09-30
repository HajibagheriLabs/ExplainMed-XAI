"""Train image-only and text-only baselines under identical conditions and evaluate them."""

import argparse
import shutil
from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import (
    CLASSES,
    SPLITS,
    assign_splits,
    load_images,
    load_metadata,
    load_splits,
)
from explainmed.evaluate import classification_metrics
from explainmed.model import ImageClassifier, ImageEncoder, TextClassifier, TextEncoder
from explainmed.text import describe_all, tokenize
from explainmed.train import fit, flat_params, predict, resolve_device, start_mlflow

METRIC_COLUMNS = [
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    *(f"recall_{name}" for name in CLASSES),
    "val_macro_f1",
    "best_epoch",
]


def baselines(
    cfg: Config, metadata: pd.DataFrame, device: torch.device
) -> dict[str, tuple[dict[str, torch.Tensor], Callable[[], nn.Module]]]:
    """Inputs and a model builder for each baseline."""
    classes, dropout = len(CLASSES), cfg.model.dropout

    def image_model() -> nn.Module:
        return ImageClassifier(ImageEncoder(pretrained=True), classes, dropout)

    def text_model() -> nn.Module:
        return TextClassifier(TextEncoder(cfg.model.text_encoder), classes, dropout)

    def text_inputs(include_diagnosis_method: bool) -> dict[str, torch.Tensor]:
        texts = describe_all(metadata, include_diagnosis_method)
        return tokenize(texts, cfg.model.text_encoder, device)

    return {
        "image_only": ({"images": load_images(metadata, cfg, device)}, image_model),
        "text_only": (text_inputs(False), text_model),
        # diagnosis method leaks the label, so this row measures the leak, not a baseline
        "text_only_with_diagnosis_method": (text_inputs(True), text_model),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    transformers_logging.set_verbosity_error()

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    split = assign_splits(metadata, load_splits(cfg)).to_numpy()
    rows = {
        name: torch.tensor(np.flatnonzero(split == name), device=device)
        for name in SPLITS
    }
    labels = torch.tensor(metadata["label"].to_numpy(), device=device)
    test_rows = rows["test"].cpu().numpy()
    truth = metadata["label"].to_numpy()[test_rows]

    run_root = cfg.paths.runs_dir / "baselines"
    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_root / "config.yaml")
    start_mlflow(cfg, "baselines")

    results = []
    for name, (inputs, build_model) in baselines(cfg, metadata, device).items():
        for repeat in range(cfg.train.repeats):
            seed = cfg.seed + repeat
            run_dir = run_root / name / f"seed{seed}"
            with mlflow.start_run(run_name=f"{name}_seed{seed}"):
                mlflow.log_params({**flat_params(cfg), "model": name, "run_seed": seed})
                model = build_model()
                history = fit(
                    model,
                    inputs,
                    labels,
                    rows["train"],
                    rows["val"],
                    cfg,
                    seed,
                    run_dir / "best.pt",
                )
                for record in history:
                    epoch_metrics = {k: v for k, v in record.items() if k != "epoch"}
                    mlflow.log_metrics(epoch_metrics, step=record["epoch"])

                model.load_state_dict(torch.load(run_dir / "best.pt"))
                predicted, probabilities = predict(model, inputs, rows["test"], cfg)
                metrics = classification_metrics(truth, predicted, CLASSES)
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                mlflow.log_artifact(str(run_root / "config.yaml"))

            best = max(history, key=lambda record: record["val_macro_f1"])
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame(
                {
                    "image_id": metadata["image_id"].to_numpy()[test_rows],
                    "label": truth,
                    "predicted": predicted,
                    **{f"prob_{c}": probabilities[:, i] for i, c in enumerate(CLASSES)},
                }
            ).to_csv(run_dir / "test_predictions.csv", index=False)
            results.append(
                {
                    "model": name,
                    "seed": seed,
                    **metrics,
                    "val_macro_f1": best["val_macro_f1"],
                    "best_epoch": best["epoch"],
                }
            )
            print(
                f"{name} seed {seed}: test macro-F1 {metrics['macro_f1']:.4f}, "
                f"balanced accuracy {metrics['balanced_accuracy']:.4f}, "
                f"best epoch {best['epoch']}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

    runs = pd.DataFrame(results)
    summary = runs.groupby("model", sort=False)[METRIC_COLUMNS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "repeats", cfg.train.repeats)

    reports_dir = cfg.paths.reports_dir
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    runs.to_csv(reports_dir / "baselines_runs.csv", index=False, **csv_options)
    summary.to_csv(reports_dir / "baselines.csv", **csv_options)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
