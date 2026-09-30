"""Train image-only and text-only baselines under identical conditions and evaluate them."""

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import CLASSES, load_images, load_metadata, split_rows
from explainmed.evaluate import metric_names, summarise_runs
from explainmed.model import ImageClassifier, ImageEncoder, TextClassifier, TextEncoder
from explainmed.text import describe_all, tokenize
from explainmed.train import resolve_device, start_mlflow, train_and_evaluate


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
    rows = split_rows(metadata, cfg, device)
    start_mlflow(cfg, "baselines")
    results = []
    for name, (inputs, build_model) in baselines(cfg, metadata, device).items():
        results += train_and_evaluate(
            name,
            inputs,
            build_model,
            metadata,
            rows,
            cfg,
            Path(args.config),
            cfg.paths.runs_dir / "baselines",
        )

    runs = pd.DataFrame(results)
    summary = summarise_runs(
        runs, [*metric_names(CLASSES), "val_macro_f1", "best_epoch"]
    )
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    runs.to_csv(
        cfg.paths.reports_dir / "baselines_runs.csv", index=False, **csv_options
    )
    summary.to_csv(cfg.paths.reports_dir / "baselines.csv", **csv_options)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
