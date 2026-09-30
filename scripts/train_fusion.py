"""Train the late-fusion model under the baseline conditions and compare it with both."""

import argparse
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from torch import nn
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import CLASSES, load_images, load_metadata, split_rows
from explainmed.evaluate import (
    classification_metrics,
    confusion_matrix,
    metric_names,
    summarise_runs,
)
from explainmed.model import FusionClassifier, ImageEncoder, TextEncoder
from explainmed.style import FIGURE_STYLE, INK, SEQUENTIAL, SURFACE
from explainmed.text import describe_all, tokenize
from explainmed.train import predict, resolve_device, start_mlflow, train_and_evaluate

MODALITY_INPUTS = {"text": ("input_ids", "attention_mask"), "image": ("images",)}
SUMMARY_ORDER = [
    "image_only",
    "text_only",
    "fusion",
    "fusion_minus_image_only",
    "fusion_text_shuffled",
    "fusion_image_shuffled",
]


def shuffled_modality_runs(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    test_rows: torch.Tensor,
    truth: np.ndarray,
    cfg: Config,
    seed: int,
) -> list[dict]:
    """Test metrics with one modality permuted across test images at a time."""
    test_inputs = {name: tensor[test_rows] for name, tensor in inputs.items()}
    order = torch.arange(len(test_rows), device=test_rows.device)
    generator = torch.Generator(device=test_rows.device).manual_seed(seed)
    permutation = torch.randperm(
        len(test_rows), generator=generator, device=test_rows.device
    )
    runs = []
    for modality, names in MODALITY_INPUTS.items():
        shuffled = {**test_inputs, **{n: test_inputs[n][permutation] for n in names}}
        predicted, _ = predict(model, shuffled, order, cfg)
        runs.append(
            {
                "model": f"fusion_{modality}_shuffled",
                "seed": seed,
                **classification_metrics(truth, predicted, CLASSES),
            }
        )
    return runs


def pooled_confusion(model_dir: Path) -> np.ndarray:
    """Confusion counts summed over the test predictions of every seed."""
    files = sorted(model_dir.glob("seed*/test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"no test predictions under {model_dir}")
    total = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for file in files:
        predictions = pd.read_csv(file)
        total += confusion_matrix(
            predictions["label"], predictions["predicted"], len(CLASSES)
        )
    return total


def plot_confusions(matrices: dict[str, np.ndarray]) -> Figure:
    fig = Figure(figsize=(12, 5.2), layout="constrained")
    axes = fig.subplots(1, len(matrices))
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    for ax, (title, counts) in zip(axes, matrices.items()):
        share = counts / counts.sum(axis=1, keepdims=True)
        image = ax.imshow(share, cmap=colormap, vmin=0.0, vmax=1.0)
        for (row, column), value in np.ndenumerate(share):
            # cells under half a percent would print as 0%, so they stay blank
            if value >= 0.005:
                ax.text(
                    column,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=SURFACE if value > 0.55 else INK,
                )
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(title)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.colorbar(
        image,
        ax=axes,
        shrink=0.85,
        label="share of the true class",
        format=PercentFormatter(1.0),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    transformers_logging.set_verbosity_error()

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    rows = split_rows(metadata, cfg, device)
    texts = describe_all(metadata, include_diagnosis_method=False)
    inputs = {
        "images": load_images(metadata, cfg, device),
        **tokenize(texts, cfg.model.text_encoder, device),
    }

    def build_model() -> nn.Module:
        return FusionClassifier(
            ImageEncoder(pretrained=True),
            TextEncoder(cfg.model.text_encoder),
            len(CLASSES),
            cfg.model.fusion_hidden_dim,
            cfg.model.dropout,
        )

    run_root = cfg.paths.runs_dir / "fusion"
    start_mlflow(cfg, "fusion")
    results = train_and_evaluate(
        "fusion", inputs, build_model, metadata, rows, cfg, Path(args.config), run_root
    )

    truth = metadata["label"].to_numpy()[rows["test"].cpu().numpy()]
    for seed in [record["seed"] for record in results]:
        model = build_model().to(device)
        model.load_state_dict(
            torch.load(run_root / "fusion" / f"seed{seed}" / "best.pt")
        )
        results += shuffled_modality_runs(model, inputs, rows["test"], truth, cfg, seed)
    fusion_runs = pd.DataFrame(results)

    baseline_runs = pd.read_csv(cfg.paths.reports_dir / "baselines_runs.csv")
    baseline_runs = baseline_runs[
        baseline_runs["model"].isin(["image_only", "text_only"])
    ]
    runs = pd.concat([baseline_runs, fusion_runs], ignore_index=True)
    columns = metric_names(CLASSES)
    by_seed = runs.set_index(["model", "seed"])[columns]
    # fusion and image-only share seeds, so batch order and augmentation are paired too
    gain = (by_seed.loc["fusion"] - by_seed.loc["image_only"]).dropna()
    if len(gain) != len(by_seed.loc["fusion"]):
        raise ValueError("fusion and image-only runs do not share the same seeds")
    gain = gain.reset_index().assign(model="fusion_minus_image_only")
    summary = summarise_runs(pd.concat([runs, gain]), columns).loc[SUMMARY_ORDER]

    baseline_dir = cfg.paths.runs_dir / "baselines" / "image_only"
    matrices = {
        "image_only": pooled_confusion(baseline_dir),
        "fusion": pooled_confusion(run_root / "fusion"),
    }
    confusion_rows = [
        {
            "model": model,
            "true_class": CLASSES[row],
            "predicted_class": CLASSES[column],
            "count": int(count),
            "share_of_true_class": count / counts[row].sum(),
        }
        for model, counts in matrices.items()
        for (row, column), count in np.ndenumerate(counts)
    ]

    reports_dir = cfg.paths.reports_dir
    figure_file = reports_dir / "figures" / "confusion_matrices.png"
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    fusion_runs.to_csv(reports_dir / "fusion_runs.csv", index=False, **csv_options)
    summary.to_csv(reports_dir / "fusion_results.csv", **csv_options)
    pd.DataFrame(confusion_rows).to_csv(
        reports_dir / "confusion_matrices.csv", index=False, **csv_options
    )
    seeds = cfg.train.repeats
    with matplotlib.rc_context(FIGURE_STYLE):
        titles = {
            "image_only": f"Image only, test split, {seeds} seeds pooled",
            "fusion": f"Fusion, test split, {seeds} seeds pooled",
        }
        fig = plot_confusions({titles[m]: counts for m, counts in matrices.items()})
        fig.savefig(figure_file, dpi=150)

    with mlflow.start_run(run_name="fusion_comparison"):
        mlflow.log_artifact(str(reports_dir / "fusion_results.csv"))
        mlflow.log_artifact(str(figure_file))
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
