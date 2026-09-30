"""Adebayo et al. model- and label-randomisation sanity checks for every attribution method."""

import argparse
import dataclasses

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import CLASSES, load_images, load_metadata, split_rows
from explainmed.evaluate import classification_metrics
from explainmed.explain import METHODS, MODALITY, attribute, load_fusion
from explainmed.faithfulness import fixed_subset, unit_scores, valid_units
from explainmed.model import FusionClassifier, build_fusion
from explainmed.sanity import cascading_randomisation, map_similarity, shuffled_labels
from explainmed.style import (
    FIGURE_STYLE,
    METHOD_COLOURS,
    METHOD_LABELS,
    SECONDARY_INK,
    SEQUENTIAL,
)
from explainmed.text import describe_all, tokenize
from explainmed.train import (
    fit,
    flat_params,
    make_batch,
    predict,
    resolve_device,
    run_seeds,
    seed_everything,
    start_mlflow,
)

MEASURES = ("rank_correlation", "top_overlap", "collapsed")


def short_name(layer: str) -> str:
    """Axis label for a module path such as image_encoder.features.8."""
    last = layer.rsplit(".", 1)[-1]
    if layer == "fusion.4":
        return "classifier output"
    if layer == "fusion.1":
        return "fusion hidden"
    if layer.startswith("image_encoder.features."):
        return f"image block {last}"
    if layer.endswith("embeddings"):
        return "text embeddings"
    return f"text layer {last}"


def attributions(
    model: FusionClassifier,
    methods: list[str],
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    cfg: Config,
) -> dict[str, torch.Tensor]:
    """Raw maps of each method, at the input resolution."""
    return {m: attribute(m, model, batch, target, cfg.explain) for m in methods}


def compare(
    original: dict[str, torch.Tensor],
    other: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: Config,
    generator: torch.Generator,
) -> dict[str, dict[str, float]]:
    """Mean similarity per method between two sets of maps, on the faithfulness units."""
    results = {}
    for method, maps in other.items():
        modality = MODALITY[method]
        first = unit_scores(original[method], modality, cfg.faithfulness.patch_size)
        second = unit_scores(maps, modality, cfg.faithfulness.patch_size)
        valid = valid_units(batch, modality, first.shape[1])
        values = map_similarity(
            first, second, valid, cfg.faithfulness.top_fraction, generator
        )
        rank = values["rank_correlation"].cpu().numpy()
        results[method] = {
            **{m: float(np.nanmean(values[m].cpu().numpy())) for m in MEASURES},
            "images": int(np.isfinite(rank).sum()),
        }
    return results


def rows_for(
    seed: int, check: str, step: int, layer: str, results: dict[str, dict]
) -> list[dict]:
    return [
        {"seed": seed, "check": check, "method": method, "step": step, "layer": layer}
        | values
        for method, values in results.items()
    ]


def train_on_shuffled_labels(
    cfg: Config,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    rows: dict[str, torch.Tensor],
    seed: int,
) -> tuple[FusionClassifier, dict[str, float | int]]:
    """The fusion model retrained briefly on permuted labels, and how well it fit them."""
    shuffled = shuffled_labels(labels, seed)
    short = dataclasses.replace(
        cfg,
        train=dataclasses.replace(
            cfg.train, epochs=cfg.sanity.label_randomisation_epochs
        ),
    )
    run_dir = cfg.paths.runs_dir / "sanity" / "label_randomised" / f"seed{seed}"
    with mlflow.start_run(run_name=f"label_randomised_seed{seed}"):
        mlflow.log_params({**flat_params(short), "run_seed": seed})
        seed_everything(seed)
        model = build_fusion(cfg.model, len(CLASSES), pretrained=True)
        # selecting on the shuffled training labels keeps the epoch that memorised them best
        history = fit(
            model,
            inputs,
            shuffled,
            rows["train"],
            rows["train"],
            short,
            seed,
            run_dir / "best.pt",
        )
        for record in history:
            mlflow.log_metrics(
                {
                    "train_loss": record["train_loss"],
                    "train_macro_f1_shuffled_labels": record["val_macro_f1"],
                },
                step=record["epoch"],
            )
        model.load_state_dict(torch.load(run_dir / "best.pt"))
        predicted, _ = predict(model, inputs, rows["test"], cfg)
        truth = labels[rows["test"]].cpu().numpy()
        test = classification_metrics(truth, predicted, CLASSES)
        best = max(history, key=lambda record: record["val_macro_f1"])
        summary = {
            "seed": seed,
            "epochs": short.train.epochs,
            "best_epoch": best["epoch"],
            "train_macro_f1_shuffled_labels": best["val_macro_f1"],
            "test_macro_f1_true_labels": test["macro_f1"],
            "test_balanced_accuracy_true_labels": test["balanced_accuracy"],
        }
        mlflow.log_metrics({k: v for k, v in summary.items() if k != "seed"})
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    return model.eval(), summary


def verdicts(runs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """A method passes a check only if its similarity is below the threshold every time."""
    rows = []
    for method in METHODS:
        mine = runs[runs["method"] == method]
        cascade = mine[mine["check"] == "model_randomisation"]
        label = mine[mine["check"] == "label_randomisation"]
        reference = mine[mine["check"] == "independent_seed"]
        by_step = cascade.groupby("step", sort=True)["rank_correlation"].mean()
        worst = cascade.loc[cascade["rank_correlation"].idxmax()]
        passes_model = bool((cascade["rank_correlation"] < threshold).all())
        passes_label = bool((label["rank_correlation"] < threshold).all())
        if passes_model and passes_label:
            outcome = "passes both checks"
        elif passes_model or passes_label:
            failed = "label" if passes_model else "model"
            outcome = f"fails {failed} randomisation"
        else:
            outcome = "fails both checks"
        rows.append(
            {
                "method": method,
                "threshold": threshold,
                "first_step_similarity": by_step.iloc[0],
                "final_step_similarity": by_step.iloc[-1],
                "highest_similarity": worst["rank_correlation"],
                "highest_similarity_layer": worst["layer"],
                "passes_model_randomisation": passes_model,
                "label_randomisation_similarity": label["rank_correlation"].mean(),
                "passes_label_randomisation": passes_label,
                "independent_seed_similarity": reference["rank_correlation"].mean(),
                "verdict": outcome,
            }
        )
    return pd.DataFrame(rows)


def plot_checks(summary: pd.DataFrame, threshold: float) -> Figure:
    fig = Figure(figsize=(12, 4.6), layout="constrained")
    axes = fig.subplots(1, 3, width_ratios=[1.5, 1.1, 0.9])
    cascade = summary[summary["check"] == "model_randomisation"]
    for ax, modality in zip(axes[:2], ("image", "text")):
        for method in [m for m in METHODS if MODALITY[m] == modality]:
            line = cascade[cascade["method"] == method].sort_values("step")
            mean, spread = line["rank_correlation_mean"], line["rank_correlation_std"]
            ax.plot(
                line["step"],
                mean,
                color=METHOD_COLOURS[method],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=METHOD_LABELS[method],
            )
            ax.fill_between(
                line["step"],
                mean - spread,
                mean + spread,
                color=METHOD_COLOURS[method],
                alpha=0.15,
                linewidth=0,
            )
            labels = [short_name(layer) for layer in line["layer"]]
        ax.set_xticks(line["step"], labels, rotation=55, ha="right")
        ax.set_title(f"Model randomisation, {modality} branch", loc="left")
        ax.set_xlabel("randomised down to, from the output")
        ax.legend(loc="upper right")
    label = summary[summary["check"] == "label_randomisation"].set_index("method")
    reference = summary[summary["check"] == "independent_seed"].set_index("method")
    positions = np.arange(len(METHODS))
    axes[2].bar(
        positions,
        [label.loc[m, "rank_correlation_mean"] for m in METHODS],
        yerr=[label.loc[m, "rank_correlation_std"] for m in METHODS],
        color=[METHOD_COLOURS[m] for m in METHODS],
        width=0.6,
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": SECONDARY_INK},
        label="retrained on shuffled labels",
    )
    axes[2].scatter(
        positions,
        [reference.loc[m, "rank_correlation_mean"] for m in METHODS],
        marker="D",
        s=28,
        color=SECONDARY_INK,
        zorder=3,
        label="another seed trained on true labels",
    )
    axes[2].set_xticks(positions, [METHOD_LABELS[m] for m in METHODS], rotation=20)
    axes[2].set_title("Label randomisation", loc="left")
    axes[2].legend(loc="upper right", fontsize=8)
    lowest = (summary["rank_correlation_mean"] - summary["rank_correlation_std"]).min()
    bottom = min(-0.2, np.floor(lowest * 10) / 10)
    for ax in axes:
        ax.axhline(threshold, color=SECONDARY_INK, linestyle="--", linewidth=1)
        # headroom above 1 keeps the legends clear of the data
        ax.set_ylim(bottom, 1.4)
        ax.set_yticks(np.arange(np.ceil(bottom * 5) / 5, 1.01, 0.2))
        ax.grid(axis="y")
    axes[0].set_ylabel("rank correlation with the original map")
    axes[0].annotate(
        f"fail at or above {threshold}",
        (0, threshold),
        xycoords=("axes fraction", "data"),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=SECONDARY_INK,
    )
    fig.supxlabel(
        "Mean over the sanity subset and three fusion seeds, with the across-seed sd. "
        "A valid explanation must lose its similarity to the original map.",
        fontsize=8,
        color=SECONDARY_INK,
    )
    return fig


def plot_examples(
    image: np.ndarray, columns: dict[str, dict[str, np.ndarray]], title: str
) -> Figure:
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    methods = [m for m in METHODS if MODALITY[m] == "image"]
    fig = Figure(figsize=(2.0 * (len(columns) + 1), 4.6), layout="constrained")
    axes = fig.subplots(len(methods), len(columns) + 1, squeeze=False)
    for row, method in enumerate(methods):
        axes[row, 0].imshow(image)
        axes[row, 0].set_ylabel(METHOD_LABELS[method], fontsize=10)
        for column, (heading, maps) in enumerate(columns.items(), start=1):
            heat = maps[method]
            top = max(float(np.quantile(heat, 0.99)), 1e-12)
            axes[row, column].imshow(np.clip(heat / top, 0, 1), cmap=colormap)
            if row == 0:
                axes[row, column].set_title(heading, fontsize=9, fontweight="normal")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    axes[0, 0].set_title("input", fontsize=9, fontweight="normal")
    fig.suptitle(title, x=0.0, ha="left", fontsize=11, fontweight="semibold")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    # deterministic cudnn kernels make gradient attributions reproducible to the bit
    seed_everything(cfg.seed)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    rows = split_rows(metadata, cfg, device)
    subset = fixed_subset(rows["test"], cfg.faithfulness.subset_size, cfg.seed)
    subset = subset[: cfg.sanity.subset_size]
    texts = describe_all(metadata, include_diagnosis_method=False)
    inputs = {
        "images": load_images(metadata, cfg, device),
        **tokenize(texts, cfg.model.text_encoder, device),
    }
    labels = torch.tensor(metadata["label"].to_numpy(), device=device)
    batch = make_batch(inputs, subset, cfg, generator=None)
    seeds = run_seeds(cfg)
    start_mlflow(cfg, "sanity")

    records, label_runs, example_columns = [], [], {}
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        model = load_fusion(cfg, seed, device)
        predicted, _ = predict(model, inputs, subset, cfg)
        # every comparison explains the class this model predicted, so only weights differ
        target = torch.as_tensor(predicted, device=device)
        original = attributions(model, list(METHODS), batch, target, cfg)
        keep_examples = seed == cfg.seed
        if keep_examples:
            example_columns["trained model"] = {
                m: v[0].cpu().numpy() for m, v in original.items()
            }

        other_seed = seeds[(index + 1) % len(seeds)]
        other = load_fusion(cfg, other_seed, device)
        compared = attributions(other, list(METHODS), batch, target, cfg)
        records += rows_for(
            seed,
            "independent_seed",
            0,
            "another seed trained on true labels",
            compare(original, compared, batch, cfg, generator),
        )
        del other, compared

        for modality in ("image", "text"):
            methods = [m for m in METHODS if MODALITY[m] == modality]
            cascade = cascading_randomisation(model, modality, seed)
            for step, (layer, randomised) in enumerate(cascade, start=1):
                maps = attributions(randomised, methods, batch, target, cfg)
                records += rows_for(
                    seed,
                    "model_randomisation",
                    step,
                    layer,
                    compare(original, maps, batch, cfg, generator),
                )
                if keep_examples and modality == "image":
                    example_columns[short_name(layer)] = {
                        m: v[0].cpu().numpy() for m, v in maps.items()
                    }
            print(f"seed {seed}: {modality} cascade done", flush=True)

        shuffled_model, summary = train_on_shuffled_labels(
            cfg, inputs, labels, rows, seed
        )
        label_runs.append(summary)
        maps = attributions(shuffled_model, list(METHODS), batch, target, cfg)
        records += rows_for(
            seed,
            "label_randomisation",
            0,
            "retrained on shuffled labels",
            compare(original, maps, batch, cfg, generator),
        )
        if keep_examples:
            example_columns["shuffled labels"] = {
                m: v[0].cpu().numpy() for m, v in maps.items() if MODALITY[m] == "image"
            }
        print(f"seed {seed}: label randomisation done", flush=True)
        del model, shuffled_model
        torch.cuda.empty_cache()

    runs = pd.DataFrame(records)
    summary = runs.groupby(["check", "method", "step", "layer"], sort=False)[
        [*MEASURES, "images"]
    ].agg(["mean", "std"])
    summary.columns = [f"{measure}_{stat}" for measure, stat in summary.columns]
    summary = summary.reset_index()
    threshold = cfg.sanity.similarity_threshold
    verdict_table = verdicts(runs, threshold)

    reports_dir = cfg.paths.reports_dir
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    runs.to_csv(reports_dir / "sanity_runs.csv", index=False, **csv_options)
    summary.to_csv(reports_dir / "sanity_checks.csv", index=False, **csv_options)
    verdict_table.to_csv(
        reports_dir / "sanity_verdicts.csv", index=False, **csv_options
    )
    pd.DataFrame(label_runs).to_csv(
        reports_dir / "label_randomisation.csv", index=False, **csv_options
    )

    image_id = metadata["image_id"].iloc[int(subset[0])]
    trained, *cascade_names, shuffled = list(example_columns)
    # the head's two layers, the top image block, mid-branch, and the stem
    steps = [0, 1, 2, len(cascade_names) // 2, len(cascade_names) - 1]
    chosen = [trained, *(cascade_names[i] for i in steps), shuffled]
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(FIGURE_STYLE):
        plot_checks(summary, threshold).savefig(figures / "sanity_checks.png", dpi=150)
        plot_examples(
            inputs["images"][int(subset[0])].permute(1, 2, 0).cpu().numpy(),
            {name: example_columns[name] for name in chosen},
            f"Image maps of {image_id} as the fusion model (seed {cfg.seed}) is randomised "
            "from the output down, and after retraining on shuffled labels",
        ).savefig(figures / "sanity_examples.png", dpi=150)

    with mlflow.start_run(run_name="sanity_checks"):
        mlflow.log_params(
            {"subset_size": cfg.sanity.subset_size, "threshold": threshold}
        )
        for name in (
            "sanity_checks.csv",
            "sanity_verdicts.csv",
            "label_randomisation.csv",
        ):
            mlflow.log_artifact(str(reports_dir / name))
        mlflow.log_artifact(str(figures / "sanity_checks.png"))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(verdict_table.T.to_string())
        print(pd.DataFrame(label_runs).to_string(index=False))


if __name__ == "__main__":
    main()
