"""Score every attribution method against a random-attribution control on a fixed test subset."""

import argparse
from statistics import NormalDist

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import CLASSES, load_images, load_masks, load_metadata, split_rows
from explainmed.explain import MODALITY, attribute, load_fusion
from explainmed.faithfulness import (
    area_under_curve,
    blurred_reference,
    fixed_subset,
    fractions,
    gini,
    lesion_localisation,
    perturbation_curves,
    random_attribution,
    rank_correlation,
    top_overlap,
    unit_scores,
    valid_units,
)
from explainmed.style import FIGURE_STYLE, METHOD_COLOURS, METHOD_LABELS, SECONDARY_INK
from explainmed.text import describe_all, tokenize
from explainmed.train import (
    make_batch,
    predict,
    resolve_device,
    run_seeds,
    start_mlflow,
)

CONTROLS = {"image": "random_image", "text": "random_text"}
# the blurred image is the protocol fixed in advance; mean colour checks its sensitivity
PRIMARY = "blurred image"
MODES = ("deletion", "insertion")
CHECK = "mean colour"
AGREEMENT_PAIRS = [
    ("grad_cam", "integrated_gradients"),
    ("grad_cam", "random_image"),
    ("integrated_gradients", "random_image"),
]
PER_IMAGE_METRICS = [
    "deletion_auc",
    "insertion_auc",
    "gini",
    "mass_in_lesion",
    "mass_chance",
    "pointing_game",
    "pointing_chance",
    "empty_map",
]
CHECK_METRICS = ["deletion_auc_mean_colour", "insertion_auc_mean_colour"]


def modality_of(method: str) -> str:
    return MODALITY.get(method) or next(m for m, c in CONTROLS.items() if c == method)


def suffix(reference: str) -> str:
    return "" if reference == PRIMARY else "_" + reference.replace(" ", "_")


def batched_curves(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    units: torch.Tensor,
    modality: str,
    target: torch.Tensor,
    reference: torch.Tensor,
    cfg: Config,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    parts = {"deletion": [], "insertion": []}
    for rows in torch.arange(len(target), device=target.device).split(
        cfg.train.batch_size
    ):
        chunk = {name: tensor[rows] for name, tensor in batch.items()}
        result = perturbation_curves(
            model,
            chunk,
            units[rows],
            modality,
            target[rows],
            reference[rows],
            cfg.faithfulness,
            generator,
        )
        for mode, collected in parts.items():
            collected.append(result[mode])
    return {mode: torch.cat(collected) for mode, collected in parts.items()}


def evaluate_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    lesion: torch.Tensor,
    references: dict[str, torch.Tensor],
    cfg: Config,
    seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]]]:
    """Per-image metrics and deletion and insertion curves of every method and control."""
    fc = cfg.faithfulness
    generator = torch.Generator(device=target.device).manual_seed(seed)
    scores = {
        method: unit_scores(
            attribute(method, model, batch, target, cfg.explain),
            modality,
            fc.patch_size,
        )
        for method, modality in MODALITY.items()
    }
    for modality, control in CONTROLS.items():
        size = next(s for m, s in scores.items() if MODALITY[m] == modality).shape[1]
        scores[control] = random_attribution(
            valid_units(batch, modality, size), generator
        )

    metrics, curves = {}, {}
    for method, units in scores.items():
        modality = modality_of(method)
        valid = valid_units(batch, modality, units.shape[1])
        curves[method] = batched_curves(
            model, batch, units, modality, target, references[PRIMARY], cfg, generator
        )
        metrics[method] = {
            "deletion_auc": area_under_curve(curves[method]["deletion"], fc),
            "insertion_auc": area_under_curve(curves[method]["insertion"], fc),
            "gini": gini(units, valid),
            # an all-zero map ranks its units at random, exactly like the control
            "empty_map": (units.sum(dim=1) == 0).float(),
        }
        if modality == "image":
            metrics[method].update(lesion_localisation(units, lesion))
            for name, reference in references.items():
                if name == PRIMARY:
                    continue
                check = batched_curves(
                    model, batch, units, modality, target, reference, cfg, generator
                )
                for mode, curve in check.items():
                    metrics[method][f"{mode}_auc{suffix(name)}"] = area_under_curve(
                        curve, fc
                    )
    metrics["agreement"] = {}
    for first, second in AGREEMENT_PAIRS:
        valid = valid_units(batch, "image", scores[first].shape[1])
        pair = f"{first}~{second}"
        metrics["agreement"][f"{pair}~rank_correlation"] = rank_correlation(
            scores[first], scores[second], valid
        )
        metrics["agreement"][f"{pair}~top_overlap"] = top_overlap(
            scores[first], scores[second], valid, fc.top_fraction, generator
        )
    return metrics, curves


def paired_against_control(
    per_image: pd.DataFrame, level: float, reference: str = PRIMARY
) -> list[dict[str, float | int | str | bool]]:
    """Per seed and method, the mean paired AUC difference to the control with its CI."""
    z = NormalDist().inv_cdf(0.5 + level / 2)
    end = suffix(reference)
    rows = []
    for (seed, method), group in per_image.groupby(["seed", "method"], sort=False):
        control = CONTROLS[modality_of(method)]
        if method == control or group[f"deletion_auc{end}"].isna().all():
            continue
        paired = group.set_index("image_id")
        base = per_image[(per_image["seed"] == seed) & (per_image["method"] == control)]
        base = base.set_index("image_id").loc[paired.index]
        record = {"seed": seed, "method": method}
        for mode, better in (("deletion", -1), ("insertion", 1)):
            difference = paired[f"{mode}_auc{end}"] - base[f"{mode}_auc{end}"]
            half_width = z * difference.std(ddof=1) / np.sqrt(len(difference))
            mean = difference.mean()
            record[f"{mode}_minus_random"] = mean
            record[f"{mode}_ci_low"] = mean - half_width
            record[f"{mode}_ci_high"] = mean + half_width
            record[f"{mode}_share_better"] = float((better * difference > 0).mean())
            # "clearly" means the whole interval lies on one side of zero
            if better * (mean - better * half_width) > 0:
                record[f"{mode}_vs_random"] = "better"
            elif better * (mean + better * half_width) < 0:
                record[f"{mode}_vs_random"] = "worse"
            else:
                record[f"{mode}_vs_random"] = "inconclusive"
        rows.append(record)
    return rows


def summarise(by_seed: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and sd over seeds per group, and whether the method beat random in every seed."""
    skip = {*keys, "seed", "images"}
    numeric = [
        c for c in by_seed.columns if c not in skip and not c.endswith("_vs_random")
    ]
    grouped = by_seed.groupby(keys, sort=False)
    summary = grouped[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    for mode in MODES:
        summary[f"{mode}_vs_random"] = grouped[f"{mode}_vs_random"].agg(across_seeds)
    methods = [key if isinstance(key, str) else key[0] for key in summary.index]
    summary["verdict"] = [
        "control" if method in CONTROLS.values() else verdict(d, i)
        for method, d, i in zip(
            methods, summary["deletion_vs_random"], summary["insertion_vs_random"]
        )
    ]
    return summary


def across_seeds(statuses: pd.Series) -> str:
    """A comparison holds only if it holds in every seed."""
    if statuses.isna().all():
        return ""
    return statuses.iloc[0] if statuses.nunique(dropna=False) == 1 else "inconclusive"


def verdict(deletion: str, insertion: str) -> str:
    statuses = {"deletion": deletion, "insertion": insertion}
    better = [mode for mode, status in statuses.items() if status == "better"]
    worse = [mode for mode, status in statuses.items() if status == "worse"]
    if len(better) == len(MODES):
        return "beats random on deletion and insertion"
    parts = [f"beats random on {better[0]} only" if better else "never beats random"]
    if worse:
        parts.append(f"worse than random on {' and '.join(worse)}")
    return "; ".join(parts)


def plot_curves(curves: pd.DataFrame, summary: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10, 7.4), layout="constrained")
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    titles = {
        "deletion": "Deletion: most-attributed first, lower is better",
        "insertion": "Insertion: most-attributed first, higher is better",
    }
    for row, modality in enumerate(("image", "text")):
        methods = [m for m in METHOD_LABELS if modality_of(m) == modality]
        for column, mode in enumerate(("deletion", "insertion")):
            ax = axes[row, column]
            for method in methods:
                line = curves[(curves["method"] == method) & (curves["mode"] == mode)]
                auc = summary.loc[method, f"{mode}_auc_mean"]
                ax.plot(
                    line["fraction"],
                    line["probability_mean"],
                    color=METHOD_COLOURS[method],
                    linestyle="--" if method in CONTROLS.values() else "-",
                    linewidth=1.8,
                    label=f"{METHOD_LABELS[method]} (AUC {auc:.3f})",
                )
            unit = "image patches" if modality == "image" else "content tokens"
            verb = "removed" if mode == "deletion" else "restored"
            ax.set_xlabel(f"share of {unit} {verb}")
            ax.set_title(
                f"{modality.capitalize()} · {titles[mode]}", fontsize=10, loc="left"
            )
            ax.grid(axis="y")
            ax.legend(loc="lower left" if mode == "deletion" else "lower right")
        axes[row, 0].set_ylabel("probability of the predicted class")
    axes[0, 0].set_ylim(0, 1)
    fig.supxlabel(
        "Mean over the test subset and the three fusion seeds. Removed image patches are "
        "replaced by a blurred copy of the image; removed tokens are hidden from attention.",
        fontsize=8,
        color=SECONDARY_INK,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    transformers_logging.set_verbosity_error()
    fc = cfg.faithfulness

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    test_rows = split_rows(metadata, cfg, device)["test"]
    subset = fixed_subset(test_rows, fc.subset_size, cfg.seed)
    texts = describe_all(metadata, include_diagnosis_method=False)
    inputs = {
        "images": load_images(metadata, cfg, device),
        **tokenize(texts, cfg.model.text_encoder, device),
    }
    batch = make_batch(inputs, subset, cfg, generator=None)
    masks = load_masks(metadata, subset, cfg, device).float()
    lesion = F.avg_pool2d(masks[:, None], fc.patch_size).flatten(1)
    references = {
        PRIMARY: blurred_reference(batch["images"], fc.blur_sigma),
        # zero after normalisation is the imagenet mean colour
        CHECK: torch.zeros_like(batch["images"]),
    }
    image_ids = metadata["image_id"].to_numpy()[subset.cpu().numpy()]
    truth = metadata["label"].to_numpy()[subset.cpu().numpy()]

    image_rows, agreement_rows, curve_rows = [], [], []
    for seed in run_seeds(cfg):
        model = load_fusion(cfg, seed, device)
        predicted, probabilities = predict(model, inputs, subset, cfg)
        target = torch.as_tensor(predicted, device=device)
        metrics, curves = evaluate_model(
            model, batch, target, lesion, references, cfg, seed
        )
        agreement = metrics.pop("agreement")
        for method, values in metrics.items():
            frame = pd.DataFrame(
                {
                    "seed": seed,
                    "method": method,
                    "image_id": image_ids,
                    "true_class": [CLASSES[c] for c in truth],
                    "predicted_class": [CLASSES[c] for c in predicted],
                    "probability": probabilities[np.arange(len(predicted)), predicted],
                    **{k: v.cpu().numpy() for k, v in values.items()},
                }
            )
            image_rows.append(frame)
            for mode, curve in curves[method].items():
                for fraction, value in zip(
                    fractions(fc).tolist(), curve.mean(dim=0).tolist()
                ):
                    curve_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "mode": mode,
                            "fraction": fraction,
                            "probability": value,
                        }
                    )
        for key, values in agreement.items():
            first, second, measure = key.split("~")
            agreement_rows.append(
                {
                    "seed": seed,
                    "pair": f"{first} vs {second}",
                    "measure": measure,
                    "value": float(np.nanmean(values.cpu().numpy())),
                }
            )
        print(f"seed {seed} done", flush=True)
        del model
        torch.cuda.empty_cache()

    per_image = pd.concat(image_rows, ignore_index=True)
    per_image = per_image.reindex(
        columns=[*per_image.columns[:6], *PER_IMAGE_METRICS, *CHECK_METRICS]
    )
    paired = pd.DataFrame(paired_against_control(per_image, fc.confidence_level))
    by_seed = (
        per_image.groupby(["seed", "method"], sort=False)[PER_IMAGE_METRICS]
        .mean()
        .reset_index()
        .merge(paired, on=["seed", "method"], how="left")
    )
    by_seed.insert(2, "images", fc.subset_size)
    summary = summarise(by_seed, ["method"])
    summary.insert(0, "modality", [modality_of(m) for m in summary.index])
    summary.insert(1, "seeds", len(run_seeds(cfg)))

    image_methods = per_image[per_image["method"].map(modality_of) == "image"]
    check_parts = []
    for reference in (PRIMARY, CHECK):
        end = suffix(reference)
        aucs = (
            image_methods.groupby(["seed", "method"], sort=False)[
                [f"deletion_auc{end}", f"insertion_auc{end}"]
            ]
            .mean()
            .rename(columns={f"{m}_auc{end}": f"{m}_auc" for m in MODES})
            .reset_index()
        )
        paired = pd.DataFrame(
            paired_against_control(image_methods, fc.confidence_level, reference)
        )
        merged = aucs.merge(paired, on=["seed", "method"], how="left")
        check_parts.append(merged.assign(reference=reference))
    reference_check = summarise(pd.concat(check_parts), ["method", "reference"])

    agreement = pd.DataFrame(agreement_rows)
    agreement_summary = (
        agreement.groupby(["pair", "measure"], sort=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    curve_frame = pd.DataFrame(curve_rows)
    curve_summary = (
        curve_frame.groupby(["method", "mode", "fraction"], sort=False)["probability"]
        .agg(["mean", "std"])
        .add_prefix("probability_")
        .reset_index()
    )

    reports_dir = cfg.paths.reports_dir
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    by_seed.to_csv(reports_dir / "faithfulness_runs.csv", index=False, **csv_options)
    summary.to_csv(reports_dir / "faithfulness.csv", **csv_options)
    reference_check.to_csv(
        reports_dir / "faithfulness_reference_check.csv", **csv_options
    )
    agreement_summary.to_csv(
        reports_dir / "method_agreement.csv", index=False, **csv_options
    )
    curve_summary.to_csv(
        reports_dir / "faithfulness_curves.csv", index=False, **csv_options
    )
    run_dir = cfg.paths.runs_dir / "faithfulness"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(run_dir / "per_image.csv", index=False, **csv_options)
    figure_file = reports_dir / "figures" / "faithfulness_curves.png"
    with matplotlib.rc_context(FIGURE_STYLE):
        plot_curves(curve_summary, summary).savefig(figure_file, dpi=150)

    start_mlflow(cfg, "explanations")
    with mlflow.start_run(run_name="faithfulness"):
        mlflow.log_params({"subset_size": fc.subset_size, "seeds": run_seeds(cfg)})
        for method in summary.index:
            for metric in ("deletion_auc_mean", "insertion_auc_mean", "gini_mean"):
                mlflow.log_metric(f"{method}_{metric}", summary.loc[method, metric])
        for name in ("faithfulness.csv", "method_agreement.csv"):
            mlflow.log_artifact(str(reports_dir / name))
        mlflow.log_artifact(str(figure_file))

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary.T.to_string())
        print(agreement_summary.to_string(index=False))
        print(reference_check.T.to_string())


if __name__ == "__main__":
    main()
