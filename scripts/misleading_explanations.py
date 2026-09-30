"""Find confidently wrong predictions whose explanations still look plausible."""

import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score
from transformers import logging as transformers_logging

from explainmed.config import Config, load_config
from explainmed.data import (
    CLASSES,
    MALIGNANT,
    load_images,
    load_masks,
    load_metadata,
    split_rows,
)
from explainmed.explain import attribute, load_fusion
from explainmed.faithfulness import lesion_localisation, rank_correlation, unit_scores
from explainmed.style import FIGURE_STYLE, SECONDARY_INK, SEQUENTIAL, SERIES
from explainmed.text import describe_all, tokenize
from explainmed.train import make_batch, predict, resolve_device, run_seeds

IMAGE_METHODS = ("grad_cam", "integrated_gradients")
# each signal is oriented so that a higher score should mean a likelier error
SIGNALS = {
    "model confidence": lambda frame: 1 - frame["probability"],
    "Grad-CAM mass in lesion": lambda frame: 1 - frame["grad_cam_mass_in_lesion"],
    "Grad-CAM peak in lesion": lambda frame: 1 - frame["grad_cam_pointing_game"],
    "integrated gradients mass in lesion": lambda frame: (
        1 - frame["integrated_gradients_mass_in_lesion"]
    ),
}


def explain_test_split(
    cfg: Config,
    seed: int,
    inputs: dict[str, torch.Tensor],
    test_rows: torch.Tensor,
    truth: np.ndarray,
    lesion: torch.Tensor,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    """Per test image: prediction, confidence, and how well each map sits on the lesion."""
    device = test_rows.device
    model = load_fusion(cfg, seed, device)
    predicted, probabilities = predict(model, inputs, test_rows, cfg)
    target = torch.as_tensor(predicted, device=device)
    true_target = torch.as_tensor(truth, device=device)
    batch = make_batch(inputs, test_rows, cfg, generator=None)
    patch = cfg.faithfulness.patch_size
    maps = {m: attribute(m, model, batch, target, cfg.explain) for m in IMAGE_METHODS}
    maps["grad_cam_true_class"] = attribute(
        "grad_cam", model, batch, true_target, cfg.explain
    )
    columns = {
        "seed": seed,
        "true_class": [CLASSES[c] for c in truth],
        "predicted_class": [CLASSES[c] for c in predicted],
        "probability": probabilities[np.arange(len(predicted)), predicted],
        "correct": predicted == truth,
    }
    for method in IMAGE_METHODS:
        units = unit_scores(maps[method], "image", patch)
        for name, values in lesion_localisation(units, lesion).items():
            columns[f"{method}_{name}"] = values.cpu().numpy()
    predicted_units = unit_scores(maps["grad_cam"], "image", patch)
    true_units = unit_scores(maps["grad_cam_true_class"], "image", patch)
    valid = torch.ones_like(predicted_units, dtype=torch.bool)
    # how much the map changes when asked about the right class instead of the chosen one
    columns["grad_cam_predicted_vs_true_class"] = (
        rank_correlation(predicted_units, true_units, valid).cpu().numpy()
    )
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(columns), maps


def flag_misleading(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mark confidently wrong predictions whose Grad-CAM map looks as good as a correct one."""
    typical = frame.loc[frame["correct"], "grad_cam_mass_in_lesion"].median()
    confident = frame["probability"] >= cfg.misleading.confidence_threshold
    wrong = ~frame["correct"]
    plausible = (frame["grad_cam_pointing_game"] == 1) & (
        frame["grad_cam_mass_in_lesion"] >= typical
    )
    missed_malignancy = frame["true_class"].isin(MALIGNANT) & ~frame[
        "predicted_class"
    ].isin(MALIGNANT)
    return frame.assign(
        confidently_wrong=confident & wrong,
        plausible_explanation=plausible,
        missed_malignancy=missed_malignancy & wrong,
        correct_median_mass_in_lesion=typical,
    )


def error_signals(frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    """How well each signal separates wrong from correct predictions, as an AUROC."""
    rows = []
    for seed, group in frame.groupby("seed"):
        for signal, score in SIGNALS.items():
            values = score(group)
            usable = values.notna()
            rows.append(
                {
                    "seed": seed,
                    "signal": signal,
                    "auroc_for_errors": roc_auc_score(
                        ~group.loc[usable, "correct"], values[usable]
                    ),
                    "images": int(usable.sum()),
                }
            )
    return rows


def counts(frame: pd.DataFrame) -> list[dict[str, float | int]]:
    rows = []
    for seed, group in frame.groupby("seed"):
        wrong = ~group["correct"]
        misleading = group["confidently_wrong"] & group["plausible_explanation"]
        rows.append(
            {
                "seed": seed,
                "test_images": len(group),
                "wrong": int(wrong.sum()),
                "confidently_wrong": int(group["confidently_wrong"].sum()),
                "confidently_wrong_with_plausible_explanation": int(misleading.sum()),
                "of_which_missed_malignancy": int(
                    (misleading & group["missed_malignancy"]).sum()
                ),
                "missed_malignancy": int(group["missed_malignancy"].sum()),
                "missed_malignancy_confident": int(
                    (group["missed_malignancy"] & group["confidently_wrong"]).sum()
                ),
                "correct_median_mass_in_lesion": group[
                    "correct_median_mass_in_lesion"
                ].iloc[0],
                "plausible_share_correct": group.loc[
                    ~wrong, "plausible_explanation"
                ].mean(),
                "plausible_share_wrong": group.loc[
                    wrong, "plausible_explanation"
                ].mean(),
                "plausible_share_confidently_wrong": group.loc[
                    group["confidently_wrong"], "plausible_explanation"
                ].mean(),
                "grad_cam_predicted_vs_true_class_when_wrong": group.loc[
                    wrong, "grad_cam_predicted_vs_true_class"
                ].mean(),
            }
        )
    return rows


def pick_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    """The most confident misleading cases, missed malignancies first, one per true class."""
    candidates = frame[frame["confidently_wrong"] & frame["plausible_explanation"]]
    candidates = candidates.sort_values(
        ["missed_malignancy", "probability"], ascending=[False, False]
    )
    return candidates.groupby("true_class", sort=False).head(1).head(count)


def plot_cases(
    cases: pd.DataFrame,
    images: np.ndarray,
    maps: dict[str, np.ndarray],
    masks: np.ndarray,
) -> Figure:
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    columns = {
        "grad_cam": "Grad-CAM, predicted class",
        "grad_cam_true_class": "Grad-CAM, true class",
        "integrated_gradients": "Integrated gradients, predicted class",
    }
    fig = Figure(figsize=(10.5, 2.75 * len(cases) + 0.6), layout="constrained")
    axes = fig.subplots(len(cases), len(columns) + 1, squeeze=False)
    for row, case in enumerate(cases.itertuples()):
        axes[row, 0].imshow(images[row])
        axes[row, 0].set_title(
            f"true {case.true_class}, predicted {case.predicted_class}\n"
            f"p = {case.probability:.2f} · {case.image_id}",
            fontsize=9,
            fontweight="normal",
        )
        for column, method in enumerate(columns, start=1):
            heat = maps[method][row]
            top = max(float(np.quantile(heat, 0.99)), 1e-12)
            strength = np.clip(heat / top, 0.0, 1.0)
            rgba = colormap(strength)
            rgba[..., 3] = 0.85 * strength
            axes[row, column].imshow(images[row])
            axes[row, column].imshow(rgba)
        for ax in axes[row]:
            ax.contour(masks[row], levels=[0.5], colors=SERIES[1], linewidths=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
        axes[row, 1].set_xlabel(
            f"{case.grad_cam_mass_in_lesion:.0%} of the map inside the lesion",
            fontsize=8,
            color=SECONDARY_INK,
        )
    for column, heading in enumerate(["Image and lesion outline", *columns.values()]):
        axes[0, column].annotate(
            heading,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 38),
            textcoords="offset points",
            fontsize=10,
            fontweight="semibold",
        )
    fig.supxlabel(
        "Confidently wrong test predictions whose Grad-CAM peak lies inside the lesion "
        "and whose share of attribution inside it is at least that of the median correct "
        "prediction.\nThe orange line is the lesion segmentation; maps saturate at their "
        "99% quantile for display.",
        fontsize=8,
        color=SECONDARY_INK,
    )
    return fig


def plot_signals(frame: pd.DataFrame, signals: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10.5, 3.8), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    panels = [
        ("probability", "model confidence", "probability of the predicted class"),
        (
            "grad_cam_mass_in_lesion",
            "Grad-CAM mass in lesion",
            "share inside the lesion",
        ),
        (
            "integrated_gradients_mass_in_lesion",
            "integrated gradients mass in lesion",
            "share inside the lesion",
        ),
    ]
    auroc = signals.groupby("signal")["auroc_for_errors"].mean()
    for ax, (column, signal, label) in zip(axes, panels):
        for correct, colour, name in (
            (True, SERIES[0], "correct"),
            (False, SERIES[1], "wrong"),
        ):
            values = np.sort(frame.loc[frame["correct"] == correct, column].dropna())
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                color=colour,
                linewidth=1.8,
                label=f"{name} predictions",
            )
        ax.set_title(
            f"{signal[0].upper()}{signal[1:]}\nerror AUROC {auroc[signal]:.2f}",
            loc="left",
        )
        ax.set_xlabel(label)
        ax.set_xlim(0, 1)
        ax.grid(axis="y")
    axes[0].set_ylabel("cumulative share of predictions")
    axes[0].legend(loc="upper left")
    fig.supxlabel(
        "Test split, three fusion seeds pooled. AUROC for errors is 0.5 when a signal "
        "cannot tell wrong predictions from correct ones.",
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

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    test_rows = split_rows(metadata, cfg, device)["test"]
    texts = describe_all(metadata, include_diagnosis_method=False)
    inputs = {
        "images": load_images(metadata, cfg, device),
        **tokenize(texts, cfg.model.text_encoder, device),
    }
    truth = metadata["label"].to_numpy()[test_rows.cpu().numpy()]
    image_ids = metadata["image_id"].to_numpy()[test_rows.cpu().numpy()]
    masks = load_masks(metadata, test_rows, cfg, device)
    lesion = F.avg_pool2d(masks.float()[:, None], cfg.faithfulness.patch_size).flatten(
        1
    )

    frames, figure_maps = [], None
    for seed in run_seeds(cfg):
        frame, maps = explain_test_split(cfg, seed, inputs, test_rows, truth, lesion)
        frames.append(flag_misleading(frame.assign(image_id=image_ids), cfg))
        if seed == cfg.seed:
            figure_maps = {name: values.cpu() for name, values in maps.items()}
        print(f"seed {seed} done", flush=True)
    per_image = pd.concat(frames, ignore_index=True)
    signals = pd.DataFrame(error_signals(per_image))
    signal_summary = (
        signals.groupby("signal", sort=False)["auroc_for_errors"]
        .agg(["mean", "std"])
        .add_prefix("auroc_for_errors_")
        .reset_index()
    )
    case_counts = pd.DataFrame(counts(per_image))
    count_summary = case_counts.drop(columns="seed").agg(["mean", "std"]).T

    primary = per_image[per_image["seed"] == cfg.seed].reset_index(drop=True)
    misleading = primary[
        primary["confidently_wrong"] & primary["plausible_explanation"]
    ]
    cases = pick_cases(primary, cfg.misleading.cases)
    positions = cases.index.to_list()
    case_images = (
        inputs["images"][test_rows[positions]].permute(0, 2, 3, 1).cpu().numpy()
    )

    reports_dir = cfg.paths.reports_dir
    csv_options = {"float_format": "%.4f", "lineterminator": "\n"}
    case_columns = [
        "seed",
        "image_id",
        "true_class",
        "predicted_class",
        "probability",
        "missed_malignancy",
        "grad_cam_mass_in_lesion",
        "grad_cam_mass_chance",
        "integrated_gradients_mass_in_lesion",
        "grad_cam_predicted_vs_true_class",
    ]
    misleading[case_columns].sort_values("probability", ascending=False).to_csv(
        reports_dir / "misleading_cases.csv", index=False, **csv_options
    )
    signals.to_csv(reports_dir / "error_signals_runs.csv", index=False, **csv_options)
    signal_summary.to_csv(reports_dir / "error_signals.csv", index=False, **csv_options)
    case_counts.to_csv(
        reports_dir / "misleading_counts.csv", index=False, **csv_options
    )
    run_dir = cfg.paths.runs_dir / "misleading"
    run_dir.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(run_dir / "per_image.csv", index=False, **csv_options)
    figures = reports_dir / "figures"
    with matplotlib.rc_context(FIGURE_STYLE):
        plot_cases(
            cases,
            case_images,
            {name: values[positions].numpy() for name, values in figure_maps.items()},
            masks[positions].cpu().numpy(),
        ).savefig(figures / "misleading_explanations.png", dpi=150)
        plot_signals(per_image, signals).savefig(figures / "error_signals.png", dpi=150)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(signal_summary.to_string(index=False))
        print(count_summary.to_string())
        print(cases[case_columns].to_string(index=False))


if __name__ == "__main__":
    main()
