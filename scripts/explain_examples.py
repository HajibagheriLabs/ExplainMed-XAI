"""Render the three attribution methods for a fixed, seeded sample of test cases."""

import argparse

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from transformers import AutoTokenizer
from transformers import logging as transformers_logging

from explainmed.config import load_config
from explainmed.data import CLASSES, load_images, load_masks, load_metadata, split_rows
from explainmed.explain import METHODS, attribute, content_tokens, load_fusion
from explainmed.style import FIGURE_STYLE, SECONDARY_INK, SEQUENTIAL, SERIES
from explainmed.text import describe_all, tokenize
from explainmed.train import make_batch, predict, resolve_device, seed_everything

TITLES = {
    "grad_cam": "Grad-CAM",
    "integrated_gradients": "Integrated gradients",
    "attention_rollout": "Attention rollout (text)",
}
# integrated gradients concentrates on a few pixels, so its display saturates at this quantile
DISPLAY_QUANTILE = 0.99


def pick_examples(
    truth: np.ndarray, predicted: np.ndarray, per_group: int, seed: int
) -> dict[str, np.ndarray]:
    """Positions of correct and incorrect cases, each with distinct true classes."""
    order = np.random.default_rng(seed).permutation(len(truth))
    picked = {}
    for group, correct in (("correct", True), ("incorrect", False)):
        chosen, seen = [], set()
        for position in order:
            is_correct = truth[position] == predicted[position]
            if is_correct == correct and truth[position] not in seen:
                chosen.append(position)
                seen.add(truth[position])
            if len(chosen) == per_group:
                break
        if len(chosen) < per_group:
            raise ValueError(f"fewer than {per_group} {group} test predictions")
        picked[group] = np.array(chosen)
    return picked


def overlay(ax, image: np.ndarray, heat: np.ndarray, mask: np.ndarray) -> None:
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    top = max(float(np.quantile(heat, DISPLAY_QUANTILE)), 1e-12)
    strength = np.clip(heat / top, 0.0, 1.0)
    rgba = colormap(strength)
    rgba[..., 3] = 0.85 * strength
    ax.imshow(image)
    ax.imshow(rgba)
    ax.contour(mask, levels=[0.5], colors=SERIES[1], linewidths=0.9)


def merge_word_pieces(
    tokens: list[str], weights: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Whole words for display, each weighted by its strongest word piece."""
    words, merged = [], []
    for token, weight in zip(tokens, weights):
        if token.startswith("##") and words:
            words[-1] += token[2:]
            merged[-1] = max(merged[-1], weight)
        else:
            words.append(token)
            merged.append(weight)
    return words, np.array(merged)


def token_bars(ax, tokens: list[str], weights: np.ndarray) -> None:
    positions = np.arange(len(tokens))
    ax.barh(positions, weights, color=SERIES[0], height=0.7)
    ax.set_yticks(positions, tokens, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")


def render(cases: pd.DataFrame, images, maps, masks, tokens, title: str) -> Figure:
    fig = Figure(figsize=(11, 2.7 * len(cases) + 0.6), layout="constrained")
    axes = fig.subplots(len(cases), 4, width_ratios=[1, 1, 1, 1.25], squeeze=False)
    for row, case in enumerate(cases.itertuples()):
        image = images[row]
        axes[row, 0].imshow(image)
        axes[row, 0].contour(masks[row], levels=[0.5], colors=SERIES[1], linewidths=0.9)
        axes[row, 0].set_title(
            f"true {case.true_class}, predicted {case.predicted_class}\n"
            f"p = {case.probability:.2f} · {case.image_id}",
            fontsize=9,
            fontweight="normal",
        )
        for column, method in enumerate(("grad_cam", "integrated_gradients"), start=1):
            overlay(axes[row, column], image, maps[method][row], masks[row])
        token_bars(axes[row, 3], tokens[row], maps["attention_rollout"][row])
        for ax in axes[row, :3]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    for column, heading in enumerate(["Image and lesion outline", *TITLES.values()]):
        axes[0, column].annotate(
            heading,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 38),
            textcoords="offset points",
            fontsize=11,
            fontweight="semibold",
        )
    fig.suptitle(title, x=0.0, ha="left", fontsize=12, fontweight="semibold")
    fig.supxlabel(
        "Maps explain the predicted class and are scaled to their own maximum. Image maps "
        f"saturate at their {DISPLAY_QUANTILE:.0%} quantile for display, and the orange "
        "line is the lesion segmentation.\nA word's text weight is that of its strongest "
        "word piece.",
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
    # deterministic cudnn kernels make gradient attributions reproducible to the bit
    seed_everything(cfg.seed)
    transformers_logging.set_verbosity_error()

    metadata = load_metadata(cfg.paths.metadata_csv, cfg.paths.images_dir)
    test_rows = split_rows(metadata, cfg, device)["test"]
    texts = describe_all(metadata, include_diagnosis_method=False)
    images = load_images(metadata, cfg, device)
    inputs = {"images": images, **tokenize(texts, cfg.model.text_encoder, device)}

    model = load_fusion(cfg, cfg.seed, device)
    predicted, probabilities = predict(model, inputs, test_rows, cfg)
    truth = metadata["label"].to_numpy()[test_rows.cpu().numpy()]
    picked = pick_examples(truth, predicted, cfg.explain.examples_per_group, cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)

    records, panels = [], {}
    for group, positions in picked.items():
        rows = test_rows[torch.as_tensor(positions, device=device)]
        batch = make_batch(inputs, rows, cfg, generator=None)
        target = torch.as_tensor(predicted[positions], device=device)
        maps = {
            method: attribute(method, model, batch, target, cfg.explain).cpu().numpy()
            for method in METHODS
        }
        keep = content_tokens(batch).bool().cpu().numpy()
        ids = batch["input_ids"].cpu().numpy()
        words = [
            merge_word_pieces(
                tokenizer.convert_ids_to_tokens(ids[i][keep[i]].tolist()),
                maps["attention_rollout"][i][keep[i]],
            )
            for i in range(len(positions))
        ]
        tokens = [token for token, _ in words]
        maps["attention_rollout"] = [weight for _, weight in words]
        cases = pd.DataFrame(
            {
                "group": group,
                "image_id": metadata["image_id"].to_numpy()[rows.cpu().numpy()],
                "true_class": [CLASSES[c] for c in truth[positions]],
                "predicted_class": [CLASSES[c] for c in predicted[positions]],
                "probability": probabilities[positions, predicted[positions]],
            }
        )
        records.append(cases)
        panels[group] = (
            cases,
            images[rows].permute(0, 2, 3, 1).cpu().numpy(),
            maps,
            load_masks(metadata, rows, cfg, device).cpu().numpy(),
            tokens,
        )

    reports_dir = cfg.paths.reports_dir
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(FIGURE_STYLE):
        for group, panel in panels.items():
            title = f"{group.capitalize()} test predictions of the fusion model"
            fig = render(*panel, f"{title} (seed {cfg.seed})")
            fig.savefig(reports_dir / "figures" / f"explanations_{group}.png", dpi=150)
    pd.concat(records).to_csv(
        reports_dir / "explanation_examples.csv",
        index=False,
        float_format="%.4f",
        lineterminator="\n",
    )
    print(pd.concat(records).to_string(index=False))


if __name__ == "__main__":
    main()
