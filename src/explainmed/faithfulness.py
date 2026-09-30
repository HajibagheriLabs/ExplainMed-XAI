"""Deletion and insertion AUC, sparsity, localisation, and agreement on patch or token units."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2.functional import gaussian_blur

from explainmed.config import FaithfulnessConfig
from explainmed.explain import Batch, content_tokens


def fixed_subset(rows: torch.Tensor, size: int, seed: int) -> torch.Tensor:
    """A seeded sample of `rows`; a smaller size gives a prefix of a larger one."""
    if size > len(rows):
        raise ValueError(f"cannot sample {size} of {len(rows)} rows")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed))
    return rows[order[:size].to(rows.device)]


def unit_scores(maps: torch.Tensor, modality: str, patch_size: int) -> torch.Tensor:
    """Attribution per unit, [N, U]: mean over each patch for images, per token for text."""
    if modality == "text":
        return maps
    if maps.shape[-1] % patch_size or maps.shape[-2] % patch_size:
        raise ValueError(
            f"image size {tuple(maps.shape[-2:])} is not a multiple of {patch_size}"
        )
    return F.avg_pool2d(maps[:, None], patch_size).flatten(1)


def valid_units(batch: Batch, modality: str, num_units: int) -> torch.Tensor:
    """Bool [N, U] of the units an attribution may rank; special tokens and padding are not."""
    if modality == "text":
        return content_tokens(batch).bool()
    return torch.ones(
        (len(batch["images"]), num_units),
        dtype=torch.bool,
        device=batch["images"].device,
    )


def random_attribution(valid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """The control: uniform noise over the valid units, carrying no information."""
    noise = torch.rand(valid.shape, generator=generator, device=valid.device)
    return noise * valid


def blurred_reference(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """What a removed image patch is replaced with: the same image, heavily blurred."""
    kernel = 2 * math.ceil(3 * sigma) + 1
    return gaussian_blur(images, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def descending_ranks(
    scores: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Rank of each unit, 0 for the highest score; invalid units rank last."""
    num_units = scores.shape[1]
    # ties are broken at random, or equal scores would be removed in raster order
    shuffle = torch.rand(scores.shape, generator=generator, device=scores.device)
    shuffle = shuffle.argsort(dim=1)
    shuffled = torch.where(valid, scores, -torch.inf).gather(1, shuffle)
    order = shuffle.gather(1, shuffled.argsort(dim=1, descending=True, stable=True))
    positions = torch.arange(num_units, device=scores.device).expand_as(order)
    return torch.empty_like(order).scatter_(1, order, positions)


@torch.no_grad()
def perturbation_curves(
    model: nn.Module,
    batch: Batch,
    scores: torch.Tensor,
    modality: str,
    target: torch.Tensor,
    reference: torch.Tensor | None,
    cfg: FaithfulnessConfig,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Target-class probability [N, steps + 1] as top-ranked units are deleted or inserted."""
    model.eval()
    valid = valid_units(batch, modality, scores.shape[1])
    ranks = descending_ranks(scores, valid, generator)
    available = valid.sum(dim=1, keepdim=True)
    curves = {"deletion": [], "insertion": []}
    for fraction in fractions(cfg).tolist():
        selected = ranks < torch.round(fraction * available)
        for mode, collected in curves.items():
            perturbed = _perturb(batch, selected, modality, reference, mode, cfg)
            collected.append(_target_probability(model, perturbed, target))
    return {mode: torch.stack(values, dim=1) for mode, values in curves.items()}


def fractions(cfg: FaithfulnessConfig) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, cfg.steps + 1)


def area_under_curve(curves: torch.Tensor, cfg: FaithfulnessConfig) -> torch.Tensor:
    """Trapezoidal area under each [N, steps + 1] curve over the fraction axis [0, 1]."""
    return torch.trapezoid(curves, fractions(cfg).to(curves.device), dim=1)


def gini(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Gini index of each row's valid scores: 0 when uniform, near 1 when concentrated."""
    # hurley and rickard's form; an all-zero row has no distribution and gives nan
    ordered = torch.where(valid, scores, torch.inf).sort(dim=1).values
    kept = torch.isfinite(ordered)
    ordered = torch.where(kept, ordered, 0.0)
    count = kept.sum(dim=1, keepdim=True)
    k = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    total = ordered.sum(dim=1, keepdim=True)
    share = ordered / torch.where(total > 0, total, 1.0)
    index = 1 - 2 * (share * (count - k + 0.5) / count * kept).sum(dim=1)
    return torch.where(total[:, 0] > 0, index, torch.nan)


def rank_correlation(
    a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Spearman correlation of each row pair over valid units, with ties given mean ranks."""
    ranks_a, ranks_b = _average_ranks(a, valid), _average_ranks(b, valid)
    count = valid.sum(dim=1, keepdim=True)
    centred_a = (ranks_a - (ranks_a * valid).sum(1, keepdim=True) / count) * valid
    centred_b = (ranks_b - (ranks_b * valid).sum(1, keepdim=True) / count) * valid
    covariance = (centred_a * centred_b).sum(dim=1)
    spread = (centred_a.square().sum(dim=1) * centred_b.square().sum(dim=1)).sqrt()
    # a constant map has no ranking, so its correlation is undefined
    return torch.where(
        spread > 0, covariance / torch.where(spread > 0, spread, 1.0), torch.nan
    )


def top_overlap(
    a: torch.Tensor,
    b: torch.Tensor,
    valid: torch.Tensor,
    fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Intersection over union of the top `fraction` of valid units of each map."""
    keep = torch.clamp(torch.round(fraction * valid.sum(dim=1, keepdim=True)), min=1)
    top_a = descending_ranks(a, valid, generator) < keep
    top_b = descending_ranks(b, valid, generator) < keep
    return (top_a & top_b).sum(dim=1) / (top_a | top_b).sum(dim=1)


def lesion_localisation(
    scores: torch.Tensor, lesion: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Attribution share in the lesion, whether the top unit is in it, and both chance levels."""
    total = scores.sum(dim=1)
    inside = (scores * lesion).sum(dim=1) / torch.where(total > 0, total, 1.0)
    hit = (lesion.gather(1, scores.argmax(dim=1, keepdim=True))[:, 0] >= 0.5).float()
    return {
        "mass_in_lesion": torch.where(total > 0, inside, torch.nan),
        "mass_chance": lesion.mean(dim=1),
        "pointing_game": torch.where(total > 0, hit, torch.nan),
        "pointing_chance": (lesion >= 0.5).float().mean(dim=1),
    }


def _perturb(
    batch: Batch,
    selected: torch.Tensor,
    modality: str,
    reference: torch.Tensor | None,
    mode: str,
    cfg: FaithfulnessConfig,
) -> Batch:
    hidden = selected if mode == "deletion" else ~selected
    if modality == "text":
        visible = batch["attention_mask"].bool() & (
            batch["special_tokens_mask"].bool() | ~hidden
        )
        return {**batch, "attention_mask": visible.long()}
    images = batch["images"]
    side = images.shape[-1] // cfg.patch_size
    grid = hidden.view(-1, 1, side, side)
    pixels = grid.repeat_interleave(cfg.patch_size, dim=2).repeat_interleave(
        cfg.patch_size, dim=3
    )
    return {**batch, "images": torch.where(pixels, reference, images)}


def _target_probability(
    model: nn.Module, batch: Batch, target: torch.Tensor
) -> torch.Tensor:
    device = target.device
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        logits = model(batch)
    return logits.float().softmax(dim=1).gather(1, target[:, None])[:, 0]


def _average_ranks(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # invalid entries sort last, so the valid ones hold ranks 1..n as in scipy's rankdata
    ordered, order = torch.where(valid, values, torch.inf).sort(dim=1)
    num_units = values.shape[1]
    starts = torch.ones_like(ordered, dtype=torch.bool)
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    group = starts.cumsum(dim=1) - 1
    position = torch.arange(1, num_units + 1, device=values.device, dtype=values.dtype)
    position = position.expand_as(ordered)
    first = torch.zeros_like(ordered).scatter_reduce(
        1, group, position, "amin", include_self=False
    )
    last = torch.zeros_like(ordered).scatter_reduce(
        1, group, position, "amax", include_self=False
    )
    mean_rank = ((first + last) / 2).gather(1, group)
    return torch.empty_like(values).scatter_(1, order, mean_rank)
