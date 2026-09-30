"""Adebayo et al. sanity checks: cascading model randomisation and label randomisation."""

from __future__ import annotations

import copy
from collections.abc import Iterator

import torch
from torch import nn

from explainmed.faithfulness import rank_correlation, top_overlap
from explainmed.model import FusionClassifier


def branch_layers(
    model: FusionClassifier, modality: str
) -> list[tuple[str, nn.Module]]:
    """Layers from the output back to the input of `modality`, in cascading order."""
    head = [layer for layer in reversed(model.fusion) if list(layer.parameters())]
    if modality == "image":
        blocks = list(reversed(model.image_encoder.features))
    elif modality == "text":
        transformer = model.text_encoder.transformer
        blocks = [*reversed(transformer.transformer.layer), transformer.embeddings]
    else:
        raise ValueError(f"unknown modality {modality!r}")
    names = {id(module): name for name, module in model.named_modules()}
    return [(names[id(layer)], layer) for layer in (*head, *blocks)]


def randomise(module: nn.Module) -> None:
    """Re-initialise every parameter and batch-norm statistic of `module`."""
    covered = set()
    for part in module.modules():
        if hasattr(part, "reset_parameters"):
            part.reset_parameters()
            covered.update(id(p) for p in part.parameters(recurse=False))
    missed = [name for name, p in module.named_parameters() if id(p) not in covered]
    if missed:
        raise ValueError(f"no initialiser resets parameters {missed}")


def cascading_randomisation(
    model: FusionClassifier, modality: str, seed: int
) -> Iterator[tuple[str, FusionClassifier]]:
    """A copy of the model with one more layer randomised at each step, from the top."""
    randomised = copy.deepcopy(model)
    for step, (name, layer) in enumerate(branch_layers(randomised, modality)):
        with torch.random.fork_rng():
            torch.manual_seed(seed + step)
            randomise(layer)
        yield name, randomised


def shuffled_labels(labels: torch.Tensor, seed: int) -> torch.Tensor:
    """Labels permuted across images: the class balance survives, the link to the image does not."""
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator, device=labels.device)
    return labels[order]


def map_similarity(
    original: torch.Tensor,
    other: torch.Tensor,
    valid: torch.Tensor,
    top_fraction: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Rank correlation and top-unit overlap per map pair; nan where the original is constant."""
    uninformative = rank_correlation(original, original, valid).isnan()
    collapsed = rank_correlation(other, other, valid).isnan() & ~uninformative
    # a map that randomisation flattens to a constant shares no ranking with the original
    correlation = rank_correlation(original, other, valid).nan_to_num(0.0)
    overlap = top_overlap(original, other, valid, top_fraction, generator)
    return {
        "rank_correlation": torch.where(uninformative, torch.nan, correlation),
        "top_overlap": torch.where(uninformative, torch.nan, overlap),
        "collapsed": collapsed.float(),
    }
