import dataclasses
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from explainmed.config import load_config
from explainmed.faithfulness import (
    area_under_curve,
    gini,
    perturbation_curves,
    rank_correlation,
    top_overlap,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class TopLeftDetector(nn.Module):
    """Class 0 logit is ten times the mean of the top-left 2x2 patch; class 1 logit is 0."""

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        evidence = 10 * batch["images"][:, :, :2, :2].mean(dim=(1, 2, 3))
        return torch.stack([evidence, torch.zeros_like(evidence)], dim=1)


def test_deletion_and_insertion_match_a_hand_computed_example() -> None:
    cfg = dataclasses.replace(
        load_config(DEFAULT_CONFIG).faithfulness, steps=4, patch_size=2
    )
    batch = {"images": torch.ones(1, 1, 4, 4)}
    reference = torch.zeros_like(batch["images"])
    target = torch.tensor([0])
    generator = torch.Generator().manual_seed(0)
    faithful = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    backwards = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

    def curves(scores: torch.Tensor) -> dict[str, torch.Tensor]:
        return perturbation_curves(
            TopLeftDetector(), batch, scores, "image", target, reference, cfg, generator
        )

    # with the evidence present p = 1 / (1 + e^-10); without it p = 0.5
    p = 1 / (1 + math.exp(-10))
    expected_faithful = 0.25 * ((p + 0.5) / 2 + 3 * 0.5)
    expected_backwards = 0.25 * (3 * p + (p + 0.5) / 2)
    first, last = curves(faithful), curves(backwards)
    assert area_under_curve(first["deletion"], cfg).item() == pytest.approx(
        expected_faithful
    )
    assert area_under_curve(last["deletion"], cfg).item() == pytest.approx(
        expected_backwards
    )
    assert area_under_curve(first["insertion"], cfg).item() == pytest.approx(
        expected_backwards
    )


def test_sparsity_and_agreement_match_hand_computed_values() -> None:
    valid = torch.ones(2, 4, dtype=torch.bool)
    concentrated_and_uniform = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    assert gini(concentrated_and_uniform, valid).tolist() == pytest.approx([0.75, 0.0])

    # tied values share the mean rank: ranks (1, 2.5, 2.5, 4) against (1, 3, 2, 4)
    a = torch.tensor([[1.0, 2.0, 2.0, 3.0]])
    b = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    assert rank_correlation(a, b, valid[:1]).item() == pytest.approx(3 / math.sqrt(10))
    # the last unit is padding, so only the first three are compared
    padded = torch.tensor([[True, True, True, False]])
    assert rank_correlation(a, -a, padded).item() == pytest.approx(-1.0)

    generator = torch.Generator().manual_seed(0)
    first = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    second = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    overlap = top_overlap(first, second, valid[:1], 0.5, generator)
    assert overlap.item() == pytest.approx(1 / 3)
