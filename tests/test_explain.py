import dataclasses
from pathlib import Path

import torch
from transformers import DistilBertConfig

from explainmed.config import load_config
from explainmed.data import CLASSES
from explainmed.explain import METHODS, MODALITY, attribute, content_tokens
from explainmed.model import FusionClassifier, ImageEncoder, TextEncoder

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_every_method_returns_scaled_maps_over_its_input_grid() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(cfg.explain, batch_size=2, ig_steps=4, ig_batch_size=8)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    text_config = DistilBertConfig(
        vocab_size=50, dim=16, n_layers=2, n_heads=2, hidden_dim=32
    )
    model = FusionClassifier(
        ImageEncoder(pretrained=False), TextEncoder(text_config), len(CLASSES), 8, 0.0
    )
    # the second text is padded, and its padding must get no attribution
    batch = {
        "images": torch.randn(3, 3, 64, 64),
        "input_ids": torch.tensor([[1, 7, 8, 9, 2], [1, 7, 8, 9, 2], [1, 7, 2, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]] * 2 + [[1, 1, 1, 0, 0]]),
        "special_tokens_mask": torch.tensor([[1, 0, 0, 0, 1]] * 2 + [[1, 0, 1, 1, 1]]),
    }
    target = torch.tensor([0, 3, 6])
    grids = {"image": (64, 64), "text": (5,)}

    for method in METHODS:
        maps = attribute(method, model, batch, target, cfg)
        assert maps.shape == (3, *grids[MODALITY[method]]), method
        assert maps.dtype == torch.float32
        assert torch.isfinite(maps).all()
        assert maps.min() >= 0.0
        peaks = maps.flatten(1).amax(dim=1)
        assert torch.all((peaks == 0) | torch.isclose(peaks, torch.tensor(1.0)))
        if MODALITY[method] == "text":
            assert torch.all(maps[content_tokens(batch) == 0] == 0)
