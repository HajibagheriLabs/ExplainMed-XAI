import dataclasses
from pathlib import Path

import torch

from explainmed.config import load_config
from explainmed.data import CLASSES
from explainmed.model import ImageClassifier, ImageEncoder
from explainmed.train import fit

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_two_training_steps_write_a_loadable_checkpoint(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, epochs=1, batch_size=len(CLASSES))
    )
    torch.set_num_threads(cfg.cpu_threads)
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(
        0, 256, (14, 3, 32, 32), dtype=torch.uint8, generator=generator
    )
    labels = torch.arange(14) % len(CLASSES)
    model = ImageClassifier(ImageEncoder(pretrained=False), len(CLASSES), dropout=0.0)
    checkpoint = tmp_path / "best.pt"

    history = fit(
        model,
        {"images": images},
        labels,
        train_rows=torch.arange(14),
        val_rows=torch.arange(7),
        cfg=cfg,
        seed=0,
        checkpoint=checkpoint,
    )

    assert len(history) == 1
    assert checkpoint.exists()
    fresh = ImageClassifier(ImageEncoder(pretrained=False), len(CLASSES), dropout=0.0)
    fresh.load_state_dict(torch.load(checkpoint))
