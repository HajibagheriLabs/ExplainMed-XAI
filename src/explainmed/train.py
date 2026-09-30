"""Training loop shared by the unimodal baselines and the fusion model."""

from __future__ import annotations

import random

import numpy as np
import torch

from explainmed.config import Config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn autotuning picks algorithms by timing, which makes reruns diverge
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(cfg: Config) -> torch.device:
    """Device named in the config, with the cpu thread cap applied."""
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for cuda but no cuda device is available")
    torch.set_num_threads(cfg.cpu_threads)
    return torch.device(cfg.device)
