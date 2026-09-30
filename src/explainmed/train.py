"""Training loop shared by the unimodal baselines and the fusion model."""

from __future__ import annotations

import dataclasses
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from explainmed.config import Config, TrainConfig
from explainmed.data import CLASSES, eval_transform, train_transform
from explainmed.evaluate import classification_metrics
from explainmed.model import ImageEncoder, TextEncoder


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


def parameter_groups(model: nn.Module, cfg: TrainConfig) -> list[dict]:
    """Optimizer groups giving each pretrained encoder its own learning rate."""
    encoder_lrs = {ImageEncoder: cfg.image_encoder_lr, TextEncoder: cfg.text_encoder_lr}
    groups, claimed = [], set()
    for module in model.modules():
        if type(module) in encoder_lrs:
            params = list(module.parameters())
            groups.append({"params": params, "lr": encoder_lrs[type(module)]})
            claimed.update(id(p) for p in params)
    rest = [p for p in model.parameters() if id(p) not in claimed]
    return [*groups, {"params": rest, "lr": cfg.head_lr}]


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, so every class contributes equally to the loss."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    if (counts == 0).any():
        raise ValueError("every class needs training examples to be weighted")
    return counts.sum() / (num_classes * counts)


def fit(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    train_rows: torch.Tensor,
    val_rows: torch.Tensor,
    cfg: Config,
    seed: int,
    checkpoint: Path,
) -> list[dict[str, float]]:
    """Train and keep the weights of the epoch with the best validation macro-F1."""
    device = labels.device
    use_amp = device.type == "cuda"
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.to(device)

    tc = cfg.train
    optimizer = torch.optim.AdamW(
        parameter_groups(model, tc), weight_decay=tc.weight_decay
    )
    total_steps = tc.epochs * math.ceil(len(train_rows) / tc.batch_size)
    warmup_steps = max(1, round(tc.warmup_fraction * total_steps))

    def warmup_then_cosine(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_then_cosine)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    weight = (
        class_weights(labels[train_rows], len(CLASSES)) if tc.balanced_loss else None
    )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        losses = []
        order = torch.randperm(len(train_rows), generator=generator, device=device)
        for rows in train_rows[order].split(tc.batch_size):
            batch = _batch(inputs, rows, cfg, generator)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(batch), labels[rows], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.detach())

        predicted, _ = predict(model, inputs, val_rows, cfg)
        metrics = classification_metrics(
            labels[val_rows].cpu().numpy(), predicted, CLASSES
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": torch.stack(losses).mean().item(),
                "val_macro_f1": metrics["macro_f1"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save(model.state_dict(), checkpoint)
    return history


@torch.no_grad()
def predict(
    model: nn.Module, inputs: dict[str, torch.Tensor], rows: torch.Tensor, cfg: Config
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted class and class probabilities for each row."""
    model.eval()
    use_amp = rows.device.type == "cuda"
    probabilities = []
    for chunk in rows.split(cfg.train.batch_size):
        with torch.autocast(rows.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(_batch(inputs, chunk, cfg, generator=None))
        probabilities.append(logits.float().softmax(dim=1))
    probabilities = torch.cat(probabilities).cpu().numpy()
    return probabilities.argmax(axis=1), probabilities


def _batch(
    inputs: dict[str, torch.Tensor],
    rows: torch.Tensor,
    cfg: Config,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    # a generator means training, so images get augmented
    batch = {name: tensor[rows] for name, tensor in inputs.items()}
    if "images" in batch:
        batch["images"] = (
            eval_transform(batch["images"], cfg.data)
            if generator is None
            else train_transform(batch["images"], cfg.data, generator)
        )
    return batch


def flat_params(cfg: Config) -> dict[str, str]:
    """The config as dotted keys, for experiment tracking."""
    flat = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}{key}.", inner)
        else:
            flat[prefix.rstrip(".")] = str(value)

    walk("", dataclasses.asdict(cfg))
    return flat


def start_mlflow(cfg: Config, experiment: str) -> None:
    """Point MLflow at the local store under the configured directory."""
    # imported here because mlflow takes seconds to import and only scripts log runs
    import mlflow

    root = cfg.paths.mlflow_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # mlflow 3 refuses the plain file store, so the local store is sqlite in the same folder
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment, artifact_location=(root / "artifacts").as_uri()
        )
    mlflow.set_experiment(experiment)
