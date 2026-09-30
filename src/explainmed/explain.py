"""Grad-CAM, integrated gradients, and text attention rollout behind one attribution interface.

Every method takes the fusion model, a batch, and the class to explain, and returns
non-negative evidence for that class over its modality's input grid: [N, H, W] pixels for
image methods, [N, T] tokens for the text method. Each map is scaled so its maximum is 1; a map
with no positive evidence is all zeros. Special tokens and padding always get zero.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam

from explainmed.config import Config, ExplainConfig
from explainmed.data import CLASSES
from explainmed.model import FusionClassifier, build_fusion

Batch = dict[str, torch.Tensor]


def grad_cam(
    model: FusionClassifier, batch: Batch, target: torch.Tensor, cfg: ExplainConfig
) -> torch.Tensor:
    """Grad-CAM at the last convolutional block, upsampled to the input resolution."""
    images = batch["images"]
    layer = model.image_encoder.features[-1]
    cam = LayerGradCam(_image_forward(model), layer).attribute(
        images,
        target=target,
        additional_forward_args=(batch["input_ids"], batch["attention_mask"]),
        relu_attributions=True,
    )
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear")
    return _scale(cam[:, 0].detach())


def integrated_gradients(
    model: FusionClassifier, batch: Batch, target: torch.Tensor, cfg: ExplainConfig
) -> torch.Tensor:
    """Integrated gradients from a mean-colour image, summed over colour channels."""
    images = batch["images"]
    # zero after normalisation is the imagenet mean colour; a black baseline would give
    # dark pigment, the structure that matters most in melanoma, almost no attribution
    attributions = IntegratedGradients(_image_forward(model)).attribute(
        images,
        baselines=torch.zeros_like(images),
        target=target,
        additional_forward_args=(batch["input_ids"], batch["attention_mask"]),
        n_steps=cfg.ig_steps,
        internal_batch_size=cfg.ig_batch_size,
    )
    return _scale(attributions.sum(dim=1).clamp(min=0).detach())


@torch.no_grad()
def attention_rollout(
    model: FusionClassifier, batch: Batch, target: torch.Tensor, cfg: ExplainConfig
) -> torch.Tensor:
    """Attention rollout from the [CLS] token over the content tokens of the text."""
    # attention does not depend on the class being explained, so `target` is unused
    transformer = model.text_encoder.transformer
    # the default sdpa attention kernels do not return attention weights
    transformer.set_attn_implementation("eager")
    attentions = transformer(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_attentions=True,
    ).attentions
    length = batch["input_ids"].shape[1]
    identity = torch.eye(length, device=batch["input_ids"].device)
    rollout = identity.expand(len(batch["input_ids"]), length, length)
    for attention in attentions:
        # the identity term accounts for the residual connection around each layer
        mixed = 0.5 * attention.mean(dim=1) + 0.5 * identity
        rollout = (mixed / mixed.sum(dim=-1, keepdim=True)) @ rollout
    return _scale(rollout[:, 0] * content_tokens(batch))


METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "grad_cam": grad_cam,
    "integrated_gradients": integrated_gradients,
    "attention_rollout": attention_rollout,
}
MODALITY = {
    "grad_cam": "image",
    "integrated_gradients": "image",
    "attention_rollout": "text",
}


def attribute(
    method: str,
    model: FusionClassifier,
    batch: Batch,
    target: torch.Tensor,
    cfg: ExplainConfig,
) -> torch.Tensor:
    """Attributions of `method` for `target`, batched to bound GPU memory."""
    if method not in METHODS:
        raise ValueError(f"unknown attribution method {method!r}")
    model.eval()
    chunks = []
    for start in range(0, len(target), cfg.batch_size):
        part = {name: t[start : start + cfg.batch_size] for name, t in batch.items()}
        chunks.append(
            METHODS[method](model, part, target[start : start + cfg.batch_size], cfg)
        )
    return torch.cat(chunks)


def content_tokens(batch: Batch) -> torch.Tensor:
    """Float mask of the tokens a text attribution may assign weight to."""
    return (
        batch["attention_mask"].bool() & ~batch["special_tokens_mask"].bool()
    ).float()


def fusion_checkpoint(cfg: Config, seed: int) -> Path:
    return cfg.paths.runs_dir / "fusion" / "fusion" / f"seed{seed}" / "best.pt"


def load_fusion(cfg: Config, seed: int, device: torch.device) -> FusionClassifier:
    """The trained fusion model of one seed, in eval mode."""
    checkpoint = fusion_checkpoint(cfg, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} is missing; run `make train` first")
    model = build_fusion(cfg.model, len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


def _image_forward(model: FusionClassifier) -> Callable[..., torch.Tensor]:
    # captum perturbs the first argument and passes the text through unchanged
    def forward(
        images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return model(
            {
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    return forward


def _scale(maps: torch.Tensor) -> torch.Tensor:
    peak = maps.flatten(1).amax(dim=1).view(-1, *[1] * (maps.dim() - 1))
    # a map without positive evidence stays zero instead of dividing by zero
    return torch.where(peak > 0, maps / torch.where(peak > 0, peak, 1.0), 0.0)
