"""Image encoder, text encoder, and the fusion classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoConfig, AutoModel, PretrainedConfig

from explainmed.config import ModelConfig


class ImageEncoder(nn.Module):
    """EfficientNet-B0 trunk mapping normalised images to a pooled feature vector."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = backbone.classifier[1].in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(images)).flatten(1)


class TextEncoder(nn.Module):
    """DistilBERT mapping token ids to the hidden state of the [CLS] token."""

    def __init__(self, source: str | PretrainedConfig) -> None:
        super().__init__()
        # a config builds untrained weights, for checkpoints and tests that skip the download
        self.transformer = (
            AutoModel.from_config(source)
            if isinstance(source, PretrainedConfig)
            else AutoModel.from_pretrained(source)
        )
        self.out_dim = self.transformer.config.dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return hidden.last_hidden_state[:, 0]


def _head(in_dim: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))


class ImageClassifier(nn.Module):
    def __init__(self, encoder: ImageEncoder, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = _head(encoder.out_dim, num_classes, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(batch["images"]))


class TextClassifier(nn.Module):
    def __init__(self, encoder: TextEncoder, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = _head(encoder.out_dim, num_classes, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encoder(batch["input_ids"], batch["attention_mask"]))


class FusionClassifier(nn.Module):
    """Late fusion: concatenated image and text features through a small MLP."""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_encoder.out_dim + text_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = self.image_encoder(batch["images"])
        text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        return self.fusion(torch.cat([image, text], dim=1))


def build_fusion(
    cfg: ModelConfig, num_classes: int, pretrained: bool
) -> FusionClassifier:
    """The configured fusion model, from pretrained encoders or untrained for loading."""
    text_source = (
        cfg.text_encoder if pretrained else AutoConfig.from_pretrained(cfg.text_encoder)
    )
    return FusionClassifier(
        ImageEncoder(pretrained),
        TextEncoder(text_source),
        num_classes,
        cfg.fusion_hidden_dim,
        cfg.dropout,
    )
