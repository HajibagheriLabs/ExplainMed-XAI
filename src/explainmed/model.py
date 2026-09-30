"""Image encoder, text encoder, and the fusion classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from transformers import AutoModel


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

    def __init__(self, pretrained_name: str) -> None:
        super().__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_name)
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
