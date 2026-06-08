"""
Two-head CNN for ISAC target detection and size classification.

Architecture:
    backbone : 3 x (Conv2d-BN-ReLU-MaxPool2d)  ->  128 ch x 8 x 16
    pool     : AdaptiveAvgPool2d(1)             ->  128
    fc_shared: Linear(128 -> 64) + ReLU + Dropout
    head_presence : Linear(64 -> 1)   presence logit  (BCEWithLogitsLoss)
    head_size     : Linear(64 -> 4)   size logits     (CrossEntropyLoss)
                    classes: 0=absent, 1=small, 2=medium, 3=large
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class ISACDetectorCNN(nn.Module):
    """Two-head CNN for ISAC presence detection and size classification."""

    def __init__(self, num_size_classes: int = 4, dropout: float = 0.5):
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_block(1,   32),   # (B,  1, 64, 128) -> (B,  32, 32, 64)
            _conv_block(32,  64),   # (B, 32, 32,  64) -> (B,  64, 16, 32)
            _conv_block(64, 128),   # (B, 64, 16,  32) -> (B, 128,  8, 16)
        )
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.dropout     = nn.Dropout(dropout)
        self.fc_shared   = nn.Linear(128, 64)
        self.head_presence = nn.Linear(64, 1)
        self.head_size     = nn.Linear(64, num_size_classes)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 1, 64, 128)
        Returns presence_logit (B,) and size_logits (B, 4).
        """
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)
        feats = self.dropout(F.relu(self.fc_shared(feats)))
        return self.head_presence(feats).squeeze(-1), self.head_size(feats)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ISACDetectorCNN()
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randn(2, 1, 64, 128)
    p, s = model(x)
    print(f"Input:          {tuple(x.shape)}")
    print(f"Presence logit: {tuple(p.shape)}")
    print(f"Size logits:    {tuple(s.shape)}  argmax={s.argmax(dim=-1).numpy()}")