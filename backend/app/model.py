"""
AcousticSpace - Baseline Classifier (Week 2)

Late-fusion architecture:
  Branch A (CNN)  : mel-spectrogram -> conv stack -> embedding
  Branch B (MLP)   : [RT60, DRR, C50] -> small MLP -> embedding
  Fusion           : concat(embedding_A, embedding_B) -> FC -> 2-class logits

This is deliberately small/shallow for Week 2 — it exists to prove the
data -> tensor -> model -> prediction path works end-to-end. Week 3
swaps Branch A for a fine-tuned HuggingFace Audio Spectrogram
Transformer (AST) and keeps the same late-fusion idea with the RIR branch.
"""

import torch
import torch.nn as nn


class SpectrogramCNN(nn.Module):
    """Small conv stack over the mel-spectrogram, ending in a fixed-size embedding."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x128 -> 32x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64 -> 16x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 64x1x1
        )
        self.project = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, frames)
        h = self.features(x)
        h = h.flatten(1)
        return self.project(h)


class RIRFeatureMLP(nn.Module):
    """Small MLP over the 3 scalar RIR features (RT60, DRR, C50)."""

    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AcousticSpaceBaseline(nn.Module):
    """Late-fusion classifier: spectrogram CNN + RIR MLP -> 2-class logits."""

    def __init__(self, spec_embed_dim: int = 32, rir_embed_dim: int = 16):
        super().__init__()
        self.spec_branch = SpectrogramCNN(spec_embed_dim)
        self.rir_branch = RIRFeatureMLP(rir_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(spec_embed_dim + rir_embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),  # class 0 = genuine, class 1 = fake
        )

    def forward(self, mel: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
        spec_embed = self.spec_branch(mel)
        rir_embed = self.rir_branch(rir)
        fused = torch.cat([spec_embed, rir_embed], dim=1)
        return self.classifier(fused)
