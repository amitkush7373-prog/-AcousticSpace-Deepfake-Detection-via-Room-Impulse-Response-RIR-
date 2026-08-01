"""
AcousticSpace - AST Late-Fusion Model (Week 3)

Same late-fusion idea as the Week 2 baseline, but the spectrogram branch
is now a pretrained Audio Spectrogram Transformer instead of a small
CNN — it already understands general audio structure from AudioSet
pretraining, so it should need less data to fine-tune well for the
genuine-vs-fake task than training a CNN from scratch.

  Branch A (AST)  : mel-spectrogram -> pretrained transformer -> pooled embedding
  Branch B (MLP)   : [RT60, DRR, C50] -> small MLP -> embedding  (same as Week 2)
  Fusion            : concat -> FC -> 2-class logits

The backbone is frozen by default (`freeze_backbone=True`) since our
current dataset is tiny — fine-tuning only the small fusion head avoids
overfitting a 90M-parameter transformer on a handful of clips. Once
real ASVspoof data is loaded, set freeze_backbone=False (or unfreeze
just the last few encoder layers) for a proper fine-tune.
"""

import torch
import torch.nn as nn
from transformers import ASTModel

from app.model import RIRFeatureMLP  # reused as-is from Week 2
from app.ast_features import MODEL_NAME


class AcousticSpaceAST(nn.Module):
    def __init__(self, rir_embed_dim: int = 16, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = ASTModel.from_pretrained(MODEL_NAME)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.rir_branch = RIRFeatureMLP(rir_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + rir_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # 0 = genuine, 1 = fake
        )

    def forward(self, input_values: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values=input_values)
        pooled = outputs.pooler_output          # (batch, hidden_size)
        rir_embed = self.rir_branch(rir)        # (batch, rir_embed_dim)
        fused = torch.cat([pooled, rir_embed], dim=1)
        return self.classifier(fused)
