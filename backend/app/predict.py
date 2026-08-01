"""
AcousticSpace - Inference (Week 2 bonus)

Loads the trained checkpoint once and exposes a simple predict() function
so the API can serve baseline predictions. Full "Results UI" wiring is a
Week 3 deliverable per the project plan, but the model + endpoint are
ready ahead of time since it was low-effort to add here.
"""

from pathlib import Path
from functools import lru_cache

import torch
import torch.nn.functional as F

from app.model import AcousticSpaceBaseline
from app.features import prepare_model_inputs

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "baseline_cnn.pt"
LABELS = {0: "genuine", 1: "fake"}


@lru_cache(maxsize=1)
def _load_model() -> AcousticSpaceBaseline | None:
    if not CHECKPOINT_PATH.exists():
        return None
    model = AcousticSpaceBaseline()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(path: str) -> dict | None:
    """Returns {"label": "genuine"|"fake", "confidence": float} or None if no checkpoint yet."""
    model = _load_model()
    if model is None:
        return None

    mel, rir = prepare_model_inputs(path)
    mel_t = torch.from_numpy(mel).unsqueeze(0)   # add batch dim
    rir_t = torch.from_numpy(rir).unsqueeze(0)

    with torch.no_grad():
        logits = model(mel_t, rir_t)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(probs.argmax().item())
    return {
        "label": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx].item()), 4),
        "probabilities": {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }
