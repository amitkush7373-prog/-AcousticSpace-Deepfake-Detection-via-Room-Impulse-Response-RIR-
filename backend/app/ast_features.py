"""
AcousticSpace - AST Feature Extraction (Week 3)

Wraps HuggingFace's ASTFeatureExtractor, which handles the specific
mel-spectrogram normalization the pretrained AST checkpoint expects
(different conventions from our own Librosa pipeline used for the
Week 2 CNN — the AST branch needs its own preprocessing to match how
it was pretrained on AudioSet).

NOTE: `from_pretrained` downloads weights from huggingface.co the first
time it runs. That download needs to happen on a machine with normal
internet access — it is not reachable from this sandbox, so this file
is architecturally validated here but the actual weight download and
fine-tuning run must happen on your own machine.
"""

from functools import lru_cache

import torch
from transformers import ASTFeatureExtractor

from app.audio_pipeline import load_audio

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.3593"


@lru_cache(maxsize=1)
def get_feature_extractor() -> ASTFeatureExtractor:
    return ASTFeatureExtractor.from_pretrained(MODEL_NAME)


def prepare_ast_input(path: str) -> torch.Tensor:
    """Returns input_values tensor of shape (1, max_length, num_mel_bins)."""
    y, sr = load_audio(path, sr=16000)
    fe = get_feature_extractor()
    inputs = fe(y, sampling_rate=sr, return_tensors="pt")
    return inputs["input_values"]
