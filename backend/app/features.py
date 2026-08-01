"""
AcousticSpace - Model Input Preparation (Week 2)

The CNN needs fixed-size tensors, but audio clips vary in length, so we
pad/truncate the mel-spectrogram to a fixed number of frames. RIR scalar
features (RT60, DRR, C50) are normalized into a small feature vector for
the late-fusion MLP branch.
"""

from __future__ import annotations

import numpy as np

from app.audio_pipeline import (
    load_audio,
    extract_mel_spectrogram,
    extract_rir_features,
)

N_MELS = 64
FIXED_FRAMES = 128  # ~4 seconds at default librosa hop settings for 16kHz audio

# Rough normalization ranges so RIR features sit roughly in [-1, 1] for the MLP
RT60_SCALE = 2.0     # seconds
DRR_SCALE = 30.0     # dB
C50_SCALE = 30.0     # dB


def _pad_or_truncate(spec: np.ndarray, fixed_frames: int = FIXED_FRAMES) -> np.ndarray:
    n_mels, n_frames = spec.shape
    if n_frames == fixed_frames:
        return spec
    if n_frames > fixed_frames:
        # center crop
        start = (n_frames - fixed_frames) // 2
        return spec[:, start:start + fixed_frames]
    pad_width = fixed_frames - n_frames
    left = pad_width // 2
    right = pad_width - left
    # pad with the spectrogram's own floor value rather than zero, so the
    # padded region doesn't look like an artificial cliff to the CNN
    floor = spec.min()
    return np.pad(spec, ((0, 0), (left, right)), mode="constant", constant_values=floor)


def prepare_rir_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """Standalone RIR feature vector, reusable by non-CNN models (e.g. the AST branch)."""
    rir = extract_rir_features(y, sr)
    return np.array(
        [
            np.clip(rir.rt60_seconds / RT60_SCALE, 0.0, 2.0),
            np.clip(rir.drr_db / DRR_SCALE, -2.0, 2.0),
            np.clip(rir.c50_db / C50_SCALE, -2.0, 2.0),
        ],
        dtype=np.float32,
    )


def prepare_model_inputs(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mel_tensor: shape (1, N_MELS, FIXED_FRAMES), float32, roughly in [-1, 1]
      rir_vector: shape (3,), float32 -> [rt60_norm, drr_norm, c50_norm]
    """
    y, sr = load_audio(path)

    mel = extract_mel_spectrogram(y, sr, n_mels=N_MELS)  # in dB, roughly [-80, 0]
    mel = _pad_or_truncate(mel, FIXED_FRAMES)
    mel_norm = (mel + 40.0) / 40.0  # dB -> roughly [-1, 1]
    mel_tensor = mel_norm[np.newaxis, :, :].astype(np.float32)

    rir = extract_rir_features(y, sr)
    rir_vector = np.array(
        [
            np.clip(rir.rt60_seconds / RT60_SCALE, 0.0, 2.0),
            np.clip(rir.drr_db / DRR_SCALE, -2.0, 2.0),
            np.clip(rir.c50_db / C50_SCALE, -2.0, 2.0),
        ],
        dtype=np.float32,
    )

    return mel_tensor, rir_vector
