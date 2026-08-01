"""
AcousticSpace - Audio Processing Pipeline
Week 1 deliverable: Librosa pipeline to extract spectrograms and RIR
(Room Impulse Response) features from an uploaded audio clip.

Core idea: a deepfake voice, even if vocally convincing, is rarely
synthesized WITH a physically consistent room acoustic signature.
So instead of (or in addition to) vocal-artifact detection, we
characterize the *environment* the voice claims to be recorded in.

Features extracted here:
  1. Mel-spectrogram + MFCCs       -> general acoustic fingerprint
  2. RT60 (reverberation time)     -> via Schroeder backward integration
  3. DRR  (direct-to-reverberant ratio)
  4. C50  (clarity / speech definition, 50ms early/late energy ratio)
  5. Spectral decay envelope       -> raw curve for the classifier / UI

These are estimated "blindly" from the recording itself (we don't have
a clean impulse response — we approximate one from the decay tails of
the signal), which is the same real-world constraint an analyst faces.
"""

from __future__ import annotations

import numpy as np
import librosa
from dataclasses import dataclass, asdict
from typing import Optional


SAMPLE_RATE = 16000  # standard rate for speech/deepfake datasets (ASVspoof uses 16k)


@dataclass
class RIRFeatures:
    rt60_seconds: float
    drr_db: float
    c50_db: float
    decay_envelope_db: list  # downsampled Schroeder decay curve, for plotting in the UI


@dataclass
class ExtractionResult:
    duration_seconds: float
    sample_rate: int
    mel_spectrogram: list       # shape (n_mels, n_frames), for waveform/spectrogram viz
    mfcc: list                  # shape (n_mfcc, n_frames)
    rir: RIRFeatures


def load_audio(path: str, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio, mono, resampled to target sample rate."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Loaded audio is empty")
    return y, sr


def extract_mel_spectrogram(y: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def _schroeder_backward_integration(energy: np.ndarray) -> np.ndarray:
    """
    Schroeder's method: integrate squared impulse response energy from the
    tail backward to the start, giving a smooth decay curve in dB.
    """
    cumulative = np.cumsum(energy[::-1])[::-1]
    cumulative = np.maximum(cumulative, 1e-12)  # avoid log(0)
    decay_db = 10.0 * np.log10(cumulative / cumulative[0])
    return decay_db


def _estimate_late_reverb_tail(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Blind RIR proxy: locate the highest-energy onset (treated as the
    'direct sound' impulse-like event) and take the tail that follows it,
    which behaves like a decaying impulse response for reverberant speech.
    This is a simplification appropriate for a Week 1 scaffold — the
    Week 3 AST model will learn a more robust version of this from data.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0:
        return None
    frame = int(np.argmax(onset_env))
    start_sample = librosa.frames_to_samples(frame)
    start_sample = max(0, min(start_sample, len(y) - 1))
    tail = y[start_sample:]
    if len(tail) < sr * 0.1:  # need at least 100ms of tail to be meaningful
        return None
    return tail


def extract_rir_features(y: np.ndarray, sr: int) -> RIRFeatures:
    tail = _estimate_late_reverb_tail(y, sr)
    if tail is None:
        # Not enough signal to estimate reverberant tail; return safe defaults
        return RIRFeatures(rt60_seconds=0.0, drr_db=0.0, c50_db=0.0, decay_envelope_db=[])

    energy = tail.astype(np.float64) ** 2
    decay_db = _schroeder_backward_integration(energy)

    # RT60 via linear regression on the -5dB to -25dB region (EDT-style proxy,
    # since a full -60dB range rarely survives in a short speech clip)
    idx_start = np.argmax(decay_db <= -5)
    idx_end = np.argmax(decay_db <= -25)
    if idx_end <= idx_start:
        idx_end = len(decay_db) - 1
    if idx_end > idx_start:
        t = np.arange(idx_start, idx_end + 1) / sr
        d = decay_db[idx_start:idx_end + 1]
        if len(t) >= 2:
            slope, _ = np.polyfit(t, d, 1)
            rt60 = -60.0 / slope if slope < 0 else 0.0
        else:
            rt60 = 0.0
    else:
        rt60 = 0.0
    rt60 = float(np.clip(rt60, 0.0, 5.0))

    # DRR: ratio of energy in first 5ms ("direct") vs the rest ("reverberant")
    direct_samples = int(0.005 * sr)
    direct_energy = np.sum(energy[:direct_samples]) + 1e-12
    reverb_energy = np.sum(energy[direct_samples:]) + 1e-12
    drr_db = float(10.0 * np.log10(direct_energy / reverb_energy))

    # C50: ratio of energy in first 50ms vs energy after 50ms (speech clarity)
    c50_samples = int(0.050 * sr)
    early_energy = np.sum(energy[:c50_samples]) + 1e-12
    late_energy = np.sum(energy[c50_samples:]) + 1e-12
    c50_db = float(10.0 * np.log10(early_energy / late_energy))

    # Downsample decay curve to ~100 points for lightweight UI transport
    n_points = min(100, len(decay_db))
    idxs = np.linspace(0, len(decay_db) - 1, n_points).astype(int)
    decay_envelope_db = decay_db[idxs].round(2).tolist()

    return RIRFeatures(
        rt60_seconds=round(rt60, 3),
        drr_db=round(drr_db, 2),
        c50_db=round(c50_db, 2),
        decay_envelope_db=decay_envelope_db,
    )


def process_audio_file(path: str) -> ExtractionResult:
    """Main entrypoint used by the FastAPI route."""
    y, sr = load_audio(path)
    mel = extract_mel_spectrogram(y, sr)
    mfcc = extract_mfcc(y, sr)
    rir = extract_rir_features(y, sr)

    return ExtractionResult(
        duration_seconds=round(len(y) / sr, 3),
        sample_rate=sr,
        mel_spectrogram=np.round(mel, 2).tolist(),
        mfcc=np.round(mfcc, 2).tolist(),
        rir=rir,
    )


def result_to_dict(result: ExtractionResult) -> dict:
    d = asdict(result)
    return d
