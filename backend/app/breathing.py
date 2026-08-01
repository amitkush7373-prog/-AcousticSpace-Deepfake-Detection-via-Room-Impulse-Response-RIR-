"""
AcousticSpace - Breathing Cadence Analysis (Week 3)

Real speech has breath sounds that occur at natural pauses between
phrases, roughly every 6-14 syllables depending on the speaker. Many
deepfake/TTS generators either omit breath sounds entirely, or insert
them without regard to natural breathing physiology (wrong rate, or
pauses with no breath-like energy in them at all).

This module:
  1. Detects syllable-rate onsets (speech energy peaks) via Librosa
  2. Detects pause regions between voiced segments (low-RMS stretches)
  3. Checks each pause for a breath-like low-frequency energy signature
  4. Scores how well the detected breath rate aligns with the natural
     cadence implied by the syllable rate
"""

from __future__ import annotations

import numpy as np
import librosa

MIN_PAUSE_MS = 150
BREATH_BAND_HZ = (60, 400)     # typical inhale/exhale spectral energy band
EXPECTED_SYLLABLES_PER_BREATH = (6, 14)  # natural human range
BREATH_BAND_ENERGY_THRESHOLD = 0.12      # fraction of pause energy in-band to call it a "breath"


def _rms_envelope(y: np.ndarray, sr: int, frame_length: int = 1024, hop_length: int = 256):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, times


def _detect_pauses(rms: np.ndarray, times: np.ndarray, silence_ratio: float = 0.15) -> list[tuple[float, float]]:
    threshold = silence_ratio * (rms.max() + 1e-9)
    is_silent = rms < threshold
    pauses: list[tuple[float, float]] = []
    start = None
    for i, silent in enumerate(is_silent):
        if silent and start is None:
            start = times[i]
        elif not silent and start is not None:
            end = times[i]
            if (end - start) * 1000 >= MIN_PAUSE_MS:
                pauses.append((start, end))
            start = None
    if start is not None:
        end = times[-1]
        if (end - start) * 1000 >= MIN_PAUSE_MS:
            pauses.append((start, end))
    return pauses


def _has_breath_signature(y: np.ndarray, sr: int, start_s: float, end_s: float) -> bool:
    start_sample = int(start_s * sr)
    end_sample = int(end_s * sr)
    segment = y[start_sample:end_sample]
    if len(segment) < sr * 0.05:
        return False
    fft = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1 / sr)
    band_mask = (freqs >= BREATH_BAND_HZ[0]) & (freqs <= BREATH_BAND_HZ[1])
    if not band_mask.any():
        return False
    band_energy = fft[band_mask].sum()
    total_energy = fft.sum() + 1e-9
    return (band_energy / total_energy) > BREATH_BAND_ENERGY_THRESHOLD


def analyze_breathing_cadence(y: np.ndarray, sr: int) -> dict:
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    n_syllables = len(onsets)
    duration = len(y) / sr

    rms, times = _rms_envelope(y, sr)
    pauses = _detect_pauses(rms, times)

    flagged_pauses = []  # pauses with NO breath signature -> suspicious
    breath_pauses = []
    for p in pauses:
        if _has_breath_signature(y, sr, *p):
            breath_pauses.append(p)
        else:
            flagged_pauses.append(p)

    expected_breaths = 0
    if n_syllables > 0:
        avg_expected = sum(EXPECTED_SYLLABLES_PER_BREATH) / 2
        expected_breaths = max(1, round(n_syllables / avg_expected))

    detected_breaths = len(breath_pauses)
    if expected_breaths == 0:
        alignment_score = 0.0
    else:
        alignment_score = 1.0 - min(1.0, abs(detected_breaths - expected_breaths) / expected_breaths)

    return {
        "duration_seconds": round(duration, 3),
        "n_syllable_onsets": n_syllables,
        "n_pauses_detected": len(pauses),
        "n_breath_like_pauses": detected_breaths,
        "expected_breath_count": expected_breaths,
        "cadence_alignment_score": round(float(alignment_score), 3),
        "suspicious_segments": [
            {"start": round(s, 2), "end": round(e, 2), "reason": "pause with no breath signature"}
            for s, e in flagged_pauses
        ],
    }
