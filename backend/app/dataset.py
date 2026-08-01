"""
AcousticSpace - Dataset Curation (Week 1)

Real dataset (for later weeks / actual model training):
  ASVspoof 2019/2021 LA (Logical Access) partition is the standard
  benchmark for synthetic/deepfake speech detection.
    -> https://www.asvspoof.org/  (registration required to download)
  Place downloaded audio under: backend/data/asvspoof/{train,dev,eval}/

For Week 1 we don't need the full dataset yet — we just need enough
signal to prove the Librosa + RIR pipeline works end-to-end. This
script synthesizes a small local dataset using pyroomacoustics:
  - "genuine" clips: dry speech-like tone convolved with ONE room's RIR
  - "fake" clips: the same dry signal convolved with a DIFFERENT room's
    RIR than what's claimed — simulating the acoustic-mismatch signature
    AcousticSpace is designed to catch.

Run:
  python -m app.dataset
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import pyroomacoustics as pra

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
SR = 16000


def _dry_speech_like_signal(duration_s: float = 3.0, sr: int = SR) -> np.ndarray:
    """
    Placeholder 'dry' source signal standing in for a clean voice recording.
    Built from a few harmonics + amplitude modulation to roughly mimic
    speech-like spectral/temporal structure until real speech clips are
    dropped into data/asvspoof/.
    """
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    f0 = 120.0  # rough pitch
    voiced = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)  # syllable-rate AM
    signal = voiced * envelope
    signal += 0.02 * np.random.randn(len(t))  # light noise floor
    return signal / np.max(np.abs(signal))


def _simulate_room(dims, rt60_target, source_pos, mic_pos, dry_signal, sr=SR) -> np.ndarray:
    e_absorption, max_order = pra.inverse_sabine(rt60_target, dims)
    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_pos, signal=dry_signal)
    room.add_microphone(mic_pos)
    room.simulate()
    wet = room.mic_array.signals[0]
    wet = wet / (np.max(np.abs(wet)) + 1e-9) * 0.9
    return wet.astype(np.float32)


def generate_synthetic_dataset(n_pairs: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    genuine_dir = OUT_DIR / "genuine"
    fake_dir = OUT_DIR / "fake"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Two distinct rooms: a "claimed" small room and a "mismatched" larger hall
    small_room_dims = [4.0, 3.5, 2.8]
    large_room_dims = [10.0, 8.0, 4.0]

    for i in range(n_pairs):
        dry = _dry_speech_like_signal()
        src = [1.5, 1.2, 1.5]
        mic = [2.5, 2.0, 1.5]

        # Genuine: voice and claimed room RIR match (small room, short RT60)
        genuine = _simulate_room(small_room_dims, rt60_target=0.35, source_pos=src, mic_pos=mic, dry_signal=dry)
        sf.write(genuine_dir / f"genuine_{i:03d}.wav", genuine, SR)

        # Fake: same dry voice, but acoustically it's actually from a large hall
        # (i.e. a voice claimed to be recorded in a small room, RIR says otherwise)
        fake = _simulate_room(large_room_dims, rt60_target=1.2, source_pos=[4, 3, 2], mic_pos=[6, 5, 2], dry_signal=dry)
        sf.write(fake_dir / f"fake_{i:03d}.wav", fake, SR)

    print(f"Synthetic dataset written to: {OUT_DIR}")
    print(f"  genuine/: {n_pairs} files (short-RT60 room)")
    print(f"  fake/:    {n_pairs} files (mismatched long-RT60 room)")


if __name__ == "__main__":
    generate_synthetic_dataset()
