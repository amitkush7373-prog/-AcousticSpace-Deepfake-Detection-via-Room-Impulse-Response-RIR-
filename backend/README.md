# AcousticSpace — Backend (Week 1)

## What's done this week
- FastAPI server (`app/main.py`) with `/health` and `POST /api/analyze`
- Librosa pipeline (`app/audio_pipeline.py`): mel-spectrogram, MFCC, and
  blind RIR feature estimation (RT60, DRR, C50 via Schroeder backward
  integration)
- Synthetic dataset generator (`app/dataset.py`) using pyroomacoustics,
  since ASVspoof requires registration and isn't reachable from this
  environment — swap in real ASVspoof clips under `data/asvspoof/` later
  without changing any pipeline code

## Validated
- Pipeline runs end-to-end on generated genuine vs. mismatched-room clips
- DRR and C50 correctly separate direction (mismatched/larger room →
  more negative DRR/C50, as physically expected)
- RT60 estimate is currently noisy on continuous (non-impulsive) signal —
  expected for a Week 1 blind estimate; will tighten once cepstral
  estimation / real speech data is in the loop

## Run it
```bash
pip install --break-system-packages -r requirements.txt
python -m app.dataset            # generates data/synthetic/{genuine,fake}/
uvicorn app.main:app --reload --port 8000
```

Test:
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@data/synthetic/genuine/genuine_000.wav"
```

## Next (Week 2)
- Baseline CNN/Transformer classifier on extracted features
- Wire real ASVspoof data into `data/asvspoof/`
