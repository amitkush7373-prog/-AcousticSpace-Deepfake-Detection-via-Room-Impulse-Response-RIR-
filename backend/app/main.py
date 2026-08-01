"""
AcousticSpace - FastAPI Server (Week 1: Core Setup)

Endpoints:
  GET  /health          -> liveness check
  POST /api/analyze      -> upload an audio clip, get spectrogram + RIR features back

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

import shutil
import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.audio_pipeline import process_audio_file, result_to_dict, load_audio
from app.predict import predict as run_baseline_prediction, _load_model as _load_baseline_model
from app.breathing import analyze_breathing_cadence

# Week 4: cap CPU threads so we don't oversubscribe when running under
# a container with a CPU limit (torch defaults to using every core it sees,
# which fights with the container's actual quota and adds latency jitter).
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 25

app = FastAPI(
    title="AcousticSpace API",
    description="Deepfake audio detection via Room Impulse Response analysis",
    version="0.1.0",
)

# Week 1: wide-open CORS for local dev against the Vite frontend (localhost:5173).
# Tighten this to specific origins before deployment (Week 4).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_model():
    """
    Load the baseline checkpoint into memory once at process startup
    instead of on the first incoming request — otherwise whoever sends
    the first request eats a multi-second model-load penalty that every
    subsequent request avoids. This is the main lever for consistent
    inference latency in a deployed setting.
    """
    _load_baseline_model()


@app.get("/health")
def health():
    return {"status": "ok", "service": "acousticspace-api"}


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

        t0 = time.perf_counter()

        result = process_audio_file(tmp_path)
        # baseline_prediction is None until `python -m app.train` has produced
        # a checkpoint; the fine-tuned AST model (app/finetune_ast.py) will
        # eventually replace this once it's trained with internet access.
        baseline_prediction = run_baseline_prediction(tmp_path)

        y, sr = load_audio(tmp_path)
        breathing = analyze_breathing_cadence(y, sr)

        inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "filename": file.filename,
            **result_to_dict(result),
            "baseline_prediction": baseline_prediction,
            "breathing_analysis": breathing,
            "inference_ms": inference_ms,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)
