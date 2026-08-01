export interface RIRFeatures {
  rt60_seconds: number;
  drr_db: number;
  c50_db: number;
  decay_envelope_db: number[];
}

export interface BaselinePrediction {
  label: "genuine" | "fake";
  confidence: number;
  probabilities: Record<string, number>;
}

export interface SuspiciousSegment {
  start: number;
  end: number;
  reason: string;
}

export interface BreathingAnalysis {
  duration_seconds: number;
  n_syllable_onsets: number;
  n_pauses_detected: number;
  n_breath_like_pauses: number;
  expected_breath_count: number;
  cadence_alignment_score: number;
  suspicious_segments: SuspiciousSegment[];
}

export interface AnalyzeResponse {
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  mel_spectrogram: number[][];
  mfcc: number[][];
  rir: RIRFeatures;
  baseline_prediction: BaselinePrediction | null;
  breathing_analysis: BreathingAnalysis;
}

const API_BASE = "http://localhost:8000";

export async function analyzeAudio(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Analyze request failed (${res.status})`);
  }

  return res.json();
}
