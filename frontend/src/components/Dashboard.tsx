import type { AnalyzeResponse } from "../lib/api";
import { WaveformPlayer } from "./WaveformPlayer";
import { SpectrogramCanvas } from "./SpectrogramCanvas";
import { ResultsPanel } from "./ResultsPanel";

/**
 * Week 3: confidence-score panel is now live, backed by the baseline
 * model + breathing-cadence analysis. Suspicious segments (pauses with
 * no breath signature) are highlighted directly on the waveform.
 */
interface DashboardProps {
  file: File | null;
  result: AnalyzeResponse | null;
  isLoading: boolean;
  error: string | null;
}

export function Dashboard({ file, result, isLoading, error }: DashboardProps) {
  const suspiciousSegments = result?.breathing_analysis?.suspicious_segments ?? [];

  return (
    <div className="dashboard">
      <section className="panel panel--waveform">
        <header className="panel__header">
          <span className="panel__eyebrow">Waveform</span>
          <span className="panel__status panel__status--live">live</span>
        </header>
        <div className="panel__body">
          <WaveformPlayer file={file} suspiciousSegments={suspiciousSegments} />
        </div>
      </section>

      <section className="panel panel--spectrogram">
        <header className="panel__header">
          <span className="panel__eyebrow">Mel Spectrogram</span>
          <span className="panel__status panel__status--live">live</span>
        </header>
        <div className="panel__body">
          {isLoading && <div className="panel__body--placeholder">Analyzing…</div>}
          {!isLoading && <SpectrogramCanvas melSpectrogram={result?.mel_spectrogram ?? null} />}
        </div>
      </section>

      <section className="panel panel--score">
        <header className="panel__header">
          <span className="panel__eyebrow">Confidence Score</span>
          <span className="panel__status panel__status--live">live · baseline model</span>
        </header>
        <div className="panel__body">
          <ResultsPanel result={result} isLoading={isLoading} />
        </div>
      </section>

      <section className="panel panel--rir">
        <header className="panel__header">
          <span className="panel__eyebrow">RIR Diagnostics</span>
          <span className="panel__status panel__status--live">live</span>
        </header>
        <div className="panel__body">
          {error && <p className="upload-error" role="alert">{error}</p>}
          <dl className="metric-list">
            <div className="metric-list__row">
              <dt>RT60</dt>
              <dd>{result ? `${result.rir.rt60_seconds}s` : "—"}</dd>
            </div>
            <div className="metric-list__row">
              <dt>DRR</dt>
              <dd>{result ? `${result.rir.drr_db} dB` : "—"}</dd>
            </div>
            <div className="metric-list__row">
              <dt>C50</dt>
              <dd>{result ? `${result.rir.c50_db} dB` : "—"}</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  );
}
