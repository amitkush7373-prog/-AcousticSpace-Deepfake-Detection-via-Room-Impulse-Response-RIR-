import type { AnalyzeResponse } from "../lib/api";

interface ResultsPanelProps {
  result: AnalyzeResponse | null;
  isLoading: boolean;
}

/**
 * Week 3: the confidence-score panel is now "live" — it shows the
 * baseline model's verdict alongside the breathing-cadence alignment
 * score, and lists any suspicious segments (pauses with no breath
 * signature) that were flagged and highlighted on the waveform.
 */
export function ResultsPanel({ result, isLoading }: ResultsPanelProps) {
  if (isLoading) {
    return <div className="panel__body--placeholder">Analyzing…</div>;
  }

  if (!result?.baseline_prediction) {
    return (
      <div className="score-readout">
        <span className="score-readout__value">—</span>
        <span className="score-readout__label">no verdict yet</span>
      </div>
    );
  }

  const { label, confidence } = result.baseline_prediction;
  const breathing = result.breathing_analysis;
  const verdictClass = label === "fake" ? "verdict-badge--flagged" : "verdict-badge--verified";

  return (
    <div className="results-panel">
      <div className="verdict-row">
        <span className={`verdict-badge ${verdictClass}`}>{label}</span>
        <div className="score-readout">
          <span className="score-readout__value">{Math.round(confidence * 100)}%</span>
          <span className="score-readout__label">baseline model confidence</span>
        </div>
      </div>

      <dl className="metric-list">
        <div className="metric-list__row">
          <dt>Breathing cadence alignment</dt>
          <dd>{Math.round(breathing.cadence_alignment_score * 100)}%</dd>
        </div>
        <div className="metric-list__row">
          <dt>Syllable onsets detected</dt>
          <dd>{breathing.n_syllable_onsets}</dd>
        </div>
        <div className="metric-list__row">
          <dt>Breath-like pauses / expected</dt>
          <dd>{breathing.n_breath_like_pauses} / {breathing.expected_breath_count}</dd>
        </div>
      </dl>

      {breathing.suspicious_segments.length > 0 && (
        <div className="suspicious-list">
          <span className="panel__eyebrow">Flagged segments</span>
          <ul>
            {breathing.suspicious_segments.map((seg, i) => (
              <li key={i}>
                {seg.start}s–{seg.end}s — {seg.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
