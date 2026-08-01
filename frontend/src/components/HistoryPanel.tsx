import type { HistoryEntry } from "../lib/history";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onClear: () => void;
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/**
 * Week 4: "Add state management for tracking analysis history."
 * Persists to localStorage via useAnalysisHistory so the list survives
 * a page refresh — useful for an analyst working through a batch of clips.
 */
export function HistoryPanel({ entries, onClear }: HistoryPanelProps) {
  return (
    <section className="panel panel--history">
      <header className="panel__header">
        <span className="panel__eyebrow">Analysis History</span>
        {entries.length > 0 && (
          <button className="history-clear" onClick={onClear}>
            Clear
          </button>
        )}
      </header>
      <div className="panel__body">
        {entries.length === 0 ? (
          <div className="panel__body--placeholder">No clips analyzed yet this session</div>
        ) : (
          <ul className="history-list">
            {entries.map((entry) => (
              <li key={entry.id} className="history-item">
                <span className="history-item__name">{entry.filename}</span>
                <span className="history-item__meta">
                  {entry.verdict && (
                    <span
                      className={`verdict-badge verdict-badge--sm ${
                        entry.verdict === "fake" ? "verdict-badge--flagged" : "verdict-badge--verified"
                      }`}
                    >
                      {entry.verdict}
                    </span>
                  )}
                  <span className="history-item__time">{formatTime(entry.timestamp)}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
