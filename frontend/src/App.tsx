import { useState } from "react";
import { AudioUpload } from "./components/AudioUpload";
import { Dashboard } from "./components/Dashboard";
import { HistoryPanel } from "./components/HistoryPanel";
import { analyzeAudio, type AnalyzeResponse } from "./lib/api";
import { useAnalysisHistory } from "./lib/history";
import "./styles/app.css";

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { entries, addEntry, clear } = useAnalysisHistory();

  const handleFileSelected = async (file: File) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
    setIsLoading(true);
    try {
      const response = await analyzeAudio(file);
      setResult(response);
      addEntry(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__mark">AcousticSpace</div>
        <div className="app-header__tagline">Deepfake detection via Room Impulse Response analysis</div>
      </header>

      <main className="app-main">
        <AudioUpload
          onFileSelected={handleFileSelected}
          selectedFileName={selectedFile?.name}
        />

        {error && (
          <div className="global-error" role="alert">
            <span className="global-error__title">Analysis failed</span>
            <span className="global-error__detail">{error}</span>
          </div>
        )}

        <Dashboard file={selectedFile} result={result} isLoading={isLoading} error={null} />

        <HistoryPanel entries={entries} onClear={clear} />
      </main>

      <footer className="app-footer">
        Week 4 — containerized &amp; deployment-ready · history persists across sessions
      </footer>
    </div>
  );
}
