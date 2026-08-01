import { useCallback, useRef, useState } from "react";

const ACCEPTED = [".wav", ".mp3", ".flac", ".ogg", ".m4a"];

interface AudioUploadProps {
  onFileSelected: (file: File) => void;
  selectedFileName?: string;
}

/**
 * Signature element: a static decay-curve backdrop rendered behind the
 * drop zone. It's not decorative filler — it's a literal preview of the
 * Schroeder decay envelope AcousticSpace computes from every clip,
 * previewed here before any file is even loaded.
 */
function DecayBackdrop() {
  const points = "0,10 8,14 16,20 24,28 32,38 40,50 48,62 56,72 64,80 72,86 80,90 88,93 96,95 100,96";
  return (
    <svg
      className="decay-backdrop"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

export function AudioUpload({ onFileSelected, selectedFileName }: AudioUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndEmit = useCallback(
    (file: File) => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED.includes(ext)) {
        setError(`Unsupported format ${ext}. Use ${ACCEPTED.join(", ")}`);
        return;
      }
      setError(null);
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) validateAndEmit(file);
    },
    [validateAndEmit]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) validateAndEmit(file);
  };

  return (
    <div className="upload-block">
      <div
        className={`upload-zone ${isDragging ? "upload-zone--active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        aria-label="Upload an audio clip for acoustic analysis"
      >
        <DecayBackdrop />
        <div className="upload-zone__content">
          <span className="upload-zone__eyebrow">Step 1 — Submit clip</span>
          <span className="upload-zone__title">
            {selectedFileName ?? "Drop an audio file, or click to browse"}
          </span>
          <span className="upload-zone__hint">WAV · MP3 · FLAC · OGG · M4A, up to 25MB</span>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED.join(",")}
          onChange={handleChange}
          hidden
        />
      </div>
      {error && <p className="upload-error" role="alert">{error}</p>}
    </div>
  );
}
