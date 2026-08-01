import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";
import type { SuspiciousSegment } from "../lib/api";

interface WaveformPlayerProps {
  file: File | null;
  suspiciousSegments?: SuspiciousSegment[];
}

/**
 * Wraps Wavesurfer.js to render the actual uploaded clip's waveform and
 * provide basic playback (Week 2), plus region highlighting for
 * suspicious segments flagged by the breathing-cadence analysis (Week 3).
 */
export function WaveformPlayer({ file, suspiciousSegments = [] }: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const regions = RegionsPlugin.create();
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#8b96a5",
      progressColor: "#f2a94e",
      cursorColor: "#e8ecf1",
      height: 72,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      plugins: [regions],
    });
    wavesurferRef.current = ws;
    regionsRef.current = regions;

    ws.on("ready", () => setIsReady(true));
    ws.on("play", () => setIsPlaying(true));
    ws.on("pause", () => setIsPlaying(false));
    ws.on("finish", () => setIsPlaying(false));

    return () => {
      ws.destroy();
      wavesurferRef.current = null;
      regionsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const ws = wavesurferRef.current;
    if (!ws || !file) return;
    setIsReady(false);
    const url = URL.createObjectURL(file);
    ws.load(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Draw / refresh suspicious-segment regions whenever the clip or the
  // analysis result changes
  useEffect(() => {
    const regions = regionsRef.current;
    if (!regions) return;
    regions.clearRegions();
    if (!isReady) return;

    for (const seg of suspiciousSegments) {
      regions.addRegion({
        start: seg.start,
        end: seg.end,
        color: "rgba(229, 105, 123, 0.28)", // --color-flagged, translucent
        drag: false,
        resize: false,
      });
    }
  }, [suspiciousSegments, isReady]);

  const togglePlay = () => {
    wavesurferRef.current?.playPause();
  };

  return (
    <div className="waveform-player">
      <div ref={containerRef} className="waveform-player__track" />
      {file && (
        <button
          className="waveform-player__button"
          onClick={togglePlay}
          disabled={!isReady}
        >
          {isPlaying ? "Pause" : "Play"}
        </button>
      )}
    </div>
  );
}
