import { useEffect, useRef } from "react";

interface SpectrogramCanvasProps {
  melSpectrogram: number[][] | null; // shape: (n_mels, n_frames), values in dB
}

/**
 * Renders the mel-spectrogram returned by /api/analyze as a heatmap.
 * Low energy -> panel background color, high energy -> signal-amber,
 * keeping the same instrumentation palette used across the dashboard.
 */
export function SpectrogramCanvas({ melSpectrogram }: SpectrogramCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !melSpectrogram || melSpectrogram.length === 0) return;

    const nMels = melSpectrogram.length;
    const nFrames = melSpectrogram[0].length;

    canvas.width = nFrames;
    canvas.height = nMels;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let min = Infinity;
    let max = -Infinity;
    for (const row of melSpectrogram) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min || 1;

    const imageData = ctx.createImageData(nFrames, nMels);
    for (let mel = 0; mel < nMels; mel++) {
      // flip vertically so low mel bins (bass) sit at the bottom, like a
      // conventional spectrogram
      const displayRow = nMels - 1 - mel;
      for (let frame = 0; frame < nFrames; frame++) {
        const norm = (melSpectrogram[mel][frame] - min) / range; // 0..1
        const idx = (displayRow * nFrames + frame) * 4;

        // interpolate panel-dark -> signal-amber
        const r = Math.round(20 + norm * (242 - 20));
        const g = Math.round(25 + norm * (169 - 25));
        const b = Math.round(32 + norm * (78 - 32));

        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [melSpectrogram]);

  if (!melSpectrogram) {
    return <div className="panel__body--placeholder">Render target for spectrogram heatmap</div>;
  }

  return <canvas ref={canvasRef} className="spectrogram-canvas" />;
}
