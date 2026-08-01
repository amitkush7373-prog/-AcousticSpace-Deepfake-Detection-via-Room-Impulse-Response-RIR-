# AcousticSpace — Frontend (Week 1)

## What's done this week
- React + TypeScript + Vite scaffold
- `AudioUpload` component: drag-and-drop + click-to-browse, file-type
  validation, static decay-curve signature graphic
- `Dashboard` static layout: waveform / spectrogram / confidence-score /
  RIR-diagnostics panels, wired for data but not yet populated (that's
  Week 2–3, once Wavesurfer.js and the classifier exist)
- Design tokens in `src/styles/index.css` (dark instrumentation palette:
  IBM Plex Mono for data readouts, IBM Plex Sans for body)

## Run it
```bash
npm install
npm run dev       # http://localhost:5173
```

Backend must be running on `http://localhost:8000` for future weeks
(CORS is already pre-configured for `localhost:5173`).

## Verified
- `npx tsc -b` — zero type errors
- `npx vite build` — production build succeeds

## Next (Week 2)
- Integrate Wavesurfer.js into the waveform panel
- Wire the upload flow to `POST /api/analyze` and render real mel-spectrogram
