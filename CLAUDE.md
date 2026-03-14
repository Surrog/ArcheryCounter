# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ArcheryCounter is a React Native app that detects arrows in archery target photos and scores them automatically. The detection logic is implemented in pure TypeScript (no native C++ or OpenCV — see `docs/research.md` and `docs/plan.md` for the migration history).

## Repository Structure

```
src/              React Native JS/TS source (screens, components, hooks, algorithm)
  targetDetection.ts   Core computer vision algorithm (pure TS port of the C++ pipeline)
  imageLoader.ts       Image decoding helpers (jpeg-js for RN, jimp for Node.js tests)
  ArcheryCounter.ts    Public API: processImage(uri, base64) → RingEllipse[]
  useArcheryScorer.ts  Hook: image picker → processImage → state
  components/          RingOverlay.tsx — SVG ellipse overlay
  screens/             HomeScreen.tsx
ios/              React Native iOS project (Xcode workspace)
android/          React Native Android project (Gradle)
images/           Test images (*.jpg) used by targetDetection integration tests
docs/             Architecture and research documentation
```

## Build & Run

Install JS dependencies:
```bash
npm install
```

Run tests (includes algorithm integration tests — expect ~45 s per image in pure JS):
```bash
npm test
```

**iOS:**
```bash
cd ios && pod install && cd ..
npx react-native run-ios --simulator "iPhone 17 Pro"
```

**Android:**
```bash
npx react-native run-android
```

## Architecture

### Detection pipeline (`src/targetDetection.ts`)

`findTarget(rgba, width, height)` — the main entry point:

1. **Pretreatment** — Gaussian blur (15×15, σ=1.5) + erode×1 + dilate×3, applied per R/G/B channel
2. **Adaptive colour detection** — two-pass HSV filtering (wide range → adaptive re-centering around
   measured median hue) for yellow, red, and blue zones; returns `ColorBlob | null` per colour
3. **Centre + scale bootstrap** — mean of blob centroids = target centre; ring-width `w` from
   blob mean-radii ÷ WA zone centroid ratios (yellow 1.33w, red 3.11w, blue 5.07w)
4. **Radial profile sampling** — 360 rays from the bootstrap centre; luminance transitions along
   each ray are detected and matched to the 10 expected ring boundaries
5. **Fitzgibbon ellipse fit** — Halir-Flusser (1998) constrained algebraic fit on each ring's
   transition points; concentric centre forced from bootstrap; outlier fits rejected and filled by
   linear interpolation; rings sorted by width ascending

Returns `EllipseData[10]` — **index 0 = innermost (bullseye), index 9 = outermost**.

### HSV convention

Uses standard RGB→HSV conversion (H 0–360°, S/V 0–1). The old BGR-as-RGB quirk (`rgbToHsvFull`)
has been removed. Wide initial ranges (yellow 20–70°, red 0–18°+342–360°, blue 190–245°) are
adaptively re-centred per image around the measured median hue. See `docs/research.md` §3 for
the old failure modes and §5 for the new algorithm rationale.

### Image loading

- **React Native app**: `launchImageLibrary({ includeBase64: true, maxWidth: 1200 })` + `jpeg-js` (pure JS JPEG decoder, works in Hermes)
- **Jest tests**: `jimp` (devDependency, Node.js only) in `src/imageLoader.ts::loadImageNode`

### React Native layer

- `src/ArcheryCounter.ts` — calls `findTarget` directly (no `NativeModules`)
- `src/useArcheryScorer.ts` — hook: picks image → decodes base64 → runs algorithm → state
- `src/components/RingOverlay.tsx` — SVG overlay using `src/letterboxTransform.ts` for coordinate mapping
- `src/screens/HomeScreen.tsx` — main screen

## Key Design Notes

- Ring index 0 = innermost (bullseye), index 9 = outermost. Scoring: 10 = bullseye, 1 = outermost, 0 = miss.
- Yellow, red, and blue zones anchor the scale estimate; all 10 ring boundaries are measured via radial profiling (not extrapolated from colour blobs alone).
- `ArcheryCounterModule.java` (Android) still exists for RN module registration boilerplate but no longer loads a JNI library.
- Algorithm tests in `src/__tests__/targetDetection.test.ts` run against real photos in `images/` and take ~45 s each in pure JS; the Jest timeout is set to 120 s per test.
