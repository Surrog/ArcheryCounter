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
2. **Color filtering** — three binary masks via HSV thresholding (yellow, red, blue rings)
3. **Blob detection** — BFS flood-fill to find the largest blob per color; nearby blobs are aggregated
4. **Ellipse fitting** — boundary pixels extracted, outliers cleaned, PCA fit (closed-form 2×2 eigenvalue)
5. **Weakest element** — the most deviant of the 3 detected ellipses is discarded
6. **Linear interpolation** — least-squares regression extrapolates all 10 ring ellipses

Returns `EllipseData[10]` — **index 0 = innermost (bullseye), index 9 = outermost**.

### HSV convention

The thresholds were calibrated with OpenCV's `COLOR_RGB2HSV_FULL` applied to a BGR-loaded image (R↔B channels swapped). `rgbToHsvFull(r, g, b)` replicates this by internally using `b` as "R" and `r` as "B". See `docs/research.md` for details.

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
- The outer 4 rings (black, white) are extrapolated, never directly detected — only yellow, red, blue are detected from color.
- `ArcheryCounterModule.java` (Android) still exists for RN module registration boilerplate but no longer loads a JNI library.
- Algorithm tests in `src/__tests__/targetDetection.test.ts` run against real photos in `images/` and take ~45 s each in pure JS; the Jest timeout is set to 120 s per test.
