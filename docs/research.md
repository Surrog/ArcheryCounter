# C++ Algorithm Research Notes

Deep analysis of the `native/` C++ codebase, written to inform the TypeScript reimplementation.

---

## Pipeline Overview

```
cv::imread (BGR image)
    │
    ▼
pretreatment()          ← Gaussian blur + erode(1x) + dilate(3x) on full-color image
    │
    ▼
filter_image()          ← 5 parallel color filters → 5 binary masks
  [yellow, red, blue, black, white]
    │
    ▼ (first 3 only: yellow, red, blue)
detect_circle_approach()  ← contour detection + aggregation + cleanup + fitEllipse
  → RotatedRect[3]
    │
    ▼
find_weakest_element()  ← score each ellipse on center deviation, angle, aspect ratio
    │
    ▼
compute_final_ellipses_by_linear_interpolation()  ← linear regression → 10 rings
    │
    ▼
RotatedRect[10]         ← ring 0 = innermost (bullseye), ring 9 = outermost
```

Arrow detection (`NN::find_arrows`) is a **stub** — it returns an empty struct and does nothing.

---

## Critical: HSV Color Space Convention

All color filters call `cv::COLOR_RGB2HSV_FULL` on the image. However, `cv::imread` returns **BGR** images, not RGB. This means the R and B channels are silently swapped before the HSV conversion.

The practical effect: for a pixel that is visually yellow (R=255, G=255, B=0 in BGR), the converter sees (R=0, G=255, B=255), which is cyan. The thresholds are calibrated for this **BGR-as-RGB** behavior.

### Filter thresholds (HSV_FULL, 0–255 range, BGR-as-RGB convention)

| Color  | H min | H max | S min | S max | V min | V max |
|--------|-------|-------|-------|-------|-------|-------|
| Yellow | 136   | 140   | 64    | 255   | 0     | 255   |
| Red    | 168   | 171   | 64    | 255   | 0     | 255   |
| Blue   | 24    | 32    | 64    | 255   | 0     | 255   |
| Black  | 128   | 192   | 16    | 52    | 32    | 96    |
| White  | 0     | 220   | 0     | 25    | 0     | 255   |

**For the TypeScript port**, the same BGR-swap must be applied to replicate the behavior:
when loading an RGBA image (e.g., from jpeg-js), swap the R and B channels before computing HSV.

### HSV_FULL formula (OpenCV)
H, S, V all in range [0, 255].

```
max = max(R, G, B),  min = min(R, G, B),  delta = max - min
V = max
S = (delta / max) * 255   (0 if max == 0)
H:
  if max == R:  H = 42.5 * (G - B) / delta        (mod 256 if negative)
  if max == G:  H = 42.5 * (B - R) / delta + 85
  if max == B:  H = 42.5 * (R - G) / delta + 170
```

---

## Pretreatment (applied to the full-color BGR image)

```
copyMakeBorder(1px, WHITE)   ← prevents contour artifacts at image edge
GaussianBlur(15×15, σ=1.5)  ← smooths noise; large kernel blurs ~7px radius
erode(3×3, 1 iteration)     ← shrinks bright regions, removes noise
dilate(3×3, 3 iterations)   ← expands bright regions, closes gaps
```

The output is a color (3-channel) image. Color filtering then runs on this smoothed image.

---

## Ellipse Detection per Color (detect_circle_approach)

For each of the 3 color binary masks (yellow, red, blue), with `aggregate_contour = true`:

1. **`findContours`** — extracts all contour point sets from the binary mask
2. **Pick largest contour** by point count
3. **Aggregate nearby contours**: compute center-of-mass and mean distance of the largest
   contour; any other contour whose center is within `2.5 × mean_distance` is merged into
   the point set
4. **`cleanup_center_points`** (up to 12 iterations):
   - Fit a trial ellipse with `fitEllipse`
   - Remove points outside a band around the ellipse:
     `band = [lower, upper] × ellipse_width`
   - Band narrows each iteration (starts at [24%, 82%], converges toward [48%, 58%])
5. **`cv::fitEllipse`** (Fitzgibbon-Pilu-Fisher 1996 direct method) on the cleaned points
   → `cv::RotatedRect(center, size(width, height), angle)`

`fitEllipse` requires at least 5 points and solves a least-squares algebraic ellipse fit.

---

## Weakest Element Detection

Scores each of the 3 detected ellipses on three criteria:

| Criterion | Threshold | Score if exceeded |
|-----------|-----------|------------------|
| Distance from centroid of all 3 centers | > 20 px | +1 |
| Angle deviation from mean angle | any | +1 (always to worst) |
| Aspect ratio deviation from mean | > 0.01 | +1 |

The ellipse with **bad_score > 1** is discarded. If none scores > 1, all 3 are used.
If any ellipse has `center == (0,0)` and `size == (0,0)` (detection failed), it is
immediately returned as weakest.

---

## Linear Interpolation to 10 Rings

The 3 detected ellipses (order: **yellow=0, red=1, blue=2**) are treated as samples
along a 1D parameter `x`:

```
x-value of detected ellipse i  = i   (i.e., yellow→0, red→1, blue→2)
```

For each of the 5 parameters (centerX, centerY, width, height, angle), the code
computes linear regression `y = coef * x + constant` using only the non-ignored samples.

The 10 output rings evaluate this line at:

```
x = ring_index * 0.5 − 0.5

ring index 0 → x = −0.5   (smaller than yellow: innermost / bullseye)
ring index 1 → x =  0.0   (same scale as yellow)
ring index 2 → x =  0.5   (between yellow and red)
ring index 3 → x =  1.0   (same scale as red)
ring index 4 → x =  1.5   (between red and blue)
ring index 5 → x =  2.0   (same scale as blue)
ring index 6 → x =  2.5   (extrapolated beyond blue)
ring index 7 → x =  3.0   (extrapolated)
ring index 8 → x =  3.5   (extrapolated)
ring index 9 → x =  4.0   (extrapolated outermost)
```

**Ring 0 is the innermost (bullseye). Ring 9 is the outermost.**

> Note: `CLAUDE.md` previously stated the opposite ordering. The code is authoritative.

The outer rings (index 6–9, corresponding to the black and white rings) are **entirely
extrapolated** — never directly detected from color. The colored rings (yellow, red, blue)
serve as anchor points for the linear fit.

---

## Scoring (main.cpp)

After detection, `compute_target_ring_mask` converts each `RotatedRect` to a **filled**
ellipse mask using `cv::fillConvexPoly`. Masks are cumulative (innermost = smallest).

`get_arrow_point(tip, masks)` scores an arrow:
```
result = 10
for i = 0..9 while masks[i] at tip == 0:  result--
return result   // 10 = bullseye, 1 = outermost ring, 0 = miss
```

---

## Key Observations for TypeScript Port

### Fitzgibbon `fitEllipse` replacement
`cv::fitEllipse` solves a 6×6 generalized eigenvalue problem. A simpler alternative that
avoids external linear algebra:

**PCA on boundary pixels** of the largest blob:
- Extract boundary pixels: foreground pixels with at least one background neighbor
- Compute 2×2 covariance matrix of (x, y) coordinates
- For points uniformly distributed on an ellipse circumference:
  `eigenvalue λ → semi-axis = √(2λ)`, so full axis = `2√(2λ)`
- Angle of major axis = `atan2(λ₁ − cxx, cxy)` in degrees

This is less accurate than Fitzgibbon for noisy or sparse contours but is sufficient
for the test criteria (100 px concentric tolerance, positive sizes).

### Pretreatment on color image
The blur/erode/dilate runs on the **3-channel color image**, not grayscale. In TypeScript,
this means operating on all 3 channels independently (or equivalently, on each pixel's RGB
channels in the flat RGBA array).

### Contour finding
`cv::findContours` returns contour boundary points. A faithful reimplementation uses
connected component labeling (BFS/flood-fill) followed by extracting the border pixels
(any foreground pixel with a background neighbor in the 4-connected neighborhood).

### No grayscale preprocessing
The pipeline never converts to grayscale for target detection. Only `main.cpp` converts
to grayscale for the (stub) arrow detection.

### Saturation minimum for blue
The blue filter saturation lower bound is **64** (not 32 as one summary suggested).
Correct threshold: `H:[24,32], S:[64,255], V:[0,255]`.

### Image loading convention (TypeScript)
`jpeg-js` and jimp return **RGBA** buffers. To replicate the BGR→HSV_FULL behavior,
swap R↔B before the HSV calculation:
```ts
// rgba buffer: [R, G, B, A, R, G, B, A, ...]
const [h, s, v] = hsvFull(b, g, r);  // note: b passed as first arg (acts as "R")
```
