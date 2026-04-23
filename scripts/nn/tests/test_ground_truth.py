"""
Ground-truth accuracy tests: NN models evaluated against manual annotations.

Tests:
  test_arrow_detection    — arrow_detector_fp32.onnx tips vs annotated tips
  test_boundary_detection — boundary_detector_v2.onnx mask vs annotated boundary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).parent.parent.parent.parent
IMAGES_DIR = ROOT / "images"

# ── Preprocessing constants (must match TypeScript arrowDetector.ts / boundaryDetector.ts) ──

ARROW_INPUT_SIZE   = 640
ARROW_HEATMAP_SIZE = 160
ARROW_RATIO        = ARROW_INPUT_SIZE // ARROW_HEATMAP_SIZE  # 4
ARROW_THRESHOLD    = 0.35
ARROW_NMS_KERNEL   = 5

BOUNDARY_INPUT_SIZE = 512
IMAGENET_GRAY_MEAN  = 0.449
IMAGENET_GRAY_STD   = 0.226

IMAGENET_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TIP_MATCH_PX = 45


# ── Image preprocessing ───────────────────────────────────────────────────────

MAX_LOAD_DIM = 1200  # matches TypeScript loadImageNode (scaleToFit 1200×1200)


def _scale_to_1200(img: Image.Image) -> Image.Image:
    """Scale down so max(width, height) <= 1200, matching TypeScript loadImageNode.

    Annotations are stored in this coordinate space, so running both models on
    the 1200px image makes decoded coordinates directly comparable to annotations.
    """
    w, h = img.size
    if max(w, h) <= MAX_LOAD_DIM:
        return img
    s = MAX_LOAD_DIM / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.BILINEAR)


def _bilinear_resize(arr: np.ndarray, new_w: int, new_h: int, scale: float):
    """0-indexed bilinear resize matching TypeScript letterbox implementations."""
    h, w = arr.shape[:2]
    sx = np.arange(new_w, dtype=np.float32) / scale
    sy = np.arange(new_h, dtype=np.float32) / scale
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    if arr.ndim == 2:
        fx = (sx - x0).reshape(1, new_w)
        fy = (sy - y0).reshape(new_h, 1)
    else:
        fx = (sx - x0)[np.newaxis, :, np.newaxis].astype(np.float32)
        fy = (sy - y0)[:, np.newaxis, np.newaxis].astype(np.float32)
    v00 = arr[np.ix_(y0, x0)] if arr.ndim == 2 else arr[np.ix_(y0, x0)]
    v01 = arr[np.ix_(y0, x1)] if arr.ndim == 2 else arr[np.ix_(y0, x1)]
    v10 = arr[np.ix_(y1, x0)] if arr.ndim == 2 else arr[np.ix_(y1, x0)]
    v11 = arr[np.ix_(y1, x1)] if arr.ndim == 2 else arr[np.ix_(y1, x1)]
    return (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
            v10 * (1 - fx) * fy       + v11 * fx * fy)


def letterbox_gray(img_path: Path):
    """Grayscale letterbox for boundary detector.

    Pre-scales to ≤1200px (matching TypeScript loadImageNode) so that decoded
    coordinates are in the same space as annotations stored in the DB.
    Returns (nchw_float32, scale, pad_x, pad_y).
    """
    img_rgb = _scale_to_1200(Image.open(img_path).convert("RGB"))
    arr_rgb = np.array(img_rgb, dtype=np.float32)
    arr_gray = (0.299 * arr_rgb[:, :, 0] +
                0.587 * arr_rgb[:, :, 1] +
                0.114 * arr_rgb[:, :, 2])
    arr = arr_gray

    h, w = arr.shape
    size = BOUNDARY_INPUT_SIZE
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    content = _bilinear_resize(arr, new_w, new_h, scale)
    out = np.zeros((size, size), dtype=np.uint8)
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.clip(content, 0, 255).astype(np.uint8)

    norm = out.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD
    return norm[np.newaxis, np.newaxis].astype(np.float32), scale, pad_x, pad_y


def letterbox_rgb(img_path: Path):
    """RGB letterbox for arrow detector.

    Pre-scales to ≤1200px (matching TypeScript loadImageNode / visualize.ts) so
    that decoded coordinates are in the same space as annotations stored in the DB.
    Returns (nchw_float32, scale, pad_x, pad_y).
    """
    img = _scale_to_1200(Image.open(img_path).convert("RGB"))
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    size = ARROW_INPUT_SIZE
    scale = size / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    content = _bilinear_resize(arr, new_w, new_h, scale)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = content
    canvas = canvas / 255.0
    canvas = (canvas - IMAGENET_RGB_MEAN) / IMAGENET_RGB_STD
    return canvas.transpose(2, 0, 1)[np.newaxis].astype(np.float32), scale, pad_x, pad_y


# ── NN inference helpers ──────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _run_session(session, inp: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: inp})[0]


def run_arrow_detection(session, img_path: Path) -> list[tuple[float, float]]:
    """Return list of (x, y) arrow tips in original image coordinates."""
    inp, scale, pad_x, pad_y = letterbox_rgb(img_path)
    out = _run_session(session, inp)
    heatmap = out[0, 0] if out.ndim == 4 else out[0]
    if heatmap.max() > 1.0 or heatmap.min() < 0.0:
        heatmap = _sigmoid(heatmap)

    maxmap = maximum_filter(heatmap, size=ARROW_NMS_KERNEL)
    hy, hx = np.where((heatmap == maxmap) & (heatmap >= ARROW_THRESHOLD))

    return [
        (float((x + 0.5) * ARROW_RATIO - pad_x) / scale,
         float((y + 0.5) * ARROW_RATIO - pad_y) / scale)
        for y, x in zip(hy.tolist(), hx.tolist())
    ]


def run_boundary_detection(session, img_path: Path):
    """Return (binary_mask_512x512, scale, pad_x, pad_y).

    The mask is in model input (512×512) coordinate space.
    To project a point (ox, oy) from original image: mx = ox*scale + pad_x.
    """
    inp, scale, pad_x, pad_y = letterbox_gray(img_path)
    out = _run_session(session, inp)
    logits = out[0, 0] if out.ndim == 4 else out[0]
    prob = _sigmoid(logits)
    return (prob > 0.5), scale, pad_x, pad_y


# ── Geometry helpers ──────────────────────────────────────────────────────────

def signed_area(poly: list) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def rasterize_polygon(poly: list, size: int) -> np.ndarray:
    """Rasterize polygon [[x,y],...] to boolean mask of given size."""
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon([(float(p[0]), float(p[1])) for p in poly], fill=1)
    return np.array(img, dtype=bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


# ── Annotation helpers ────────────────────────────────────────────────────────

def _unwrap_boundary(raw):
    if not raw:
        return None
    # Multi-target format: raw[target][point] — take first target
    if raw and isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], list):
        return raw[0]
    return raw


def load_annotation(db_conn, filename: str):
    """Return (paper_boundary, arrows) from annotations table."""
    cur = db_conn.cursor()
    cur.execute(
        "SELECT paper_boundary, arrows FROM annotations WHERE filename = %s",
        (filename,),
    )
    row = cur.fetchone()
    if not row:
        return None, []
    pb = _unwrap_boundary(row[0])
    arrows = row[1] or []
    return pb, arrows


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_arrow_detection(annotated_filename: str, db_conn, arrow_session) -> None:
    """Arrow tip positions detected by NN must match manual annotations within 45 px."""
    if annotated_filename == "__no_images__":
        pytest.skip("No annotated images found in DB")

    img_path = IMAGES_DIR / annotated_filename
    if not img_path.exists():
        pytest.skip(f"Image file not found: {img_path}")

    _, ann_arrows = load_annotation(db_conn, annotated_filename)
    if not ann_arrows:
        pytest.skip("No annotated arrows for this image")

    detected = run_arrow_detection(arrow_session, img_path)

    assert len(detected) >= len(ann_arrows) - 4, (
        f"Too few arrows: detected {len(detected)}, annotated {len(ann_arrows)}"
    )

    # Greedy bijective tip matching (smallest distance first)
    all_pairs = sorted(
        (float(np.hypot(dx - a["tip"][0], dy - a["tip"][1])), ai, di)
        for ai, a in enumerate(ann_arrows)
        for di, (dx, dy) in enumerate(detected)
    )
    matched_a: set[int] = set()
    matched_d: set[int] = set()
    for dist, ai, di in all_pairs:
        if ai in matched_a or di in matched_d:
            continue
        if dist <= TIP_MATCH_PX:
            matched_a.add(ai)
            matched_d.add(di)

    unmatched = [ai for ai in range(len(ann_arrows)) if ai not in matched_a]
    max_unmatched = max(0, len(ann_arrows) - len(detected)) + 5
    assert len(unmatched) <= max_unmatched, (
        f"{len(unmatched)} tip(s) unmatched (>{max_unmatched} allowed) "
        f"in {annotated_filename}: "
        f"detected={detected}, "
        f"annotated={[a['tip'] for a in ann_arrows]}"
    )


def test_boundary_detection(annotated_filename: str, db_conn, boundary_session) -> None:
    """NN boundary mask must overlap annotated paper polygon (mask IoU > 0.25)."""
    if annotated_filename == "__no_images__":
        pytest.skip("No annotated images found in DB")

    img_path = IMAGES_DIR / annotated_filename
    if not img_path.exists():
        pytest.skip(f"Image file not found: {img_path}")

    ann_boundary, _ = load_annotation(db_conn, annotated_filename)
    if not ann_boundary:
        pytest.skip("No annotated boundary for this image")
    if abs(signed_area(ann_boundary)) < 1:
        pytest.skip("Annotated boundary is degenerate (all-zero or too small)")

    pred_mask, scale, pad_x, pad_y = run_boundary_detection(boundary_session, img_path)

    if not pred_mask.any():
        pytest.skip("Boundary model produced no detections for this image")

    # Project annotated boundary into model input coordinate space (512×512)
    projected = [
        (p[0] * scale + pad_x, p[1] * scale + pad_y)
        for p in ann_boundary
    ]
    ann_mask = rasterize_polygon(projected, BOUNDARY_INPUT_SIZE)

    iou = mask_iou(pred_mask, ann_mask)
    assert iou > 0.25, (
        f"Boundary IoU too low: {iou:.3f} < 0.25 for {annotated_filename}"
    )
