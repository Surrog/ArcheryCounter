"""
Preprocessing parity check for boundary detection training.

Applies the same RGB→gray→letterbox→normalise pipeline that BoundaryDataset
uses during training, then reports normalised pixel values at requested
coordinates.  Called by nnRegression.test.ts to assert that the Python
training preprocessing exactly matches TypeScript letterboxGray inference
preprocessing.

Any divergence here means the model is trained on different data than what
it will see at inference time.

Usage:
    python3 check_preprocessing_parity.py <image_path> '<json_coords>'

    <json_coords>  JSON array of [x, y] pairs, e.g. '[[0,0],[100,100]]'

Output (stdout): JSON object
    {
      "values":  [normalised_float, ...],  // one per coord, same order
      "scale":   float,
      "pad_x":   int,
      "pad_y":   int,
      "orig_w":  int,
      "orig_h":  int
    }
"""
import json
import sys
import os

import numpy as np
from PIL import Image

# Import letterbox and normalisation constants from the training dataset module.
sys.path.insert(0, os.path.dirname(__file__))
from boundary_dataset import letterbox, IMAGENET_GRAY_MEAN, IMAGENET_GRAY_STD  # noqa: E402

def main():
    if len(sys.argv) != 3:
        print('Usage: check_preprocessing_parity.py <image_path> <json_coords>',
              file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    coords   = json.loads(sys.argv[2])  # [[x, y], ...]

    # ── Replicate BoundaryDataset.__getitem__ preprocessing exactly ───────────
    img_rgb  = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img_rgb.size

    arr_rgb  = np.array(img_rgb, dtype=np.float32)
    arr_gray = (0.299 * arr_rgb[:, :, 0] +
                0.587 * arr_rgb[:, :, 1] +
                0.114 * arr_rgb[:, :, 2])
    img_gray = Image.fromarray(np.clip(arr_gray, 0, 255).astype(np.uint8))

    img_lb, scale, pad_x, pad_y = letterbox(img_gray)

    arr = np.array(img_lb, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_GRAY_MEAN) / IMAGENET_GRAY_STD   # (512, 512) float32

    # ── Sample requested pixels ───────────────────────────────────────────────
    values = [float(arr[y, x]) for x, y in coords]

    print(json.dumps({
        'values': values,
        'scale':  scale,
        'pad_x':  pad_x,
        'pad_y':  pad_y,
        'orig_w': orig_w,
        'orig_h': orig_h,
    }))

if __name__ == '__main__':
    main()
