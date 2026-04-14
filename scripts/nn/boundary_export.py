"""
Export the trained BoundaryDetector to ONNX.

Usage:
    python boundary_export.py [--checkpoint boundary_checkpoints/best.pt]
                              [--out boundary_detector.onnx]
"""

import argparse

import torch

from boundary_model import BoundaryDetector
from boundary_dataset import INPUT_SIZE


def export(args):
    device = torch.device('cpu')
    model = BoundaryDetector().to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    dummy = torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=17,
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image':  {0: 'batch'},
            'logits': {0: 'batch'},
        },
    )
    print(f'Exported to {args.out}')

    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.out)
        out = sess.run(None, {'image': np.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)})
        print(f'ONNX verification OK — output shape: {out[0].shape}')
    except Exception as e:
        print(f'ONNX verification skipped: {e}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='boundary_checkpoints/best.pt')
    p.add_argument('--out',        default='boundary_detector.onnx')
    return p.parse_args()


if __name__ == '__main__':
    export(parse_args())
