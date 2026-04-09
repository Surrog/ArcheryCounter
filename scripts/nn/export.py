"""
Export ArrowDetector checkpoint to ONNX.

Usage:
    python export.py --checkpoint checkpoints/best.pt --out arrow_detector.onnx

The exported model accepts a single input 'image' of shape (1, 4, INPUT_SIZE, INPUT_SIZE)
and produces two outputs:
    tip_hm    : (1, 1, HEATMAP_SIZE, HEATMAP_SIZE)
    score_map : (1, 11, HEATMAP_SIZE, HEATMAP_SIZE)

After export, run static INT8 quantisation with onnxruntime to produce
arrow_detector_int8.onnx (~4×smaller, suitable for mobile).
"""

import argparse
import os

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np

from model import ArrowDetector
from dataset import INPUT_SIZE, HEATMAP_SIZE


def export_onnx(checkpoint_path: str, out_path: str) -> None:
    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    in_ch = ckpt.get('in_channels', 4)
    model = ArrowDetector(in_channels=in_ch)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f'Input channels: {in_ch}')
    dummy = torch.zeros(1, in_ch, INPUT_SIZE, INPUT_SIZE)

    print(f'Exporting to ONNX: {out_path}')
    torch.onnx.export(
        model, dummy, out_path,
        input_names  = ['image'],
        output_names = ['tip_hm', 'score_map'],
        dynamic_axes = {'image': {0: 'batch'}},
        opset_version = 18,
        do_constant_folding = True,
    )

    # Consolidate external data into a single file (torch.onnx.export may split
    # large tensors into a .data sidecar; this merges them back inline).
    m = onnx.load(out_path)   # load_external_data=True by default
    onnx.save(m, out_path)    # re-save inline (no external data for <2 GB models)
    onnx.checker.check_model(onnx.load(out_path))
    print('ONNX model check passed.')

    # Remove orphaned .data sidecar if it still exists
    data_path = out_path + '.data'
    if os.path.exists(data_path):
        os.remove(data_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'FP32 model size: {size_mb:.1f} MB')

    # Quick runtime check
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    out  = sess.run(None, {'image': dummy.numpy()})
    print(f'Runtime check OK — outputs: {[o.shape for o in out]}')


class RealCalibrationReader(CalibrationDataReader):
    """Calibration data from actual training images for accurate INT8 quantisation."""

    def __init__(self, n_samples: int = 30):
        from dataset import ArrowDataset
        ds = ArrowDataset(augment=False, is_val=False)
        indices = list(range(min(n_samples, len(ds))))
        self.data = iter([
            {'image': ds[i]['image'].unsqueeze(0).numpy()}
            for i in indices
        ])

    def get_next(self):
        return next(self.data, None)


def quantize_onnx(fp32_path: str, int8_path: str) -> None:
    print(f'Quantising {fp32_path} → {int8_path}')
    quantize_static(
        model_input             = fp32_path,
        model_output            = int8_path,
        calibration_data_reader = RealCalibrationReader(),
        weight_type             = QuantType.QInt8,
    )
    print(f'INT8 model saved: {int8_path}')
    size_fp32 = os.path.getsize(fp32_path) / 1e6
    size_int8 = os.path.getsize(int8_path) / 1e6
    print(f'Size: FP32={size_fp32:.1f} MB  INT8={size_int8:.1f} MB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to best.pt')
    parser.add_argument('--out',        default='arrow_detector.onnx')
    parser.add_argument('--quantize',   action='store_true',
                        help='Also produce INT8 quantised model')
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.out)

    if args.quantize:
        int8_path = args.out.replace('.onnx', '_int8.onnx')
        quantize_onnx(args.out, int8_path)


if __name__ == '__main__':
    main()
