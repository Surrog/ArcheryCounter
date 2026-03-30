"""
Export ArrowDetector checkpoint to ONNX.

Usage:
    python export.py --checkpoint checkpoints/best.pt --out arrow_detector.onnx

The exported model accepts a single input 'image' of shape (1, 4, 512, 512)
and produces three outputs:
    tip_hm    : (1, 1, 128, 128)
    nock_hm   : (1, 1, 128, 128)
    score_map : (1, 11, 128, 128)

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


def export_onnx(checkpoint_path: str, out_path: str) -> None:
    print(f'Loading checkpoint: {checkpoint_path}')
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    model = ArrowDetector()
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy = torch.zeros(1, 4, 512, 512)

    print(f'Exporting to ONNX: {out_path}')
    torch.onnx.export(
        model, dummy, out_path,
        input_names  = ['image'],
        output_names = ['tip_hm', 'nock_hm', 'score_map'],
        dynamic_axes = {'image': {0: 'batch'}},
        opset_version = 17,
        do_constant_folding = True,
    )

    # Verify
    m = onnx.load(out_path)
    onnx.checker.check_model(m)
    print('ONNX model check passed.')

    # Quick runtime check
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    out  = sess.run(None, {'image': dummy.numpy()})
    print(f'Runtime check OK — outputs: {[o.shape for o in out]}')


class RandomCalibrationReader(CalibrationDataReader):
    """Provides random calibration data for static quantisation.

    In production, replace with real images from the training set for
    more accurate quantisation.
    """

    def __init__(self, n_samples: int = 20):
        self.data = iter([
            {'image': np.random.rand(1, 4, 512, 512).astype(np.float32)}
            for _ in range(n_samples)
        ])

    def get_next(self):
        return next(self.data, None)


def quantize_onnx(fp32_path: str, int8_path: str) -> None:
    print(f'Quantising {fp32_path} → {int8_path}')
    quantize_static(
        model_input        = fp32_path,
        model_output       = int8_path,
        calibration_data_reader = RandomCalibrationReader(),
        quant_type         = QuantType.QInt8,
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
