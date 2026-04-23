"""
Regression tests for dataset.py.

Run with:  uv run pytest test_dataset.py -v
"""
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import albumentations as A

from dataset import (
    HEATMAP_SIZE,
    INPUT_SIZE,
    ArrowDataset,
    draw_gaussian,
    letterbox,
    make_radial_channel,
    score_tip,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_rings(cx=100.0, cy=100.0, r_inner=10.0, r_outer=100.0):
    """Return 10 fake rings (list of dicts with 'points') centred at (cx, cy)."""
    rings = []
    for i in range(10):
        r = r_inner + (r_outer - r_inner) * i / 9
        pts = [
            [cx + r * math.cos(a), cy + r * math.sin(a)]
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)
        ]
        rings.append({"points": pts})
    return rings


def _fake_sample(n_arrows=2, img_w=400, img_h=300):
    """Return (fname, arrows, rings) as ArrowDataset.samples entries."""
    rings = _fake_rings(cx=img_w / 2, cy=img_h / 2, r_inner=10, r_outer=min(img_w, img_h) / 2 - 5)
    arrows = [
        {"tip": [img_w / 2 + i * 15, img_h / 2 + i * 10]}
        for i in range(n_arrows)
    ]
    return ("fake.jpg", arrows, rings)


# ── unit tests ────────────────────────────────────────────────────────────────

class TestLetterbox:
    def test_square_image_unchanged(self):
        img = Image.new("RGB", (512, 512))
        out, scale, px, py = letterbox(img, 512)
        assert out.size == (512, 512)
        assert scale == pytest.approx(1.0)
        assert px == 0 and py == 0

    def test_wide_image_padded_vertically(self):
        img = Image.new("RGB", (800, 400))
        out, scale, px, py = letterbox(img, 512)
        assert out.size == (512, 512)
        assert scale == pytest.approx(512 / 800)
        assert px == 0
        assert py > 0

    def test_tall_image_padded_horizontally(self):
        img = Image.new("RGB", (300, 600))
        out, scale, px, py = letterbox(img, 512)
        assert out.size == (512, 512)
        assert scale == pytest.approx(512 / 600)
        assert px > 0
        assert py == 0

    def test_point_mapping(self):
        """A point at image centre maps to the centre of the letterboxed image."""
        w, h = 400, 200
        img = Image.new("RGB", (w, h))
        _, scale, px, py = letterbox(img, 512)
        mapped_x = (w / 2) * scale + px
        mapped_y = (h / 2) * scale + py
        assert mapped_x == pytest.approx(256, abs=1)
        assert mapped_y == pytest.approx(256, abs=1)


class TestDrawGaussian:
    def test_peak_is_one(self):
        hm = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(hm, 64, 64)
        assert hm.max() == pytest.approx(1.0, abs=1e-4)

    def test_peak_location(self):
        hm = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(hm, 30, 50)
        yi, xi = np.unravel_index(hm.argmax(), hm.shape)
        assert xi == 30 and yi == 50

    def test_out_of_bounds_noop(self):
        hm = np.zeros((128, 128), dtype=np.float32)
        draw_gaussian(hm, -500, -500)
        assert hm.max() == 0.0


class TestScoreTip:
    def test_bullseye(self):
        rings = _fake_rings(cx=100, cy=100, r_inner=10, r_outer=100)
        # Just inside the innermost ring
        assert score_tip([100, 100], rings) == 10

    def test_miss(self):
        rings = _fake_rings(cx=100, cy=100, r_inner=10, r_outer=50)
        # Far outside
        assert score_tip([300, 300], rings) == 0

    def test_score_decreases_outward(self):
        rings = _fake_rings(cx=100, cy=100, r_inner=5, r_outer=100)
        scores = []
        for r in [0, 15, 30, 45, 60, 75, 90]:
            scores.append(score_tip([100 + r, 100], rings))
        assert scores == sorted(scores, reverse=True)


class TestMakeRadialChannel:
    def test_centre_is_zero(self):
        rings = _fake_rings(cx=256, cy=256, r_inner=5, r_outer=100)
        ch = make_radial_channel(rings, scale=1.0, pad_x=0, pad_y=0, size=512)
        assert ch[256, 256] == pytest.approx(0.0, abs=0.05)

    def test_shape(self):
        rings = _fake_rings()
        ch = make_radial_channel(rings, scale=1.0, pad_x=0, pad_y=0, size=64)
        assert ch.shape == (64, 64)

    def test_values_clamped(self):
        rings = _fake_rings(cx=32, cy=32, r_inner=5, r_outer=10)
        ch = make_radial_channel(rings, scale=1.0, pad_x=0, pad_y=0, size=64)
        assert ch.min() >= 0.0
        assert ch.max() <= 2.0


# ── regression: ReplayCompose radial-channel replay ───────────────────────────

class TestSpatialAugReplay:
    """Regression for the label_fields errors when replaying on the radial channel."""

    def _make_spatial(self):
        return A.ReplayCompose(
            [A.HorizontalFlip(p=1.0)],
            keypoint_params=A.KeypointParams(
                format="xy",
                label_fields=["kp_kinds", "kp_arrow_idxs"],
                remove_invisible=True,
            ),
        )

    def test_replay_does_not_raise(self):
        spatial = self._make_spatial()
        img = np.zeros((512, 512, 3), dtype=np.uint8)

        res = spatial(
            image=img,
            keypoints=[(100.0, 200.0), (300.0, 400.0)],
            kp_kinds=["tip", "nock"],
            kp_arrow_idxs=[0, 0],
        )

        # Replay on radial channel — this previously raised ValueError
        radial = np.random.rand(512, 512).astype(np.float32)
        rad_rgb = np.stack([radial] * 3, axis=-1)
        rad_res = A.ReplayCompose.replay(
            res["replay"], image=rad_rgb,
            keypoints=[], kp_kinds=[], kp_arrow_idxs=[],
        )
        assert rad_res["image"].shape == (512, 512, 3)

    def test_replay_applies_same_flip(self):
        """Radial channel must receive the same spatial transform as the image."""
        spatial = self._make_spatial()
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:, :256] = 255  # left half white

        res = spatial(image=img, keypoints=[], kp_kinds=[], kp_arrow_idxs=[])
        # After a guaranteed flip, the right half should be white
        assert res["image"][:, 256:].mean() > 200

        # Radial channel with same left-half-bright pattern
        radial = np.zeros((512, 512, 3), dtype=np.uint8)
        radial[:, :256] = 255
        rad_res = A.ReplayCompose.replay(
            res["replay"], image=radial,
            keypoints=[], kp_kinds=[], kp_arrow_idxs=[],
        )
        assert rad_res["image"][:, 256:].mean() > 200


# ── integration: ArrowDataset.__getitem__ (mocked DB) ────────────────────────

def _make_dataset(tmp_path, n_arrows=2, augment=True):
    """Build an ArrowDataset with a mocked DB and a synthetic image on disk."""
    img_w, img_h = 400, 300
    img = Image.new("RGB", (img_w, img_h), color=(128, 64, 32))
    fname = "test_img.jpg"
    img.save(tmp_path / fname)

    rings = _fake_rings(cx=img_w / 2, cy=img_h / 2, r_inner=10, r_outer=min(img_w, img_h) / 2 - 5)
    arrows = [
        {"tip": [img_w / 2 + i * 15, img_h / 2 + i * 10], "nock": [img_w / 2 + i * 15 - 30, img_h / 2 + i * 10 - 20]}
        for i in range(n_arrows)
    ]

    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = [(fname, arrows, rings)]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    with patch("dataset.psycopg2.connect", return_value=mock_conn):
        ds = ArrowDataset(
            db_url="postgresql://dummy/dummy",
            images_dir=str(tmp_path),
            augment=augment,
            val_split=0.0,
            is_val=False,
        )
    return ds


class TestArrowDatasetGetitem:
    def test_output_shapes_no_augment(self, tmp_path):
        ds = _make_dataset(tmp_path, n_arrows=2, augment=False)
        assert len(ds) == 1
        sample = ds[0]
        assert sample["image"].shape    == (4, INPUT_SIZE, INPUT_SIZE)
        assert sample["tip_hm"].shape   == (1, HEATMAP_SIZE, HEATMAP_SIZE)
        assert sample["nock_hm"].shape  == (1, HEATMAP_SIZE, HEATMAP_SIZE)
        assert sample["score_map"].shape == (HEATMAP_SIZE, HEATMAP_SIZE)

    def test_output_shapes_with_augment(self, tmp_path):
        ds = _make_dataset(tmp_path, n_arrows=2, augment=True)
        sample = ds[0]
        assert sample["image"].shape    == (4, INPUT_SIZE, INPUT_SIZE)
        assert sample["tip_hm"].shape   == (1, HEATMAP_SIZE, HEATMAP_SIZE)
        assert sample["nock_hm"].shape  == (1, HEATMAP_SIZE, HEATMAP_SIZE)
        assert sample["score_map"].shape == (HEATMAP_SIZE, HEATMAP_SIZE)

    def test_heatmap_has_peaks(self, tmp_path):
        ds = _make_dataset(tmp_path, n_arrows=2, augment=False)
        sample = ds[0]
        assert sample["tip_hm"].max().item() > 0.9

    def test_radial_channel_range(self, tmp_path):
        ds = _make_dataset(tmp_path, n_arrows=1, augment=False)
        sample = ds[0]
        rad = sample["image"][3]  # 4th channel
        assert rad.min().item() >= 0.0
        assert rad.max().item() <= 2.0 + 1e-4

    def test_meta_fields_present(self, tmp_path):
        ds = _make_dataset(tmp_path, n_arrows=1, augment=False)
        sample = ds[0]
        for key in ("filename", "scale", "pad_x", "pad_y", "orig_w", "orig_h"):
            assert key in sample["meta"], f"missing meta key: {key}"

    def test_augment_replay_no_error(self, tmp_path):
        """Regression: replay on radial channel must not raise label_fields error."""
        ds = _make_dataset(tmp_path, n_arrows=3, augment=True)
        for _ in range(5):   # run several times to hit different random augmentations
            ds[0]
