"""pytest configuration and shared fixtures for ArcheryCounter Python tests."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psycopg2
import pytest

ROOT = Path(__file__).parent.parent.parent.parent
IMAGES_DIR = ROOT / "images"
NN_DIR = Path(__file__).parent


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        dbname=os.getenv("DB_NAME", "postgres"),
    )


@pytest.fixture(scope="session")
def db_conn():
    conn = _db_connect()
    yield conn
    conn.close()


# ── ONNX model fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def arrow_session():
    model_path = NN_DIR / "arrow_detector_fp32.onnx"
    if not model_path.exists():
        pytest.skip(f"Arrow model not found: {model_path}")
    return ort.InferenceSession(str(model_path))


@pytest.fixture(scope="session")
def boundary_session():
    model_path = NN_DIR / "boundary_detector_v2.onnx"
    if not model_path.exists():
        pytest.skip(f"Boundary model not found: {model_path}")
    return ort.InferenceSession(str(model_path))


# ── Test parametrisation ─────────────────────────────────────────────────────

def _get_annotated_filenames() -> list[str]:
    """Return filenames with valid (non-corrupt) annotations from DB."""
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT filename FROM annotations
            WHERE paper_boundary IS NOT NULL
              AND paper_boundary != '[]'::jsonb
            ORDER BY filename
        """)
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "annotated_filename" in metafunc.fixturenames:
        filenames = _get_annotated_filenames()
        metafunc.parametrize(
            "annotated_filename",
            filenames if filenames else ["__no_images__"],
            scope="session",
        )
