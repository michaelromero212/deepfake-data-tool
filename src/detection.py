"""
detection.py — Deepfake detection inference stage.

Runs a lightweight pre-trained detector on each processed sample.
In this project we provide two modes:

  1. MOCK mode (default, no dependencies) — simulates detector behavior
     with realistic score distributions for demo/testing purposes.

  2. ONNX mode (optional) — loads a real EfficientNet-based deepfake
     detector exported to ONNX format. Drop any compatible ONNX model
     into models/ and set FORGE_USE_ONNX=1.

The DetectionResult schema includes model_name and model_version so the
manifest has full provenance — critical for reproducible ML experiments.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from src.schemas import DetectionResult, Label, MediaType

MOCK_MODEL_NAME = "mock-efficientnet-dfdc-v1"
MOCK_MODEL_VERSION = "0.1.0-mock"
ONNX_MODEL_PATH = Path("models/deepfake_detector.onnx")


def _mock_score(label: Label, media_type: MediaType) -> float:
    """
    Generate a realistic mock detection score.
    Real samples cluster low (0.05-0.25), synthetic samples cluster high (0.70-0.95),
    with some noise to simulate real-world model imperfection.
    """
    rng = np.random.default_rng()

    if label == Label.REAL:
        # Real: mostly low scores with occasional false positives
        base = rng.beta(a=1.5, b=8.0)  # skewed toward 0
    elif label == Label.SYNTHETIC:
        # Synthetic: mostly high scores with occasional false negatives
        base = rng.beta(a=8.0, b=1.5)  # skewed toward 1
    else:
        # Unknown: uniform
        base = rng.uniform(0.3, 0.7)

    # Audio and video deepfakes tend to be slightly harder to detect
    if media_type in (MediaType.AUDIO, MediaType.VIDEO):
        base = np.clip(base + rng.normal(0, 0.05), 0.0, 1.0)

    return float(np.clip(base, 0.0, 1.0))


def _onnx_score(processed_path: Path) -> float:
    """
    Run inference with a real ONNX model.
    Expects a 224x224 PNG image (works for image and video frame inputs).
    Audio inputs should be pre-converted to spectrograms.
    """
    try:
        import onnxruntime as ort
        import cv2

        sess = ort.InferenceSession(str(ONNX_MODEL_PATH))
        img = cv2.imread(str(processed_path))
        if img is None:
            raise ValueError(f"Cannot load image: {processed_path}")

        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW

        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: img})
        logits = result[0][0]

        # Assume binary classifier: [real_logit, fake_logit]
        score = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
        return score

    except ImportError:
        logger.warning("onnxruntime not installed — falling back to mock mode")
        return _mock_score(Label.UNKNOWN, MediaType.IMAGE)


def run_detection(
    processed_path: Path | None,
    label: Label,
    media_type: MediaType,
) -> DetectionResult:
    """
    Run deepfake detection on a processed sample.
    Returns a DetectionResult with score, model provenance, and timing.
    """
    use_onnx = os.getenv("FORGE_USE_ONNX", "0") == "1"

    start = time.perf_counter()

    if use_onnx and processed_path and ONNX_MODEL_PATH.exists():
        score = _onnx_score(processed_path)
        model_name = "onnx-efficientnet-dfdc"
        model_version = "1.0.0"
        logger.debug(f"ONNX inference: score={score:.3f} for {processed_path.name if processed_path else 'unknown'}")
    else:
        score = _mock_score(label, media_type)
        model_name = MOCK_MODEL_NAME
        model_version = MOCK_MODEL_VERSION
        logger.debug(f"Mock inference: score={score:.3f} (label={label.value})")

    elapsed_ms = (time.perf_counter() - start) * 1000

    return DetectionResult(
        model_name=model_name,
        detection_score=round(score, 4),
        model_version=model_version,
        inference_time_ms=round(elapsed_ms, 2),
    )
