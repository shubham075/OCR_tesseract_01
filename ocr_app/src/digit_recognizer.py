"""Handwritten digit recognizer using ONNX Runtime and MNIST-trained CNN model.

This module provides an offline, local CNN-based corrector that targets
handwritten English numerals (0-9) in regions where Tesseract reported
low confidence. It downloads the MNIST-8 ONNX model on first use and
caches it locally inside src/models/.
"""

import os
import re
import logging
import urllib.request
from typing import List, Tuple

import numpy as np
import PIL.Image
import PIL.ImageOps

logger = logging.getLogger(__name__)

# --- Model configuration ---
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "mnist-8.onnx")
_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mnist/model/mnist-8.onnx"
)


def _download_model() -> bool:
    """Downloads the MNIST-8 ONNX model to src/models/ if not already present.

    Returns:
        True if model is ready (already exists or successfully downloaded),
        False if download failed.
    """
    if os.path.isfile(_MODEL_PATH):
        return True

    os.makedirs(_MODEL_DIR, exist_ok=True)

    try:
        logger.info("Downloading MNIST-8 ONNX model (~26 KB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info(f"Model saved to: {_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download ONNX model: {e}")
        # Clean up broken partial download
        if os.path.isfile(_MODEL_PATH):
            os.remove(_MODEL_PATH)
        return False


def _preprocess_digit_image(img: PIL.Image.Image) -> np.ndarray:
    """Converts a PIL image crop to a normalized 1x1x28x28 ONNX input tensor.

    The MNIST model expects:
    - Grayscale, 28x28 pixels
    - Float32 values normalized to [0.0, 1.0]
    - Shape: (1, 1, 28, 28) — batch x channel x height x width
    - White digit on black background (MNIST standard)

    Args:
        img: PIL Image of the digit region (any size, any mode).

    Returns:
        numpy.ndarray of shape (1, 1, 28, 28), dtype float32.
    """
    # Convert to grayscale
    gray = img.convert("L")

    # Resize to 28x28 using Lanczos resampling
    resized = gray.resize((28, 28), PIL.Image.LANCZOS)

    # MNIST expects white digits on black background (invert if scanned page is inverted)
    arr = np.array(resized, dtype=np.float32)

    # If background appears light (mean > 127), invert to match MNIST convention
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Reshape to (1, 1, 28, 28)
    return arr.reshape(1, 1, 28, 28)


class DigitRecognizer:
    """Offline CNN-based handwritten digit recognizer using ONNX Runtime.

    Uses the MNIST-8 ONNX model from the ONNX Model Zoo for inference.
    The model is approximately 26 KB and is downloaded once on first use
    into src/models/mnist-8.onnx.
    """

    def __init__(self, confidence_threshold: float = 60.0) -> None:
        """Initializes the DigitRecognizer.

        Args:
            confidence_threshold: Tesseract confidence percentage below which
                digit regions are re-evaluated by the CNN. Default: 60.0.

        Raises:
            ImportError: If onnxruntime is not installed.
            RuntimeError: If the ONNX model cannot be downloaded or loaded.
        """
        try:
            import onnxruntime as ort
            self._ort = ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for digit recognition. "
                "Install it with: pip install onnxruntime"
            ) from e

        self.confidence_threshold = confidence_threshold
        self._session = None

    def _ensure_session(self) -> None:
        """Lazily loads the ONNX Runtime InferenceSession.

        Downloads the model on first call if not already present.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if self._session is not None:
            return

        if not _download_model():
            raise RuntimeError(
                "MNIST ONNX model could not be downloaded. "
                "Check your internet connection and try again."
            )

        try:
            self._session = self._ort.InferenceSession(
                _MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def predict_digit(self, img: PIL.Image.Image) -> Tuple[int, float]:
        """Classifies a single digit image using the ONNX CNN model.

        Args:
            img: PIL Image containing exactly one handwritten digit.

        Returns:
            A tuple of (predicted_digit: int, confidence: float).
            Confidence is in range [0.0, 1.0].
        """
        self._ensure_session()

        tensor = _preprocess_digit_image(img)

        # Run ONNX inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})

        # Output shape: (1, 10) — raw logits for digits 0-9
        logits = outputs[0][0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # numerically stable
        probs = exp_logits / exp_logits.sum()

        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        return predicted, confidence

    def correct_digit_regions(
        self,
        original_text: str,
        word_data,
        page_image: PIL.Image.Image,
        min_cnn_confidence: float = 0.6,
    ) -> Tuple[str, List[dict]]:
        """Scans Tesseract word data for low-confidence digit tokens and corrects them.

        Strategy:
        1. Iterate over all words in word_data where Tesseract confidence < threshold
           AND the word text appears to be a digit or a digit-containing token.
        2. Crop that word's bounding box from the original page image.
        3. Run CNN prediction on the cropped region.
        4. If CNN confidence >= min_cnn_confidence AND CNN prediction differs from
           Tesseract output, replace the Tesseract token with the CNN result.

        Args:
            original_text: Raw OCR text string from Tesseract.
            word_data: pandas DataFrame from pytesseract.image_to_data().
            page_image: The full preprocessed PIL Image that was OCR'd.
            min_cnn_confidence: Minimum CNN softmax probability to trust the correction.
                Default: 0.6 (60%).

        Returns:
            A tuple of:
            - corrected_text: str — the corrected text with CNN replacements applied.
            - corrections: list[dict] — log of each correction made, containing
              keys: 'original', 'corrected', 'tesseract_conf', 'cnn_conf', 'position'.
        """
        self._ensure_session()

        corrected_text = original_text
        corrections: List[dict] = []

        if word_data is None or word_data.empty:
            return corrected_text, corrections

        # Filter to words below the threshold that look like digit/numeric tokens
        _digit_pattern = re.compile(r"^[\d/\-\.]+$")

        try:
            low_conf_rows = word_data[
                (word_data["conf"].apply(
                    lambda c: isinstance(c, (int, float)) and 0 <= c < self.confidence_threshold
                )) &
                (word_data["text"].apply(
                    lambda t: isinstance(t, str) and _digit_pattern.match(t.strip()) is not None
                ))
            ]
        except Exception:
            return corrected_text, corrections

        page_w, page_h = page_image.size

        for _, row in low_conf_rows.iterrows():
            try:
                left = int(row["left"])
                top = int(row["top"])
                width = int(row["width"])
                height = int(row["height"])
                tess_text = str(row["text"]).strip()
                tess_conf = float(row["conf"])

                # Skip if bounding box is invalid or too small
                if width < 5 or height < 5:
                    continue
                if left < 0 or top < 0:
                    continue

                # Clamp to image bounds
                right = min(left + width, page_w)
                bottom = min(top + height, page_h)

                # Crop digit region from page image
                crop = page_image.crop((left, top, right, bottom))

                # Predict with CNN — only for single-character digits
                if len(tess_text) == 1 and tess_text.isdigit():
                    predicted, cnn_conf = self.predict_digit(crop)
                    cnn_str = str(predicted)

                    # Only apply correction if CNN is confident and disagrees
                    if cnn_conf >= min_cnn_confidence and cnn_str != tess_text:
                        corrected_text = corrected_text.replace(tess_text, cnn_str, 1)
                        corrections.append({
                            "original": tess_text,
                            "corrected": cnn_str,
                            "tesseract_conf": round(tess_conf, 1),
                            "cnn_conf": round(cnn_conf * 100, 1),
                            "position": (left, top, right, bottom),
                        })

            except Exception as e:
                logger.debug(f"Skipped digit correction for row: {e}")
                continue

        return corrected_text, corrections
