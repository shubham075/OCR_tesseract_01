"""OCR engine module wrapping pytesseract and managing OCR execution configuration.

Supported Page Segmentation Modes (PSM):
| PSM | Best For |
|-----|----------|
| 3   | Fully automatic (default for mixed pages) |
| 4   | Single column of text |
| 6   | Uniform block of text |
| 7   | Single text line |
| 8   | Single word |
| 11  | Sparse text — good for forms with scattered fields |
| 13  | Raw line — useful for handwriting one line at a time |
"""

import time
from dataclasses import dataclass
from typing import Callable, List
import pandas as pd
import PIL.Image
import pytesseract


@dataclass
class OCRConfig:
    """Configuration settings for the OCR execution."""
    language:        str = "eng"
    psm:             int = 3      # Page Segmentation Mode (Tesseract --psm)
    oem:             int = 3      # OCR Engine Mode (LSTM + legacy)
    dpi:             int = 300
    extra_config:    str = ""     # any extra --tessdata-dir or custom flags


@dataclass
class PageResult:
    """Contains OCR results and metadata for a single document page."""
    page_number:   int
    raw_text:      str
    word_data:     pd.DataFrame   # from pytesseract.image_to_data()
    confidence:    float          # mean confidence of words with conf > 0
    language:      str
    psm_used:      int
    processing_ms: float


class OCREngine:
    """Wraps pytesseract to execute OCR on images and validate the local installation."""

    def __init__(self, config: OCRConfig) -> None:
        """Initializes OCREngine with a configuration.

        Args:
            config: OCRConfig instance.
        """
        self.config = config

    def validate_installation(self) -> None:
        """Checks if Tesseract binary is installed and accessible.

        Raises:
            RuntimeError: If Tesseract binary is not found in system PATH.
        """
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "Tesseract binary not found. Install it and add to PATH."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Tesseract is not configured properly or inaccessible: {str(e)}"
            ) from e

    def validate_languages(self, lang_str: str) -> List[str]:
        """Validates if requested languages are installed.

        Args:
            lang_str: Language string formatted as 'eng+hin'.

        Returns:
            A list of missing language package codes.
        """
        try:
            installed = pytesseract.get_languages()
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve installed languages: {str(e)}")

        requested = lang_str.split("+")
        missing = [lang for lang in requested if lang not in installed]
        return missing

    def run_page(self, image: PIL.Image.Image, page_number: int) -> PageResult:
        """Executes OCR on a single PIL Image.

        Args:
            image: Processed PIL Image.
            page_number: 0-indexed page number of the source document.

        Returns:
            PageResult containing extracted text, word level data, and performance stats.
        """
        start_time = time.perf_counter()

        # Build Tesseract configuration string
        tess_config = f"--psm {self.config.psm} --oem {self.config.oem}"
        if self.config.extra_config:
            tess_config += f" {self.config.extra_config}"

        # Run OCR string extraction
        try:
            raw_text = pytesseract.image_to_string(
                image,
                lang=self.config.language,
                config=tess_config
            )
        except Exception as e:
            # Handle potential runtime tesseract errors
            if "TesseractNotFoundError" in str(type(e)):
                raise RuntimeError("Tesseract binary not found. Install it and add to PATH.") from e
            raise RuntimeError(f"OCR execution failed: {str(e)}") from e

        # Run OCR data extraction for word level details
        try:
            word_data = pytesseract.image_to_data(
                image,
                lang=self.config.language,
                config=tess_config,
                output_type=pytesseract.Output.DATAFRAME
            )
        except Exception:
            # Fallback to empty DataFrame if image_to_data fails
            word_data = pd.DataFrame(columns=["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"])

        # Calculate average confidence
        # Tesseract outputs conf = -1 for non-word blocks
        if "conf" in word_data.columns and not word_data.empty:
            valid_conf = word_data[word_data["conf"] > -1]["conf"]
            confidence = float(valid_conf.mean()) if not valid_conf.empty else 0.0
        else:
            confidence = 0.0

        # Ensure confidence is a valid float (not NaN)
        if pd.isna(confidence):
            confidence = 0.0

        end_time = time.perf_counter()
        processing_ms = (end_time - start_time) * 1000

        return PageResult(
            page_number=page_number,
            raw_text=raw_text,
            word_data=word_data,
            confidence=confidence,
            language=self.config.language,
            psm_used=self.config.psm,
            processing_ms=processing_ms
        )

    def run_batch(
        self,
        images: List[PIL.Image.Image],
        progress_callback: Callable[[int, int], None] | None = None
    ) -> List[PageResult]:
        """Runs OCR on a list of images.

        Args:
            images: List of PIL Image objects.
            progress_callback: Optional callback receiving (current_index, total_images).

        Returns:
            List of PageResult objects.
        """
        results = []
        total = len(images)
        for i, img in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)
            res = self.run_page(img, page_number=i)
            results.append(res)
        return results
