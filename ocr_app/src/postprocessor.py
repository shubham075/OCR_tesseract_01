"""Postprocessor module to clean and normalize raw OCR text."""

import re
import unicodedata
from dataclasses import dataclass
from typing import List
from src.ocr_engine import PageResult


@dataclass
class PostprocessorConfig:
    """Configuration settings for text cleaning and postprocessing."""
    remove_extra_whitespace:  bool = True
    remove_empty_lines:       bool = False   # keep paragraph spacing
    fix_hyphenation:          bool = True    # rejoin words split across lines
    normalize_unicode:        bool = True    # NFC normalization
    strip_control_chars:      bool = True


class TextPostprocessor:
    """Performs text normalization, whitespace cleanup, and structural formatting on OCR text."""

    def __init__(self, config: PostprocessorConfig) -> None:
        """Initializes the postprocessor.

        Args:
            config: PostprocessorConfig instance.
        """
        self.config = config

    def clean(self, text: str) -> str:
        """Cleans a raw text block based on active configuration flags.

        Args:
            text: Raw extracted OCR text.

        Returns:
            Cleaned and normalized text.
        """
        # 1. Normalize Unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # 2. Strip Control Characters (keep tabs and newlines/carriage returns)
        if self.config.strip_control_chars:
            text = "".join(
                ch for ch in text 
                if unicodedata.category(ch) != "Cc" or ch in ("\n", "\t", "\r")
            )

        # 3. Clean Whitespace
        if self.config.remove_extra_whitespace:
            lines = text.splitlines()
            cleaned_lines = []
            for line in lines:
                # Replace multiple spaces/tabs with single space
                line_cleaned = re.sub(r"[ \t]+", " ", line).strip()
                if not line_cleaned and self.config.remove_empty_lines:
                    continue
                cleaned_lines.append(line_cleaned)
            text = "\n".join(cleaned_lines)

        # 4. Fix hyphenation (only when remove_extra_whitespace is also True)
        if self.config.fix_hyphenation and self.config.remove_extra_whitespace:
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2\n", text)

        return text

    def clean_batch(self, results: List[PageResult]) -> List[PageResult]:
        """Applies text cleaning to a batch of PageResult objects in-place.

        Args:
            results: List of PageResult objects.

        Returns:
            The list of modified PageResult objects.
        """
        for res in results:
            res.raw_text = self.clean(res.raw_text)
        return results

    def merge_to_document(self, results: List[PageResult]) -> str:
        """Combines all pages of a document into a single text output with page headers.

        Args:
            results: List of PageResult objects.

        Returns:
            A single formatted document string.
        """
        pages = []
        for res in results:
            # page_number is 0-indexed, display as 1-indexed in header
            pages.append(f"[--- Page {res.page_number + 1} ---]\n{res.raw_text}")
        return "\n\n".join(pages)
