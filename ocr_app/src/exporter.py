"""Exporter module to convert OCR results to TXT, JSON, and searchable PDF formats."""

import json
import os
from typing import List
from fpdf import FPDF
from src.ocr_engine import PageResult
from src.postprocessor import TextPostprocessor


class OCRPDF(FPDF):
    """Custom FPDF class to add footer styling for page numbers."""

    def footer(self) -> None:
        """Adds a page number footer to each PDF page."""
        self.set_y(-15)
        # Check if DejaVu font was added, fallback to Helvetica if not
        if "dejavu" in self.fonts:
            self.set_font("DejaVu", "", 9)
        else:
            self.set_font("Helvetica", "I", 9)
        
        # Display page number centered
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def export_txt(results: List[PageResult], postprocessor: TextPostprocessor) -> bytes:
    """Converts OCR results to UTF-8 encoded plain text with BOM.

    Args:
        results: List of PageResult objects.
        postprocessor: TextPostprocessor instance to handle document merging.

    Returns:
        Bytes representing UTF-8-BOM encoded text.
    """
    merged_text = postprocessor.merge_to_document(results)
    return merged_text.encode("utf-8-sig")


def export_json(results: List[PageResult], original_filename: str = "unknown") -> bytes:
    """Converts OCR results to a structured JSON format.

    Args:
        results: List of PageResult.
        original_filename: The name of the processed PDF file.

    Returns:
        Bytes containing the UTF-8 encoded JSON structure.
    """
    lang = results[0].language if results else "eng"
    data = {
        "source_file": original_filename,
        "total_pages": len(results),
        "language": lang,
        "pages": [
            {
                "page_number": res.page_number + 1,  # 1-indexed for display
                "text": res.raw_text,
                "confidence": round(res.confidence, 1),
                "psm_used": res.psm_used,
                "processing_ms": round(res.processing_ms, 1),
            }
            for res in results
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def export_pdf(results: List[PageResult], original_filename: str) -> bytes:
    """Generates a text-layer PDF with OCR text mapped on each page.

    Args:
        results: List of PageResult.
        original_filename: Name of the original source document.

    Returns:
        Bytes representing the generated PDF.
    """
    pdf = OCRPDF()
    pdf.set_title(f"OCR Export - {original_filename}")
    
    # Locate DejaVuSans.ttf font
    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    
    font_added = False
    if os.path.exists(font_path):
        try:
            pdf.add_font("DejaVu", "", font_path)
            font_added = True
        except Exception:
            pass  # Fall back to standard Helvetica on failure
            
    for res in results:
        pdf.add_page()
        if font_added:
            pdf.set_font("DejaVu", size=11)
        else:
            pdf.set_font("Helvetica", size=11)
            
        # Write OCR text to the page
        # Using multi_cell to handle line breaks and text wrapping automatically
        pdf.multi_cell(0, 8, res.raw_text)
        
    return bytes(pdf.output())
