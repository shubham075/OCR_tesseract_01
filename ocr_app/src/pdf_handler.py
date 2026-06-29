"""PDF handler module for ingesting, validating, and rendering PDF documents."""

import io
import os
from typing import Callable, Any, Dict, List
import PIL.Image
from pypdf import PdfReader
import pdf2image
from pdf2image.exceptions import PDFInfoNotInstalledError


class PDFHandler:
    """Handles PDF file ingestion, metadata extraction, and page rasterization."""

    def __init__(self, file_bytes: bytes) -> None:
        """Initializes PDFHandler and validates the input bytes.

        Args:
            file_bytes: Raw bytes of the uploaded PDF file.

        Raises:
            ValueError: If the file does not start with %PDF magic bytes.
        """
        if not file_bytes.startswith(b"%PDF"):
            raise ValueError("Uploaded file is not a valid PDF.")
        
        self._file_bytes = file_bytes
        self._load_pdf()

    def _load_pdf(self) -> None:
        """Loads the PDF document using pypdf to extract metadata and page count."""
        try:
            reader = PdfReader(io.BytesIO(self._file_bytes))
            self.page_count = len(reader.pages)
            
            # Extract metadata
            meta = reader.metadata or {}
            self.metadata = {
                "title": str(meta.title) if meta.title else "Unknown",
                "author": str(meta.author) if meta.author else "Unknown",
                "creator": str(meta.creator) if meta.creator else "Unknown",
                "page_count": self.page_count,
            }
        except Exception as e:
            raise ValueError(f"Uploaded file is not a valid PDF: {str(e)}")

    def render_page(self, page_number: int, dpi: int = 300) -> PIL.Image.Image:
        """Renders a specific page of the PDF to a PIL Image.

        Args:
            page_number: 0-indexed index of the page to render.
            dpi: Dots per inch for rasterization quality.

        Returns:
            A PIL Image.Image containing the rendered page.

        Raises:
            IndexError: If page_number is out of bounds.
            RuntimeError: If Poppler is not installed.
        """
        if page_number < 0 or page_number >= self.page_count:
            raise IndexError(f"Page number {page_number} is out of bounds (0-{self.page_count - 1}).")

        try:
            # Check for local Windows poppler binaries
            local_poppler = os.path.join(
                os.path.dirname(__file__), "poppler", "poppler-26.02.0", "Library", "bin"
            )
            poppler_path = local_poppler if os.path.exists(local_poppler) else None

            # Render only the specific page to optimize memory and speed
            images = pdf2image.convert_from_bytes(
                self._file_bytes,
                dpi=dpi,
                first_page=page_number + 1,
                last_page=page_number + 1,
                poppler_path=poppler_path,
            )
            if not images:
                raise RuntimeError(f"Failed to render page {page_number}.")
            return images[0]
        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "poppler-utils is not installed. Run: sudo apt-get install poppler-utils"
            ) from e

    def render_all_pages(
        self,
        dpi: int = 300,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> List[PIL.Image.Image]:
        """Renders all pages of the PDF to a list of PIL Images.

        Args:
            dpi: Dots per inch for rasterization quality.
            progress_callback: Optional callback receiving (current_page, total_pages).

        Returns:
            A list of PIL Image.Image objects.
        """
        images = []
        for i in range(self.page_count):
            if progress_callback:
                progress_callback(i + 1, self.page_count)
            images.append(self.render_page(i, dpi=dpi))
        return images
