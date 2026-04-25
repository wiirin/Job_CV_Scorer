"""
PDF → text extraction with OCR fallback.

Pipeline:
  1. Open the PDF with **PyMuPDF** (`fitz`) and try the native text layer.
  2. For pages whose text layer is empty/very short (i.e. scanned images),
     render the page to a high-DPI bitmap and run **Tesseract OCR**
     (`pytesseract`) over it.
  3. Concatenate everything into a single string suitable for downstream
     PII removal + LLM scoring.

This module deliberately keeps OCR optional: if `pytesseract` (or the
underlying Tesseract binary) isn't available, we fall back to whatever
the native text layer gives us and emit a warning. That way the rest
of the CV-checker pipeline keeps working for ordinary digital PDFs.

Inspiration: although the user asked about https://github.com/google/langextract,
that library is for *LLM-based structured information extraction*, not OCR.
The right tool for the OCR job is PyMuPDF (text) + Tesseract (raster) —
which is what we use here.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, List, Optional, Union

logger = logging.getLogger(__name__)


PDFSource = Union[bytes, bytearray, str, Path, IO[bytes]]


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────
@dataclass
class PDFExtractionResult:
    """Structured outcome of a PDF extraction run."""

    text: str
    page_count: int = 0
    pages: List[str] = field(default_factory=list)
    method: str = "text"          # "text" | "ocr" | "hybrid" | "empty"
    ocr_pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def ocr_used(self) -> bool:
        return bool(self.ocr_pages)

    @property
    def summary(self) -> str:
        bits = [
            f"{self.page_count} page(s)",
            f"{self.char_count} chars extracted",
            f"method={self.method}",
        ]
        if self.ocr_used:
            bits.append(f"OCR on pages {self.ocr_pages}")
        return ", ".join(bits)


# ──────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────
class PDFTextExtractor:
    """Extract text from a PDF, falling back to OCR for image-only pages.

    Args:
        ocr_language: Tesseract language code(s), e.g. ``"eng"`` or
            ``"eng+tha"``. Make sure the matching ``*.traineddata`` is
            installed for your Tesseract.
        ocr_dpi: Rendering DPI used when rasterizing pages for OCR.
            300 is a good quality/speed default for most CVs.
        min_text_chars_per_page: If the native text layer of a page has
            fewer than this many characters, the page is treated as
            image-only and OCR is attempted.
        force_ocr: If ``True``, skip the native text layer entirely and
            OCR every page (useful for scanned-only PDFs).
    """

    DEFAULT_LANGUAGE = "eng"
    DEFAULT_DPI = 300
    DEFAULT_MIN_TEXT_CHARS = 40

    def __init__(
        self,
        ocr_language: str = DEFAULT_LANGUAGE,
        ocr_dpi: int = DEFAULT_DPI,
        min_text_chars_per_page: int = DEFAULT_MIN_TEXT_CHARS,
        force_ocr: bool = False,
    ):
        self.ocr_language = ocr_language
        self.ocr_dpi = ocr_dpi
        self.min_text_chars_per_page = min_text_chars_per_page
        self.force_ocr = force_ocr

    # ── Public API ─────────────────────────────────────────────
    def extract(self, source: PDFSource) -> PDFExtractionResult:
        """Extract text from any supported PDF source.

        ``source`` may be a path, raw bytes, or a binary file-like object
        (e.g. a Streamlit ``UploadedFile``).
        """
        fitz = self._import_fitz()
        doc = self._open_document(fitz, source)

        pages_text: List[str] = []
        ocr_pages: List[int] = []
        warnings: List[str] = []

        try:
            for page_index, page in enumerate(doc, start=1):
                native_text = "" if self.force_ocr else page.get_text("text").strip()
                use_ocr = (
                    self.force_ocr
                    or len(native_text) < self.min_text_chars_per_page
                )

                if use_ocr:
                    ocr_text, ocr_warning = self._ocr_page(page, page_index)
                    if ocr_warning:
                        warnings.append(ocr_warning)
                    if ocr_text:
                        pages_text.append(ocr_text)
                        ocr_pages.append(page_index)
                        continue

                    if native_text:
                        # OCR failed but we still have a sliver of native text.
                        pages_text.append(native_text)
                    else:
                        pages_text.append("")
                        warnings.append(
                            f"Page {page_index}: no extractable text "
                            f"(image-only page and OCR unavailable)."
                        )
                else:
                    pages_text.append(native_text)
        finally:
            doc.close()

        full_text = "\n\n".join(p for p in pages_text if p).strip()
        method = self._classify_method(pages_text, ocr_pages)

        result = PDFExtractionResult(
            text=full_text,
            page_count=len(pages_text),
            pages=pages_text,
            method=method,
            ocr_pages=ocr_pages,
            warnings=warnings,
        )
        logger.info("PDF extraction: %s", result.summary)
        return result

    # ── Internals ──────────────────────────────────────────────
    @staticmethod
    def _import_fitz():
        try:
            import fitz  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyMuPDF is required for PDF text extraction. "
                "Install it with `pip install pymupdf` (or `uv add pymupdf`)."
            ) from exc
        return fitz

    @staticmethod
    def _open_document(fitz, source: PDFSource):
        if isinstance(source, (str, Path)):
            return fitz.open(str(source))
        if isinstance(source, (bytes, bytearray)):
            return fitz.open(stream=bytes(source), filetype="pdf")
        # Binary file-like: read all bytes (covers Streamlit's UploadedFile).
        try:
            data = source.read()
        except AttributeError as exc:
            raise TypeError(
                f"Unsupported PDF source type: {type(source).__name__}"
            ) from exc
        return fitz.open(stream=data, filetype="pdf")

    def _ocr_page(self, page, page_index: int) -> tuple[str, Optional[str]]:
        """Run OCR on a single PyMuPDF page. Returns (text, warning)."""
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
        except ImportError:
            return "", (
                "OCR skipped: install `pytesseract` and `pillow` (and the "
                "system Tesseract binary) to OCR scanned pages."
            )

        try:
            pix = page.get_pixmap(dpi=self.ocr_dpi, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(
                img, lang=self.ocr_language
            ).strip()
            return text, None
        except pytesseract.TesseractNotFoundError:
            return "", (
                "OCR skipped: the Tesseract binary was not found on PATH. "
                "Install it (e.g. `brew install tesseract` on macOS, "
                "`apt-get install tesseract-ocr` on Debian/Ubuntu)."
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("OCR failed on page %d: %s", page_index, exc)
            return "", f"OCR failed on page {page_index}: {exc}"

    @staticmethod
    def _classify_method(pages_text: List[str], ocr_pages: List[int]) -> str:
        non_empty = sum(1 for p in pages_text if p)
        if non_empty == 0:
            return "empty"
        if not ocr_pages:
            return "text"
        if len(ocr_pages) == non_empty:
            return "ocr"
        return "hybrid"


# ──────────────────────────────────────────────
# Convenience functional API
# ──────────────────────────────────────────────
def extract_text_from_pdf(
    source: PDFSource,
    *,
    ocr_language: str = PDFTextExtractor.DEFAULT_LANGUAGE,
    ocr_dpi: int = PDFTextExtractor.DEFAULT_DPI,
    force_ocr: bool = False,
) -> PDFExtractionResult:
    """One-shot helper that mirrors :class:`PDFTextExtractor.extract`."""
    extractor = PDFTextExtractor(
        ocr_language=ocr_language,
        ocr_dpi=ocr_dpi,
        force_ocr=force_ocr,
    )
    return extractor.extract(source)


__all__ = [
    "PDFExtractionResult",
    "PDFTextExtractor",
    "extract_text_from_pdf",
]
