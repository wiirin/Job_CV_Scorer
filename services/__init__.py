"""CV Checker service modules.

This package bundles the building blocks that power the Streamlit UI:

- ``pii_remover``       — GDPR-compliant PII removal (spaCy NER + regex).
- ``relevancy_scorer``  — Claude Sonnet 4.6 (Vertex AI) CV ⇄ JD scorer.
- ``ocr_pdf_to_text``   — PyMuPDF text extraction with Tesseract OCR fallback.
- ``cv_cover_letter``   — Claude-powered cover letter feedback (CV + JD aware).
"""
from .pii_remover import (
    AnonymizationResult,
    PIIEntity,
    PIIRemovalService,
)
from .relevancy_scorer import (
    PROVIDER_CLAUDE_API,
    PROVIDER_VERTEX,
    SUPPORTED_PROVIDERS,
    RelevancyScore,
    RelevancyScorer,
)
from .ocr_pdf_to_text import (
    PDFExtractionResult,
    PDFTextExtractor,
    extract_text_from_pdf,
)
from .cv_cover_letter import (
    CoverLetterAnalyzer,
    CoverLetterFeedback,
)

__all__ = [
    "AnonymizationResult",
    "PIIEntity",
    "PIIRemovalService",
    "RelevancyScore",
    "RelevancyScorer",
    "PROVIDER_VERTEX",
    "PROVIDER_CLAUDE_API",
    "SUPPORTED_PROVIDERS",
    "PDFExtractionResult",
    "PDFTextExtractor",
    "extract_text_from_pdf",
    "CoverLetterAnalyzer",
    "CoverLetterFeedback",
]
