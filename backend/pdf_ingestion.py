"""
pdf_ingestion.py
----------------
Extracts text from patient source-note PDFs.
Strategy:
  1. Try PyMuPDF (fitz) for native text extraction.
  2. If a page yields < MIN_CHARS characters (likely scanned), fall back to
     Tesseract via pdf2image.  If Tesseract is unavailable, try EasyOCR.
  3. All failures are caught and returned as structured errors — never raised
     silently so the agent loop always knows what happened.
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdf2image

logger = logging.getLogger(__name__)

MIN_CHARS = 50          # threshold below which a page is treated as scanned
MAX_PDF_SIZE_MB = 50    # safety guard against huge files


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    page_number: int          # 1-indexed
    text: str
    method: str               # "pymupdf" | "tesseract" | "easyocr" | "empty"
    warning: Optional[str] = None


@dataclass
class IngestionResult:
    file_path: str
    success: bool
    pages: list[PageResult] = field(default_factory=list)
    full_text: str = ""
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_trace(self) -> dict:
        return {
            "file": self.file_path,
            "success": self.success,
            "page_count": len(self.pages),
            "methods_used": list({p.method for p in self.pages}),
            "warnings": self.warnings,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_with_pymupdf(pdf_path: str) -> list[PageResult]:
    """Native text extraction via PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed: pip install pymupdf")

    results: list[PageResult] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            results.append(PageResult(
                page_number=i + 1,
                text=text.strip(),
                method="pymupdf",
            ))
    finally:
        doc.close()
    return results


def _ocr_with_tesseract(pdf_path: str) -> list[PageResult]:
    """OCR fallback using pdf2image + pytesseract."""
    try:
        import pytesseract
        import pdf2image
    except ImportError:
        raise ImportError(
            "tesseract dependencies not installed: pip install pytesseract pdf2image"
        )

    images = pdf2image.convert_from_path(pdf_path, dpi=200)
    results: list[PageResult] = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, config="--psm 6")
        results.append(PageResult(
            page_number=i + 1,
            text=(text or "").strip(),
            method="tesseract",
        ))
    return results


def _ocr_with_easyocr(pdf_path: str) -> list[PageResult]:
    """Second OCR fallback using EasyOCR."""
    try:
        import easyocr
        from pdf2image import convert_from_path
        import numpy as np
    except ImportError:
        raise ImportError(
            "EasyOCR dependencies not installed: pip install easyocr pdf2image"
        )

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    images = pdf2image.convert_from_path(pdf_path, dpi=200)
    results: list[PageResult] = []
    for i, img in enumerate(images):
        arr = np.array(img)
        ocr_result = reader.readtext(arr, detail=0)
        text = "\n".join(ocr_result)
        results.append(PageResult(
            page_number=i + 1,
            text=text.strip(),
            method="easyocr",
        ))
    return results


def _merge_page_results(
    pymupdf_pages: list[PageResult],
    pdf_path: str,
) -> tuple[list[PageResult], list[str]]:
    """
    For pages that look scanned (low char count), attempt OCR and substitute.
    Returns the final page list and any warnings generated.
    """
    warnings: list[str] = []
    scanned_indices = [
        p.page_number for p in pymupdf_pages if len(p.text) < MIN_CHARS
    ]

    if not scanned_indices:
        return pymupdf_pages, warnings

    warnings.append(
        f"Pages {scanned_indices} appear scanned; attempting OCR fallback."
    )

    try:
        ocr_pages = _ocr_with_tesseract(pdf_path)
        ocr_method = "tesseract"
    except Exception as e1:
        warnings.append(f"Tesseract unavailable ({e1}); trying EasyOCR.")
        try:
            ocr_pages = _ocr_with_easyocr(pdf_path)
            ocr_method = "easyocr"
        except Exception as e2:
            warnings.append(f"EasyOCR also unavailable ({e2}). Scanned pages left empty.")
            # Mark those pages explicitly
            for p in pymupdf_pages:
                if p.page_number in scanned_indices:
                    p.method = "empty"
                    p.warning = "Scanned page; OCR unavailable — content unextracted."
            return pymupdf_pages, warnings

    # Substitute OCR text for scanned pages
    ocr_by_page = {p.page_number: p for p in ocr_pages}
    final: list[PageResult] = []
    for p in pymupdf_pages:
        if p.page_number in scanned_indices and p.page_number in ocr_by_page:
            ocr_p = ocr_by_page[p.page_number]
            final.append(PageResult(
                page_number=p.page_number,
                text=ocr_p.text,
                method=ocr_method,
                warning=f"Replaced scanned page with {ocr_method} output.",
            ))
        else:
            final.append(p)

    return final, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(pdf_path: str, retry: int = 2) -> IngestionResult:
    """
    Ingest a single PDF, returning an IngestionResult with extracted text.
    Retries up to `retry` times on transient failures.
    Never raises — all errors are captured in IngestionResult.error.
    """
    path = Path(pdf_path)

    # Basic sanity checks
    if not path.exists():
        return IngestionResult(
            file_path=pdf_path, success=False,
            error=f"File not found: {pdf_path}"
        )
    if not path.suffix.lower() == ".pdf":
        return IngestionResult(
            file_path=pdf_path, success=False,
            error=f"Not a PDF file: {pdf_path}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        return IngestionResult(
            file_path=pdf_path, success=False,
            error=f"File too large ({size_mb:.1f} MB > {MAX_PDF_SIZE_MB} MB limit)."
        )

    last_error = ""
    for attempt in range(1, retry + 2):  # 1-indexed, retry+1 total attempts
        try:
            pymupdf_pages = _extract_with_pymupdf(pdf_path)
            final_pages, warnings = _merge_page_results(pymupdf_pages, pdf_path)
            full_text = "\n\n".join(
                f"[PAGE {p.page_number}]\n{p.text}" for p in final_pages
            )
            result = IngestionResult(
                file_path=pdf_path,
                success=True,
                pages=final_pages,
                full_text=full_text,
                warnings=warnings,
            )
            if attempt > 1:
                result.warnings.append(f"Succeeded on attempt {attempt}.")
            return result

        except Exception as exc:
            last_error = traceback.format_exc()
            logger.warning(
                "PDF ingestion attempt %d/%d failed for %s: %s",
                attempt, retry + 1, pdf_path, exc
            )

    return IngestionResult(
        file_path=pdf_path,
        success=False,
        error=f"All {retry + 1} ingestion attempts failed.\n{last_error}",
    )


def ingest_patient_folder(folder_path: str) -> dict[str, IngestionResult]:
    """
    Ingest all PDFs in a patient folder.
    Returns a dict mapping filename → IngestionResult.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error("Patient folder not found: %s", folder_path)
        return {}

    results: dict[str, IngestionResult] = {}
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in: %s", folder_path)
        return {}

    for pdf_file in pdf_files:
        logger.info("Ingesting: %s", pdf_file.name)
        result = ingest_pdf(str(pdf_file))
        results[pdf_file.name] = result
        if not result.success:
            logger.error("Failed to ingest %s: %s", pdf_file.name, result.error)
        else:
            logger.info(
                "Ingested %s: %d pages via %s",
                pdf_file.name,
                len(result.pages),
                {p.method for p in result.pages},
            )

    return results
