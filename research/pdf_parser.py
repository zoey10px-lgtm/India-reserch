"""Page-grounded PDF extraction for offline evaluation.

Preserves the exact PDF page index, and section-adjacent text windows.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

from .engine import Evidence

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
SPACES = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Page:
    document_id: str
    number: int
    text: str


def read_pdf(path: str, document_id: str, remaining_ms=None, safety_ms: int = 18000, enable_ocr: bool = True) -> list[Page]:
    if remaining_ms is None:
        remaining_ms = lambda: 180000
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        result = []
        try:
            for idx in range(len(pdf)):
                if remaining_ms() < safety_ms:
                    break
                page = pdf[idx]
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded() or ""
                finally:
                    textpage.close()
                    page.close()
                result.append(Page(document_id, idx + 1, text))
        finally:
            pdf.close()
        if enable_ocr and result and remaining_ms() >= 35000:
            from .ocr import selective_ocr
            corrected = selective_ocr(path, [p.text for p in result], remaining_ms=remaining_ms)
            result = [Page(document_id, p.page, p.text) for p in corrected]
        return result
    except ImportError:
        from pypdf import PdfReader
        reader = PdfReader(path)
        result = []
        for idx, page in enumerate(reader.pages):
            if remaining_ms() < safety_ms:
                break
            result.append(Page(document_id, idx + 1, page.extract_text() or ""))
        return result


def extract_windows(pages: list[Page], max_chars: int = 650, min_chars: int = 20) -> list[Evidence]:
    """Exact-span chunks: every output is a verbatim substring of page text.

    Preserve whitespace and offsets. No paraphrasing or cleaned quotations.
    """
    evidence = []
    for page in pages:
        text = page.text or ""
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + max_chars)
            if end < len(text):
                split = max(text.rfind("\n", offset + max_chars // 2, end),
                            text.rfind(". ", offset + max_chars // 2, end))
                if split > offset:
                    end = split + (1 if text[split] == "\n" else 2)
            if end <= offset:
                end = min(len(text), offset + max_chars)
            segment = text[offset:end]
            if len(segment.strip()) >= min_chars:
                evidence.append(Evidence(page.number, segment, source_id=page.document_id))
            offset = end
    return evidence
