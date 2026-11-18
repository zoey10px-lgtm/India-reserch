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


def extract_windows(pages: list[Page], max_chars: int = 650, min_chars: int = 25) -> list[Evidence]:
    """Preserve citations and local context, including short numeric evidence.

    This intentionally does NOT discard the first page, tables or numbers.
    """
    evidence = []
    for page in pages:
        cleaned = SPACES.sub(" ", page.text.replace("\r", "\n"))
        lines = [x.strip() for x in cleaned.splitlines() if x.strip()]
        section = ""
        units = []
        for line in lines:
            if len(line) < 110 and (line.isupper() or re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", line)):
                section = line[:120]
            units.extend(SENTENCE.split(line))
        buffer = ""
        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            if len(buffer) + len(unit) + 1 > max_chars and buffer:
                if len(buffer) >= min_chars:
                    evidence.append(Evidence(page.number, buffer, section=section, source_id=page.document_id))
                buffer = ""
            if len(unit) > max_chars:
                for offset in range(0, len(unit), max_chars):
                    chunk = unit[offset:offset+max_chars]
                    if len(chunk) >= min_chars:
                        evidence.append(Evidence(page.number, chunk, section=section, source_id=page.document_id))
            else:
                buffer = f"{buffer} {unit}".strip()
        if len(buffer) >= min_chars:
            evidence.append(Evidence(page.number, buffer, section=section, source_id=page.document_id))
    return evidence
