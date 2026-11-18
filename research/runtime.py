"""Shared offline PDF assessment entry point for the CLI and participant adapter."""
from .offline_nli import OfflineNLI
from .pdf_parser import extract_windows, read_pdf
from .production import EvidenceEngine, Passage
from .two_stage_retrieval import TwoStageRetriever


class ModelInitializationError(RuntimeError):
    """A required offline checkpoint or its runtime could not be initialized."""


class _NLIAdapter:
    def __init__(self, model):
        self.model = model

    def predict_batch(self, pairs):
        return [
            (item.entailment, item.contradiction, item.neutral)
            for item in self.model.predict_batch(pairs)
        ]


def load_engine(paths, *, device="cpu", encoder_query_prefix=""):
    for role, path in paths.items():
        if not path.is_dir():
            raise ModelInitializationError(f"{role} model directory is missing: {path}")
    try:
        nli = OfflineNLI(str(paths["nli"]), device=device)
    except Exception as exc:
        raise ModelInitializationError(f"NLI failed to load from {paths['nli']}: {exc}") from exc
    try:
        retriever = TwoStageRetriever(
            str(paths["encoder"]), str(paths["reranker"]), device=device,
            query_prefix=encoder_query_prefix,
        )
    except Exception as exc:
        raise ModelInitializationError(
            f"Retrieval models failed to load (encoder={paths['encoder']}, "
            f"reranker={paths['reranker']}): {exc}"
        ) from exc
    return EvidenceEngine(_NLIAdapter(nli), retriever)


def assess_pdfs(engine, documents, criteria, target, *, remaining_ms, quality_caps=None):
    passages = []
    for document_id, path in documents:
        if remaining_ms() < 30000:
            return engine.abstain(criteria, "Time budget too low to read all documents", budget_exhausted=True)
        pages = read_pdf(str(path), str(document_id), remaining_ms=remaining_ms)
        passages.extend(Passage(e.source_id, e.page, e.text) for e in extract_windows(pages))
    return engine.evaluate(
        passages, criteria, target, remaining_ms=remaining_ms, quality_caps=quality_caps,
    )
