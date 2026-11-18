"""Offline two-stage semantic retrieval: bi-encoder top 20, cross-encoder top 5.

Both paths must refer to approved locally installed models. No network.
"""
from __future__ import annotations
from pathlib import Path
import os

class TwoStageRetriever:
    def __init__(self, encoder_path: str, reranker_path: str, device="cpu",
                 candidate_k=20, final_k=5):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer, CrossEncoder
        if not Path(encoder_path).is_dir() or not Path(reranker_path).is_dir():
            raise FileNotFoundError("Both approved local encoder and reranker paths are required")
        self.encoder = SentenceTransformer(encoder_path, device=device, local_files_only=True)
        self.reranker = CrossEncoder(reranker_path, device=device, local_files_only=True)
        self.candidate_k = candidate_k
        self.final_k = final_k

    def retrieve(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        if not passages:
            return []
        import numpy as np
        vectors = self.encoder.encode([query] + passages, batch_size=32,
                                      normalize_embeddings=True, show_progress_bar=False)
        sims = np.asarray(vectors[1:]) @ np.asarray(vectors[0])
        indices = np.argsort(-sims)[:self.candidate_k].tolist()
        pairs = [(query, passages[i]) for i in indices]
        scores = self.reranker.predict(pairs, batch_size=16)
        ranked = sorted(zip(indices, scores), key=lambda x: -float(x[1]))
        return [(int(i), float(score)) for i, score in ranked[:self.final_k]]
