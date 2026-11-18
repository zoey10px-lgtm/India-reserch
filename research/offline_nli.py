"""Offline semantic inference with an actual Hugging Face NLI checkpoint.

Requires a LOCAL, organizer-approved sequence-classification model path.
Never downloads model weights or silently substitutes keyword inference.
"""
from __future__ import annotations
from pathlib import Path
import os

from .engine import Inference


class OfflineNLI:
    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 384):
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"NLI model directory not found: {path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        label_map = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.indices = {}
        for name in ("entailment", "contradiction", "neutral"):
            found = [i for label, i in label_map.items() if name in label]
            if len(found) != 1:
                raise ValueError(f"Cannot determine NLI label mapping for {name}: {label_map}")
            self.indices[name] = found[0]

    def predict_batch(self, pairs: list[tuple[str, str]], batch_size: int = 12) -> list[Inference]:
        results: list[Inference] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = self.tokenizer(
                [p for p, _ in batch], [h for _, h in batch],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                probs = self.model(**encoded).logits.softmax(-1).float().cpu().tolist()
            for row in probs:
                results.append(Inference(
                    entailment=row[self.indices["entailment"]],
                    contradiction=row[self.indices["contradiction"]],
                    neutral=row[self.indices["neutral"]],
                ))
        return results

    def predict(self, premise: str, hypothesis: str) -> Inference:
        return self.predict_batch([(premise, hypothesis)])[0]
