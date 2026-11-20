"""Offline binary evidence classifier trained on target-specific criterion pairs."""

LABELS = {0: "insufficient", 1: "supports"}


def criterion_query(target, hypothesis):
    return f"Assessment target: {target}\nCriterion: {hypothesis}"


class ModernBertEvidence:
    def __init__(self, checkpoint, device, *, tokenizer_path=None):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if config.model_type != "modernbert" or config.id2label != LABELS:
            raise ValueError("Expected a trained ModernBERT evidence checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or checkpoint, local_files_only=True)
        if tokenizer_path is not None:
            trained_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            if (self.tokenizer.get_vocab() != trained_tokenizer.get_vocab()
                    or self.tokenizer.backend_tokenizer.to_str() != trained_tokenizer.backend_tokenizer.to_str()
                    or self.tokenizer.special_tokens_map != trained_tokenizer.special_tokens_map):
                # Use the exact training tokenizer if the packaged base tokenizer differs.
                self.tokenizer = trained_tokenizer
                import logging
                logging.getLogger(__name__).warning("Using the offline tokenizer shipped with the fine-tuned checkpoint.")
        model, details = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, config=config, output_loading_info=True,
        )
        if details["missing_keys"] or details.get("mismatched_keys"):
            raise ValueError("ModernBERT checkpoint has missing or incompatible weights")
        self.model = model.to(device).eval()
        self.device = device
        self.torch = torch

    def support(self, target, hypothesis, passages, remaining_ms, reserve_ms):
        scores = []
        query = criterion_query(target, hypothesis)
        for offset in range(0, len(passages), 4):
            if remaining_ms() <= reserve_ms:
                return None
            batch = passages[offset:offset + 4]
            inputs = self.tokenizer(
                [query] * len(batch), [p.text for p in batch],
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits
                scores.extend(logits.softmax(dim=-1)[:, 1].tolist())
        return scores if remaining_ms() > reserve_ms else None
