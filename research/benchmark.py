"""Transparent benchmark metrics; abstentions are counted as incorrect."""
from __future__ import annotations

def evaluate(predictions: list[dict], labels: list[int]) -> dict:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have equal length")
    n = len(labels)
    if not n:
        return {"n": 0, "accuracy": None, "mae": None, "abstention_rate": None}
    guessed = [p.get("predicted_trl") for p in predictions]
    covered = [(p, y) for p, y in zip(guessed, labels) if isinstance(p, int)]
    return {
        "n": n,
        "accuracy": sum(p == y for p, y in zip(guessed, labels)) / n,
        "mae": sum(abs(p-y) for p,y in covered) / len(covered) if covered else None,
        "coverage": len(covered) / n,
        "abstention_rate": 1 - len(covered) / n,
        "note": "MAE is conditional on non-abstained predictions; always report coverage",
    }
