"""Evidence graph primitives for explainability and contradiction tracing.

Edges are conservative: exact duplicates and opposing NLI polarities for the
same criterion/target. Semantic coreference is future work, not claimed here.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Edge:
    source: int
    destination: int
    relation: str

def build_edges(records: list[dict]) -> list[Edge]:
    edges = []
    for i, a in enumerate(records):
        for j in range(i + 1, len(records)):
            b = records[j]
            if a.get("text", "").strip().casefold() == b.get("text", "").strip().casefold():
                edges.append(Edge(i, j, "duplicate"))
            elif (a.get("entailment", 0) >= 0.7 and b.get("contradiction", 0) >= 0.7) or (
                b.get("entailment", 0) >= 0.7 and a.get("contradiction", 0) >= 0.7
            ):
                edges.append(Edge(i, j, "conflict_candidate"))
    return edges
