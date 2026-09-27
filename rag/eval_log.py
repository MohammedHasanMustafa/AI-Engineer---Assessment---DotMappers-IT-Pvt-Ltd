from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .models import RAGResponse

GREEN, AMBER = 0.75, 0.50


def score_color(value) -> str:
    if value is None:
        return "grey"
    return "green" if value >= GREEN else "amber" if value >= AMBER else "red"


class EvalLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")


def record_from_response(resp: RAGResponse, origin: str, accuracy: float | None = None,
                         accuracy_basis: str = "", question_id: str = "") -> dict:
    """accuracy: ground-truth score for evaluation questions; for ad-hoc UI questions it is
    a proxy = citation validity (share of the model's citations that pointed at real passages)."""
    if accuracy is None and not resp.refused:
        accuracy, accuracy_basis = resp.citation_validity, "citation validity (no ground truth)"
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "origin": origin,
        "question_id": question_id,
        "query": resp.query,
        "status": resp.status,
        "accuracy": None if accuracy is None else round(float(accuracy), 3),
        "accuracy_basis": accuracy_basis,
        "relevance": None if resp.relevance is None else round(resp.relevance, 3),
        "groundedness": resp.groundedness,
        "latency_ms": resp.latency_ms.get("total_ms"),
        "generator": resp.generator,
        "citations": [rc.chunk.chunk_id for rc in resp.cited_retrieved()],
        "security_events": resp.security_events,
        "answer": resp.answer_text()[:2000],
    }
