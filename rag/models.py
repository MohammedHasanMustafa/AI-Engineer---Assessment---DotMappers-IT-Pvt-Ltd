from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

REFUSAL_STATUSES = {"insufficient_evidence", "rejected", "blocked"}


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    page_end: int
    chunk_index: int
    text: str
    doc_hash: str
    n_words: int
    injection_flags: list[str] = field(default_factory=list)

    @property
    def pages(self) -> str:
        return f"p.{self.page}" if self.page == self.page_end else f"pp.{self.page}-{self.page_end}"

    @property
    def citation(self) -> str:
        return f"{self.source}, {self.pages}, chunk {self.chunk_id}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    rank: int = 0
    label: str = ""                    
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    fused_score: float = 0.0
    rerank_score: Optional[float] = None
    relevance: float = 0.0           
    prompt_text: str = ""              
    neutralized: int = 0              

@dataclass
class EvidenceItem:
    claim: str
    citations: list[str]               
    labels: list[str]                  
    support: Optional[float] = None    
    supported: Optional[bool] = None


@dataclass
class RAGResponse:
    query: str
    status: str = "insufficient_evidence"   # answered | conflicting | insufficient_evidence | rejected | blocked
    message: str = ""
    evidence: list[EvidenceItem] = field(default_factory=list)
    inference: str = ""
    conflicts: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    generator: str = ""
    security_events: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    groundedness: Optional[float] = None
    relevance: Optional[float] = None
    citation_validity: Optional[float] = None

    @property
    def refused(self) -> bool:
        return self.status in REFUSAL_STATUSES

    @property
    def cited_labels(self) -> list[str]:
        seen: list[str] = []
        for e in self.evidence:
            for label in e.labels:
                if label not in seen:
                    seen.append(label)
        return seen

    def cited_retrieved(self) -> list[RetrievedChunk]:
        by_label = {rc.label: rc for rc in self.retrieved}
        return [by_label[l] for l in self.cited_labels if l in by_label]

    def answer_text(self) -> str:
        """Plain-text rendering used for logging and evaluation."""
        if self.refused:
            head = "Insufficient evidence" if self.status != "rejected" else "Query rejected"
            return f"{head}. {self.message}".strip()
        lines = ["Evidence:"]
        for e in self.evidence:
            lines.append(f"- {e.claim} [{', '.join(e.labels)}]")
        if self.conflicts:
            lines.append(f"Conflicts: {self.conflicts}")
        if self.inference:
            lines.append(f"Inference: {self.inference}")
        return "\n".join(lines)
