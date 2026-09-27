from __future__ import annotations

from .models import EvidenceItem, RetrievedChunk
from .text_utils import split_sentences, tokenize


def lexical_support(claim: str, text: str) -> float:
    toks = set(tokenize(claim))
    if not toks:
        return 0.0
    return len(toks & set(tokenize(text))) / len(toks)


def score_groundedness(evidence: list[EvidenceItem], passages: list[RetrievedChunk], embedder,
                       cos_threshold: float = 0.70, lex_threshold: float = 0.60) -> float | None:
    if not evidence:
        return None
    by_label = {rc.label: rc for rc in passages}
    supported = 0
    for item in evidence:
        cited = " ".join(by_label[l].chunk.text for l in item.labels if l in by_label)
        if not cited:
            item.support, item.supported = 0.0, False
            continue
        sents = split_sentences(cited) or [cited]
        vecs = embedder.embed_texts([item.claim] + sents)
        cos = float(max(vecs[1:] @ vecs[0])) if len(sents) else 0.0
        lex = lexical_support(item.claim, cited)
        item.support = round(max(cos, lex), 3)
        item.supported = cos >= cos_threshold or lex >= lex_threshold
        supported += item.supported
    return round(supported / len(evidence), 3)
