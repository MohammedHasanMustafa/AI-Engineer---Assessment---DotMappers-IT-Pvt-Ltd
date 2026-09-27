from __future__ import annotations

from pathlib import Path

import requests

SAMPLE_PAPERS = [
    ("attention_is_all_you_need_vaswani_2017.pdf", "https://arxiv.org/pdf/1706.03762",
     "Attention Is All You Need (Transformer architecture)"),
    ("rag_knowledge_intensive_nlp_lewis_2020.pdf", "https://arxiv.org/pdf/2005.11401",
     "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("react_reasoning_acting_yao_2022.pdf", "https://arxiv.org/pdf/2210.03629",
     "ReAct: Synergizing Reasoning and Acting in Language Models (agentic AI)"),
    ("sentence_bert_reimers_2019.pdf", "https://arxiv.org/pdf/1908.10084",
     "Sentence-BERT (embeddings and semantic search)"),
    ("hallucination_survey_ji_2022.pdf", "https://arxiv.org/pdf/2202.03629",
     "Survey of Hallucination in Natural Language Generation"),
    ("indirect_prompt_injection_greshake_2023.pdf", "https://arxiv.org/pdf/2302.12173",
     "Not what you've signed up for: Indirect Prompt Injection"),
]


def download_sample_papers(dest: Path, timeout: int = 60) -> list[tuple[str, str, str]]:
    """Returns [(filename, status, message)]; never raises (offline use keeps working)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url, title in SAMPLE_PAPERS:
        path = dest / name
        if path.exists() and path.stat().st_size > 10_000:
            results.append((name, "present", title))
            continue
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "rag-assessment/1.0"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            path.write_bytes(r.content)
            results.append((name, "downloaded", f"{title} ({len(r.content) // 1024} KB)"))
        except Exception as exc:  
            results.append((name, "failed", f"{url}: {exc}"))
    return results
