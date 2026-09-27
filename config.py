from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = ROOT / "data" / "documents"      
    upload_dir: Path = ROOT / "data" / "uploads"     
    index_dir: Path = ROOT / "data" / "index"         
    log_path: Path = ROOT / "data" / "logs" / "eval_log.jsonl"
    eval_dataset: Path = ROOT / "evaluation" / "dataset.json"
    eval_results_dir: Path = ROOT / "evaluation" / "results"

    embedding_model: str = "BAAI/bge-small-en-v1.5"                 
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"     
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_timeout: int = 180
    temperature: float = 0.0
    num_ctx: int = 8192

    chunk_size_words: int = 300
    chunk_overlap_words: int = 60
    min_chunk_words: int = 30

    top_k: int = 5
    candidate_k: int = 25           
    rrf_k: int = 60                 
    dedup_threshold: float = 0.95   

    min_rerank_relevance: float = 0.10   
    min_vector_relevance: float = 0.55   

    extractive_max_sentences: int = 3
    extractive_min_sim: float = 0.50
    grounding_cos_threshold: float = 0.70
    grounding_lex_threshold: float = 0.60

    max_query_chars: int = 1000          
    hard_max_query_chars: int = 20000    
    max_file_mb: int = 50


def _from_env(base: Settings) -> Settings:
    overrides = {}
    for f in fields(base):
        raw = os.getenv(f.name.upper())
        if raw is None:
            continue
        current = getattr(base, f.name)
        try:
            if isinstance(current, Path):
                overrides[f.name] = Path(raw)
            elif isinstance(current, bool):
                overrides[f.name] = raw.lower() in {"1", "true", "yes"}
            else:
                overrides[f.name] = type(current)(raw)
        except ValueError:
            pass
    return replace(base, **overrides)


settings = _from_env(Settings())
