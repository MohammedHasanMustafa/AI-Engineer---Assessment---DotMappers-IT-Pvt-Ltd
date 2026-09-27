import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  
from rag.embeddings import HashingEmbedder  
from rag.pipeline import RAGPipeline  


class FakeLLM:
    model = "fake-llm"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def status(self):
        return True, "fake"

    def available(self):
        return True

    def chat_json(self, system, user):
        self.calls.append((system, user))
        out = self.responder(system, user)
        return out if isinstance(out, str) else json.dumps(out)


def canary_of(system: str) -> str:
    return re.search(r"CANARY-[0-9a-f]+", system).group(0)


@pytest.fixture
def cfg(tmp_path):
    return replace(settings, index_dir=tmp_path / "index", log_path=tmp_path / "log.jsonl",
                   upload_dir=tmp_path / "uploads", eval_results_dir=tmp_path / "results",
                   min_vector_relevance=0.05, extractive_min_sim=0.0)


def make_pipeline(cfg, responder=None):
    llm = FakeLLM(responder) if responder else None
    pipe = RAGPipeline(cfg, embedder=HashingEmbedder(), reranker=None, llm=llm)
    pipe.ingest_directory(ROOT / "data" / "documents")
    return pipe


@pytest.fixture
def pipeline_factory(cfg):
    return lambda responder=None: make_pipeline(cfg, responder)
