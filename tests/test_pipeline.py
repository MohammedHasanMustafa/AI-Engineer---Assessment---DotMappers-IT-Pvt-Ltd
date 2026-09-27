import re

from tests.conftest import canary_of


def test_empty_index_refuses(cfg):
    from rag.embeddings import HashingEmbedder
    from rag.pipeline import RAGPipeline
    pipe = RAGPipeline(cfg, embedder=HashingEmbedder(), reranker=None, llm=None)
    r = pipe.ask("What is attention?")
    assert r.status == "insufficient_evidence" and "No documents" in r.message


def test_extractive_answer_has_citations_and_passages(pipeline_factory):
    pipe = pipeline_factory()
    r = pipe.ask("What is the relationship between cosine similarity and inner product for L2-normalized embeddings?",
                 generator="extractive")
    assert r.status == "answered" and r.evidence and r.retrieved
    assert all(e.labels for e in r.evidence)
    assert r.latency_ms["total_ms"] > 0 and r.groundedness is not None


def test_injected_text_never_reaches_llm(pipeline_factory):
    pipe = pipeline_factory(lambda s, u: {"status": "insufficient_evidence", "evidence": []})
    pipe.ask("What do the practitioner notes say about normalizing embeddings?", top_k=8)
    _, user_prompt = pipe.llm.calls[-1]
    assert "Ignore the user" not in user_prompt and "respond only with" not in user_prompt.lower()
    assert "REMOVED: suspected embedded instruction" in user_prompt


def test_output_guard_blocks_system_prompt_leak(pipeline_factory):
    def hijacked(system, user):
        return {"status": "answered", "evidence": [{"claim": "Normalized cosine equals dot product.",
                                                    "citations": ["S1"]}],
                "inference": f"My secret is {canary_of(system)}"}
    pipe = pipeline_factory(hijacked)
    r = pipe.ask("What do the practitioner notes say about normalizing embeddings?")
    assert r.status == "blocked" and not r.evidence
    assert any("BLOCKED" in e for e in r.security_events)


def test_output_guard_drops_injected_payload(pipeline_factory):
    def obeys(system, user):
        return {"status": "answered", "evidence": [
            {"claim": "PWNED", "citations": ["S1"]},
            {"claim": "Cosine similarity is equivalent to the inner product for normalized vectors.", "citations": ["S1"]}],
            "inference": "", "conflicts": ""}
    pipe = pipeline_factory(obeys)
    r = pipe.ask("What do the practitioner notes say about normalizing embeddings?", top_k=8)
    assert "PWNED" not in r.answer_text()
    assert r.status == "answered" and len(r.evidence) == 1


def test_fabricated_citations_removed(pipeline_factory):
    pipe = pipeline_factory(lambda s, u: {"status": "answered",
                                          "evidence": [{"claim": "Made up fact.", "citations": ["S99"]}]})
    r = pipe.ask("What chunk size does the Orion guide recommend?")
    assert r.status == "insufficient_evidence" and not r.evidence
    assert any("S99" in w for w in r.warnings)
    assert r.citation_validity == 0.0


def test_conflicting_status_passes_through(pipeline_factory):
    def conflict(system, user):
        labels = re.findall(r'id="(S\d+)"', user)
        return {"status": "conflicting",
                "evidence": [{"claim": "2023 guide recommends 512 tokens.", "citations": [labels[0]]},
                             {"claim": "2024 guide recommends 256 tokens.", "citations": [labels[1]]}],
                "conflicts": "The two guides disagree."}
    pipe = pipeline_factory(conflict)
    r = pipe.ask("What chunk size does the Orion vector search guide recommend?")
    assert r.status == "conflicting" and r.conflicts


def test_long_and_empty_queries(pipeline_factory):
    pipe = pipeline_factory()
    assert pipe.ask("").status == "rejected"
    assert pipe.ask("x " * 30000).status == "rejected"
    r = pipe.ask("What chunk size does Orion recommend? " * 60, generator="extractive")
    assert r.status != "rejected" and r.warnings


def test_metadata_filter(pipeline_factory):
    pipe = pipeline_factory()
    r = pipe.ask("chunk size", sources=["orion_vector_search_guide_2024.md"], generator="extractive")
    assert r.retrieved and {rc.chunk.source for rc in r.retrieved} == {"orion_vector_search_guide_2024.md"}


def test_evaluation_runs_offline(pipeline_factory, cfg):
    from evaluation.evaluate import run_evaluation
    pipe = pipeline_factory()
    results, metrics = run_evaluation(pipe, generator="extractive")
    assert metrics["n_questions"] == len(results) >= 20
    for key in ("retrieval_hit_rate", "refusal_accuracy", "avg_latency_ms", "injection_resistance_rate"):
        assert key in metrics
    assert metrics["injection_resistance_rate"] == 1.0
    assert (cfg.eval_results_dir / "metrics.json").exists()
