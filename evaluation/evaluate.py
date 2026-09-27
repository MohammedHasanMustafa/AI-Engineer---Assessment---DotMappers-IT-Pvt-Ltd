from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from rag.eval_log import record_from_response

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "dataset.json"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def _match(source: str, expected: list[str]) -> bool:
    return any(e.lower() in source.lower() for e in expected)


def keyword_coverage(text: str, groups: list[list[str]]) -> float | None:
    if not groups:
        return None
    low = text.lower()
    return sum(any(k.lower() in low for k in g) for g in groups) / len(groups)


def evaluate_item(pipeline, item: dict, **ask_kwargs) -> dict:
    resp = pipeline.ask(item["question"], log_query=False, origin="evaluation", **ask_kwargs)
    text = resp.answer_text()
    retrieved_sources = [rc.chunk.source for rc in resp.retrieved]
    cited_sources = [rc.chunk.source for rc in resp.cited_retrieved()]
    exp = item.get("expected_sources", [])
    qtype = item["type"]
    r = {
        "id": item["id"], "type": qtype, "question": item["question"], "status": resp.status,
        "refused": resp.refused, "generator": resp.generator, "answer": text,
        "retrieved_sources": retrieved_sources, "cited_sources": cited_sources,
        "top_relevance": resp.relevance, "groundedness": resp.groundedness,
        "latency_ms": resp.latency_ms.get("total_ms"), "security_events": resp.security_events,
        "retrieval_hit": None, "citation_correctness": None, "keyword_coverage": None,
        "refusal_correct": None, "answer_correct": None, "conflict_detected": None,
        "injection_resisted": None,
    }
    if qtype in ("answerable", "contradictory") and exp:
        if qtype == "answerable":
            r["retrieval_hit"] = any(_match(s, exp) for s in retrieved_sources)
        else:
            r["retrieval_hit"] = all(any(_match(s, [e]) for s in retrieved_sources) for e in exp)
        if cited_sources:
            r["citation_correctness"] = round(sum(_match(s, exp) for s in cited_sources) / len(cited_sources), 3)

    if qtype == "answerable":
        cov = keyword_coverage(text, item.get("expected_keywords", []))
        r["keyword_coverage"] = cov
        r["refusal_correct"] = not resp.refused
        r["answer_correct"] = (not resp.refused) and (cov is None or cov >= 0.5)
        accuracy = 0.0 if resp.refused else (cov if cov is not None else 1.0)
    elif qtype == "unanswerable":
        r["refusal_correct"] = r["answer_correct"] = resp.refused
        accuracy = float(resp.refused)
    elif qtype == "contradictory":
        covered = sum(any(_match(s, [e]) for s in cited_sources) for e in exp)
        r["conflict_detected"] = r["answer_correct"] = resp.status == "conflicting" or covered >= 2
        accuracy = float(r["conflict_detected"])
    else:  # injection
        leaked = [f for f in item.get("forbidden_strings", []) if f.lower() in text.lower()]
        r["injection_resisted"] = r["answer_correct"] = not leaked
        r["leaked_strings"] = leaked
        accuracy = float(not leaked)

    pipeline.log.append(record_from_response(resp, "evaluation", accuracy,
                                             f"ground truth ({qtype})", item["id"]))
    return r


def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(results: list[dict], config: dict) -> dict:
    by = lambda t: [r for r in results if r["type"] == t]  # noqa: E731
    lat = [r["latency_ms"] for r in results if r["latency_ms"] is not None]
    return {
        "n_questions": len(results),
        "counts_by_type": {t: len(by(t)) for t in ("answerable", "unanswerable", "contradictory", "injection")},
        "retrieval_hit_rate": _mean(r["retrieval_hit"] for r in results),
        "citation_correctness": _mean(r["citation_correctness"] for r in results),
        "answer_groundedness": _mean(r["groundedness"] for r in results if not r["refused"]),
        "refusal_accuracy": _mean(r["refusal_correct"] for r in results),
        "correct_refusal_rate_unanswerable": _mean(r["refusal_correct"] for r in by("unanswerable")),
        "false_refusal_rate_answerable": _mean((not r["refusal_correct"]) for r in by("answerable")),
        "answer_accuracy_answerable": _mean(r["answer_correct"] for r in by("answerable")),
        "contradiction_detection_rate": _mean(r["conflict_detected"] for r in by("contradictory")),
        "injection_resistance_rate": _mean(r["injection_resisted"] for r in by("injection")),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p95_latency_ms": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)], 1) if lat else None,
        "generators_used": sorted({r["generator"] for r in results}),
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_evaluation(pipeline, dataset_path: Path = DATASET_PATH, out_dir: Path | None = None,
                   progress=None, **ask_kwargs) -> tuple[list[dict], dict]:
    items = load_dataset(dataset_path)
    results = []
    for n, item in enumerate(items, start=1):
        results.append(evaluate_item(pipeline, item, **ask_kwargs))
        if progress:
            progress(n, len(items), item)
    cfg = pipeline.cfg
    config = {"embedding_model": cfg.embedding_model, "reranker_model": cfg.reranker_model,
              "llm_model": cfg.llm_model, "chunk_size_words": cfg.chunk_size_words,
              "chunk_overlap_words": cfg.chunk_overlap_words,
              **{k: v for k, v in ask_kwargs.items() if k != "sources"}}
    metrics = aggregate(results, config)
    out_dir = Path(out_dir or cfg.eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cols = ["id", "type", "question", "status", "generator", "retrieval_hit", "citation_correctness",
            "keyword_coverage", "groundedness", "refusal_correct", "answer_correct",
            "conflict_detected", "injection_resisted", "top_relevance", "latency_ms"]
    with (out_dir / "eval_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    return results, metrics


def main() -> None:
    from config import settings
    from rag.pipeline import RAGPipeline

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-k", type=int, default=settings.top_k)
    ap.add_argument("--extractive", action="store_true", help="skip the LLM (extractive answers)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-hybrid", action="store_true")
    args = ap.parse_args()

    pipe = RAGPipeline(settings)
    if len(pipe.store) == 0:
        print("Index empty; ingesting data/documents ...")
        for r in pipe.ingest_directory(settings.docs_dir, settings.upload_dir):
            print(f"  [{r.status}] {r.source}: {r.message}")
    print("LLM:", pipe.llm_status()[1])

    def progress(n, total, item):
        print(f"  {n:>2}/{total} {item['id']} {item['type']}")

    _, metrics = run_evaluation(pipe, progress=progress, top_k=args.top_k,
                                use_rerank=not args.no_rerank, use_hybrid=not args.no_hybrid,
                                generator="extractive" if args.extractive else "auto")
    print(json.dumps({k: v for k, v in metrics.items() if k != "config"}, indent=2))
    print(f"Results written to {settings.eval_results_dir}")


if __name__ == "__main__":
    main()
