from rag.embeddings import HashingEmbedder
from rag.ingestion import chunk_pages, clean_text, extract_pages, ingest_document, IngestionError
from rag.vector_store import FaissVectorStore
import pytest


def _doc(n_sentences=200):
    return " ".join(f"Sentence number {i} talks about retrieval quality and embeddings." for i in range(n_sentences))


def test_chunks_respect_size_and_overlap():
    chunks = chunk_pages([(1, _doc())], "a.txt", "h", size=100, overlap=20, min_words=10)
    assert len(chunks) > 5
    assert all(c.n_words <= 100 + 10 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        tail = a.text.split()[-20:]
        assert " ".join(tail[-8:]) in b.text      


def test_chunk_metadata_preserved_across_pages():
    pages = [(1, _doc(30)), (2, _doc(30))]
    chunks = chunk_pages(pages, "paper.pdf", "h", size=120, overlap=20, min_words=10)
    assert chunks[0].page == 1 and chunks[-1].page_end == 2
    assert all(c.chunk_id.startswith("paper-p") for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_clean_text_repairs_hyphenation_and_drops_page_numbers():
    assert clean_text("retrie-\nval works\n\n12\n# Heading") == "retrieval works Heading."


def test_unsupported_and_corrupt_files():
    with pytest.raises(IngestionError):
        extract_pages("x.docx", b"hello")
    with pytest.raises(IngestionError):
        extract_pages("x.pdf", b"this is not a pdf at all")
    with pytest.raises(IngestionError):
        extract_pages("x.pdf", b"%PDF-1.4\n garbage garbage")


def test_ingest_status_codes(cfg):
    store, emb = FaissVectorStore(cfg.index_dir), HashingEmbedder()
    assert ingest_document("empty.txt", b"   \n ", store, emb, cfg).status == "empty"
    assert ingest_document("bad.pdf", b"%PDF-1.7 broken", store, emb, cfg).status == "error"
    assert ingest_document("x.exe", b"MZ", store, emb, cfg).status == "unsupported"
    data = _doc(80).encode()
    assert ingest_document("ok.txt", data, store, emb, cfg).status == "indexed"
    assert ingest_document("copy.txt", data, store, emb, cfg).status == "skipped"  
    assert store.has_source("ok.txt") and len(store) > 0


def test_injection_chunks_flagged_at_ingestion(pipeline_factory):
    pipe = pipeline_factory()
    flagged = [c for c in pipe.store.chunks if c.injection_flags]
    assert flagged and all(c.source == "prompt_injection_test_document.md" for c in flagged)
