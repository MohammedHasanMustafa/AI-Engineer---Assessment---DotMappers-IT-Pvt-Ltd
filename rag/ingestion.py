from __future__ import annotations

import hashlib
import io
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import Chunk
from .security import scan_for_injection
from .text_utils import normalize_ws, split_sentences, strip_control

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class IngestionError(Exception):
    """File could not be read (corrupt, encrypted, binary, unsupported)."""


class EmptyDocumentError(IngestionError):
    """File was readable but contained no usable text."""


@dataclass
class IngestResult:
    source: str
    status: str               
    message: str = ""
    n_chunks: int = 0
    n_pages: int = 0
    injection_chunks: int = 0
    seconds: float = 0.0


def extract_pages(name: str, data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, raw_text)]. TXT/MD files are a single page."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type '{ext or 'none'}'. Accepted: PDF, TXT, MD.")
    if not data or not data.strip():
        raise EmptyDocumentError("File is empty.")
    if ext == ".pdf":
        return _extract_pdf(data)
    return [(1, _decode_text(data))]


def _extract_pdf(data: bytes) -> list[tuple[int, str]]:
    if b"%PDF-" not in data[:1024]:
        raise IngestionError("Corrupt PDF: missing %PDF header.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  
                raise IngestionError("PDF is password-protected.") from exc
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                pages.append((i, page.extract_text() or ""))
            except Exception as exc:  
                log.warning("Could not extract page %s: %s", i, exc)
                pages.append((i, ""))
    except IngestionError:
        raise
    except Exception as exc:  
        raise IngestionError(f"Corrupt or unreadable PDF ({type(exc).__name__}: {exc}).") from exc
    if not pages:
        raise EmptyDocumentError("PDF has no pages.")
    return pages


def _decode_text(data: bytes) -> str:
    if data.count(b"\x00") > max(8, len(data) // 100):
        raise IngestionError("File looks binary, not text. Upload a PDF, TXT or Markdown file.")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def clean_text(raw: str) -> str:
    """Normalise unicode, repair PDF hyphenation, drop page-number lines, flatten markdown."""
    t = unicodedata.normalize("NFKC", raw.replace("\x00", " "))
    t = strip_control(t)
    t = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", t)          
    out = []
    for line in t.splitlines():
        s = re.sub(r"^(>\s*)+", "", line.strip())               
        if not s or re.fullmatch(r"(page\s*)?\d{1,4}", s, re.IGNORECASE):
            continue
        if s.startswith("#"):                                  
            s = s.lstrip("#").strip()
            if s and s[-1] not in ".!?:":
                s += "."
        elif re.match(r"^[-*+]\s+", s):                         
            s = re.sub(r"^[-*+]\s+", "", s)
            if s and s[-1] not in ".!?:;":
                s += "."
        if s:
            out.append(s)
    return normalize_ws(" ".join(out))


def chunk_pages(pages: list[tuple[int, str]], source: str, doc_hash: str,
                size: int, overlap: int, min_words: int) -> list[Chunk]:
    """Sentence-aware sliding window over the whole document.

    * never cuts a sentence (sentences longer than `size` are hard-split),
    * windows may cross page boundaries; `page`..`page_end` records the span,
    * consecutive windows share ~`overlap` words of whole sentences,
    * a tail shorter than `min_words` is merged into the last window.
    """
    units: list[tuple[int, list[str]]] = []
    for page_no, raw in pages:
        for sent in split_sentences(clean_text(raw)):
            words = sent.split()
            for i in range(0, len(words), size):
                units.append((page_no, words[i:i + size]))
    if not units:
        return []

    suffix = [0] * (len(units) + 1)
    for i in range(len(units) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + len(units[i][1])

    stem = re.sub(r"[^A-Za-z0-9]+", "_", Path(source).stem)[:40].strip("_") or "doc"
    chunks: list[Chunk] = []
    i = 0
    while i < len(units):
        j, n = i, 0
        while j < len(units) and (n + len(units[j][1]) <= size or j == i):
            n += len(units[j][1])
            j += 1
        if j < len(units) and suffix[j] < min_words:            
            n += suffix[j]
            j = len(units)
        window = units[i:j]
        text = " ".join(" ".join(w) for _, w in window)
        idx = len(chunks)
        chunks.append(Chunk(
            chunk_id=f"{stem}-p{window[0][0]}-c{idx:04d}",
            source=source, page=window[0][0], page_end=window[-1][0], chunk_index=idx,
            text=text, doc_hash=doc_hash, n_words=n,
        ))
        if j >= len(units):
            break
        k, back = j, 0                                          
        while k > i + 1 and back + len(units[k - 1][1]) <= overlap:
            k -= 1
            back += len(units[k][1])
        i = k
    return chunks


def ingest_document(name: str, data: bytes, store, embedder, cfg) -> IngestResult:
    t0 = time.perf_counter()
    source = Path(name).name
    ext = Path(source).suffix.lower()

    def done(status: str, message: str, **kw) -> IngestResult:
        return IngestResult(source, status, message, seconds=round(time.perf_counter() - t0, 2), **kw)

    if ext not in SUPPORTED_EXTENSIONS:
        return done("unsupported", f"Unsupported file type '{ext or 'none'}'. Accepted: PDF, TXT, MD.")
    if len(data) > cfg.max_file_mb * 1024 * 1024:
        return done("error", f"File is larger than {cfg.max_file_mb} MB.")
    if not data or not data.strip():
        return done("empty", "File is empty; nothing was indexed.")

    doc_hash = hashlib.sha256(data).hexdigest()
    if store.has_hash(doc_hash):
        return done("skipped", "Identical content is already indexed.")

    try:
        pages = extract_pages(source, data)
    except EmptyDocumentError as exc:
        return done("empty", str(exc))
    except IngestionError as exc:
        return done("error", str(exc))

    chunks = chunk_pages(pages, source, doc_hash, cfg.chunk_size_words,
                         cfg.chunk_overlap_words, cfg.min_chunk_words)
    if not chunks:
        return done("empty", "No extractable text (scanned/image-only PDF or whitespace only).",
                    n_pages=len(pages))

    replaced = store.has_source(source)
    if replaced:
        store.delete_source(source)

    flagged = 0
    for c in chunks:
        c.injection_flags = scan_for_injection(c.text)
        flagged += bool(c.injection_flags)

    vectors = embedder.embed_documents([c.text for c in chunks])
    store.add(chunks, vectors, {
        "source": source, "doc_hash": doc_hash, "n_pages": len(pages), "n_chunks": len(chunks),
        "injection_chunks": flagged, "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    msg = f"Indexed {len(chunks)} chunks from {len(pages)} page(s)."
    if replaced:
        msg += " Replaced the previous version."
    if flagged:
        msg += (f" {flagged} chunk(s) contain suspected prompt-injection text; "
                "they are stored as untrusted data and neutralised at query time.")
    return done("indexed", msg, n_chunks=len(chunks), n_pages=len(pages), injection_chunks=flagged)