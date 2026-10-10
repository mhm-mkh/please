"""JSONL loading, embedding, retrieval, and context-group reconstruction."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import time
from typing import Any, Callable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "codefuse-ai/F2LLM-v2-0.6B"


def corpus_fingerprint(records: list[dict[str, Any]], model_name: str) -> str:
    """Fingerprint ordered corpus text and model so stale indexes are rejected."""
    passages = [row["text"] for row in records]
    payload = model_name + "\0" + json.dumps(passages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_precomputed_index(
    records: list[dict[str, Any]],
    model_name: str,
    index_path: str | Path,
    metadata_path: str | Path,
    expected_dimension: int | None = None,
) -> faiss.IndexFlatIP:
    """Load the offline-built FAISS index only if its corpus/model match."""
    index_file = Path(index_path)
    metadata_file = Path(metadata_path)
    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            "Precomputed search index is missing. Run `python scripts/build_index.py` locally, "
            "then upload data/knowledge/canada.faiss and canada.index.json with the app."
        )
    with metadata_file.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    expected_fingerprint = corpus_fingerprint(records, model_name)
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "The saved search index does not match this JSONL data/model. "
            "Run `python scripts/build_index.py` again and upload the regenerated index files."
        )
    index = faiss.read_index(str(index_file))
    if index.ntotal != len(records):
        raise ValueError(
            f"Saved index has {index.ntotal} vectors for {len(records)} chunks. "
            "Regenerate it with `python scripts/build_index.py`."
        )
    if expected_dimension is not None and index.d != expected_dimension:
        raise ValueError(
            f"Saved index dimension {index.d} does not match model dimension {expected_dimension}. "
            "Regenerate it with `python scripts/build_index.py`."
        )
    return index


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate one JSON object per line from the chunk JSONL file."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Knowledge data not found: {data_path}. Add the supplied chunk JSONL file there."
        )

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with data_path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc.msg}. "
                    "Each physical line must contain one complete JSON object."
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            if not isinstance(item.get("id"), str) or not item["id"].strip():
                raise ValueError(f"Line {line_number} is missing a non-empty string 'id'.")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate chunk id {item['id']!r} at line {line_number}.")
            seen_ids.add(item["id"])
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                continue
            if not isinstance(item.get("links", []), list):
                raise ValueError(f"'links' must be an array at line {line_number}.")
            item.setdefault("context_group_id", None)
            item.setdefault("links", [])
            item.setdefault("section_path", [])
            records.append(item)

    return records


def build_index(
    records: list[dict[str, Any]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    progress: Callable[[str, float | None], None] | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[SentenceTransformer, faiss.IndexFlatIP]:
    """Load the embedder and load or build a data/model-specific FAISS index."""
    def report(message: str, fraction: float | None = None) -> None:
        if progress:
            progress(message, fraction)

    started = time.perf_counter()
    passages = [row["text"] for row in records]
    fingerprint = corpus_fingerprint(records, model_name)
    index_path = Path(cache_dir) / f"{fingerprint}.faiss" if cache_dir else None

    report("بارگذاری مدل embedding…", 0.05)
    embedder = SentenceTransformer(model_name)
    print(f"[startup] Embedding model ready in {time.perf_counter() - started:.1f}s", flush=True)

    if index_path and index_path.is_file():
        report("بازیابی نمایه ذخیره‌شده…", 0.95)
        try:
            index = faiss.read_index(str(index_path))
            if index.ntotal == len(passages):
                print(f"[startup] Reused FAISS index ({index.ntotal} chunks)", flush=True)
                report("نمایه آماده است.", 1.0)
                return embedder, index
            print("[startup] Cached FAISS index size mismatch; rebuilding", flush=True)
        except Exception as exc:
            print(f"[startup] Could not read cached index ({exc}); rebuilding", flush=True)

    if not passages:
        raise ValueError("Cannot build a search index: the knowledge file has no text chunks.")

    # The hosted app is CPU/memory constrained; smaller batches avoid a long,
    # silent first inference step on this relatively large embedding model.
    batch_size = 4
    batches = (len(passages) + batch_size - 1) // batch_size
    vectors_by_batch = []
    encoding_started = time.perf_counter()
    for batch_number, start in enumerate(range(0, len(passages), batch_size), start=1):
        batch_end = min(start + batch_size, len(passages))
        batch_message = f"تبدیل بخش‌های دانش به بردار: {start + 1} تا {batch_end} از {len(passages)}"
        report(batch_message, 0.1 + 0.8 * (batch_number - 1) / batches)
        print(f"[startup] Encoding chunks {start + 1}-{batch_end}/{len(passages)}", flush=True)
        batch_vectors = embedder.encode_document(
            passages[start:batch_end],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        vectors_by_batch.append(batch_vectors)
        report(f"بردارسازی انجام شد: {batch_end} از {len(passages)} بخش", 0.1 + 0.8 * batch_number / batches)
        print(f"[startup] Encoded {batch_end}/{len(passages)} chunks", flush=True)
    vectors = np.concatenate(vectors_by_batch, axis=0)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index_path:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = index_path.with_suffix(".tmp")
        faiss.write_index(index, str(temp_path))
        temp_path.replace(index_path)
    print(f"[startup] FAISS index ready in {time.perf_counter() - encoding_started:.1f}s", flush=True)
    report("نمایه آماده است.", 1.0)
    return embedder, index


def retrieve(
    question: str,
    records: list[dict[str, Any]],
    embedder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    limit: int = 3,
    min_similarity: float = 0.35,
) -> list[dict[str, Any]]:
    """Return ranked chunks with reconstructed context for shared group IDs."""
    query_vector = embedder.encode_query(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32").reshape(1, -1)
    count = min(max(1, limit), len(records))
    similarities, ids = index.search(query_vector, count)

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        group_id = row.get("context_group_id")
        if group_id is not None and str(group_id).strip():
            groups.setdefault(str(group_id), []).append(row)

    results: list[dict[str, Any]] = []
    emitted_contexts: set[str] = set()
    for record_index, score in zip(ids[0], similarities[0]):
        if record_index < 0 or float(score) < min_similarity:
            continue
        row = records[int(record_index)]
        group_id = row.get("context_group_id")
        result_key = f"context:{group_id}" if group_id is not None else f"chunk:{row['id']}"
        # Several retrieved chunks can point to the same context. Show that context once.
        if result_key in emitted_contexts:
            continue
        emitted_contexts.add(result_key)
        context_rows = groups.get(str(group_id), []) if group_id is not None else []
        def context_order(item: dict[str, Any]) -> tuple[int, int, str]:
            # Prefer an explicit zero-based ordering field when available. The
            # updated corpus instead supplies globally unique IDs such as
            # ``ch_263``; their numeric suffix preserves source chunk order.
            chunk_index = item.get("chunk_index")
            if chunk_index is not None:
                try:
                    return (0, int(chunk_index), str(item.get("id", "")))
                except (TypeError, ValueError):
                    pass

            chunk_id = str(item.get("chunk_id", ""))
            match = re.search(r"(\d+)$", chunk_id)
            if match:
                return (1, int(match.group(1)), str(item.get("id", "")))

            # Last-resort compatibility for older corpora without chunk_id.
            return (2, 0, str(item.get("id", "")))

        context_rows = sorted(context_rows, key=context_order)
        if context_rows:
            # Add join_after to a chunk when an exact space/newline is needed at its boundary.
            context_text = "".join(
                str(part.get("text", "")) + str(part.get("join_after", ""))
                for part in context_rows
            )
        else:
            context_text = row["text"]
        results.append(
            {
                "id": row["id"],
                "topic": row.get("topic") or row.get("document_title", "موضوع مرتبط"),
                "section_path": row.get("section_path", []),
                "text": row["text"],
                "context_group_id": group_id,
                "context_text": context_text,
                "similarity": float(score),
                "requires_current_verification": bool(
                    row.get("requires_current_verification", False)
                ),
                "links": row.get("links", []),
            }
        )
    return results


class Retriever:
    """Object wrapper around retrieval for dependency injection."""

    def __init__(self, records, embedder, index) -> None:
        self.records = records
        self.embedder = embedder
        self.index = index

    def search(self, question: str, limit: int = 3, min_similarity: float = 0.35):
        if not self.records:
            return []
        return retrieve(question, self.records, self.embedder, self.index, limit, min_similarity)
