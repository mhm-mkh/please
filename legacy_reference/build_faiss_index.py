"""Build a FAISS index locally so Streamlit Cloud skips corpus embedding at startup."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
from rag import DEFAULT_EMBEDDING_MODEL, build_index, corpus_fingerprint, load_chunks

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "canada.jsonl"
INDEX_PATH = ROOT / "data" / "canada.faiss"
METADATA_PATH = ROOT / "data" / "canada.index.json"


def main() -> None:
    records = load_chunks(DATA_PATH)
    if not records:
        raise SystemExit("No text chunks found in data/canada.jsonl")

    print(f"Building an index for {len(records)} chunks with {DEFAULT_EMBEDDING_MODEL}", flush=True)
    _, index = build_index(records, DEFAULT_EMBEDDING_MODEL)

    temporary_index = INDEX_PATH.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temporary_index))
    temporary_index.replace(INDEX_PATH)

    metadata = {
        "model": DEFAULT_EMBEDDING_MODEL,
        "fingerprint": corpus_fingerprint(records, DEFAULT_EMBEDDING_MODEL),
        "chunks": len(records),
        "dimension": index.d,
    }
    temporary_metadata = METADATA_PATH.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(METADATA_PATH)

    print(f"Saved {INDEX_PATH.relative_to(ROOT)} ({index.ntotal} vectors, {index.d} dimensions)", flush=True)
    print(f"Saved {METADATA_PATH.relative_to(ROOT)}", flush=True)
    print("Upload both files alongside canada.jsonl to the app repository.", flush=True)


if __name__ == "__main__":
    main()
