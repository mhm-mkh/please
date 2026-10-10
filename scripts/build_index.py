"""Build the corpus FAISS index locally and write matching metadata."""

import json

import faiss

from config.settings import EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_METADATA_PATH, KNOWLEDGE_PATH
from services.rag.retriever import build_index, corpus_fingerprint, load_chunks


def main() -> None:
    records = load_chunks(KNOWLEDGE_PATH)
    if not records:
        raise SystemExit("No text chunks found in the configured knowledge file.")
    _, index = build_index(records, EMBEDDING_MODEL)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_index = FAISS_INDEX_PATH.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(temp_index))
    temp_index.replace(FAISS_INDEX_PATH)
    metadata = {
        "model": EMBEDDING_MODEL,
        "fingerprint": corpus_fingerprint(records, EMBEDDING_MODEL),
        "chunks": len(records),
        "dimension": index.d,
    }
    temp_meta = FAISS_METADATA_PATH.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp_meta.replace(FAISS_METADATA_PATH)
    print(f"Built {index.ntotal} vectors ({index.d} dimensions).")
    print(f"Index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {FAISS_METADATA_PATH}")


if __name__ == "__main__":
    main()
