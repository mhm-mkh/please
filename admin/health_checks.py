"""Independent health/readiness checks for files and initialized services."""

from config.settings import FAISS_INDEX_PATH, FAISS_METADATA_PATH, KNOWLEDGE_PATH, PATHWAYS_PATH


def check_readiness(services: dict | None = None) -> dict:
    services = services or {}
    checks = {
        "knowledge_file": KNOWLEDGE_PATH.is_file(),
        "faiss_index": FAISS_INDEX_PATH.is_file(),
        "index_metadata": FAISS_METADATA_PATH.is_file(),
        "pathways_file": PATHWAYS_PATH.is_file(),
        "retriever_initialized": services.get("retriever") is not None,
    }
    return {"ready": all(checks.values()), "checks": checks}
