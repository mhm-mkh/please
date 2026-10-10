"""Central configuration; values will move here during implementation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "canada.jsonl"
PATHWAYS_PATH = PROJECT_ROOT / "data" / "pathways" / "pathways.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "knowledge" / "canada.faiss"
FAISS_METADATA_PATH = PROJECT_ROOT / "data" / "knowledge" / "canada.index.json"
EMBEDDING_MODEL = "codefuse-ai/F2LLM-v2-0.6B"
DEFAULT_RESULT_COUNT = 3
DEFAULT_MIN_SIMILARITY = 0.35
PATHWAY_SCORE_THRESHOLD = 0.30
