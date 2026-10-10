"""Validate configured corpus and pathway JSON before deployment."""

from config.settings import KNOWLEDGE_PATH, PATHWAYS_PATH
from services.pathways.pathway_loader import load_pathways
from services.rag.corpus_loader import load_corpus


def main() -> None:
    records = load_corpus(KNOWLEDGE_PATH)
    pathways = load_pathways(PATHWAYS_PATH)
    print(f"Valid knowledge chunks: {len(records)}")
    print(f"Valid pathway definitions: {len(pathways)}")


if __name__ == "__main__":
    main()
