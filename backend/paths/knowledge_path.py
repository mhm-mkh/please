"""Knowledge/RAG path: retrieve evidence relevant to a message."""

from typing import Any

from ..request_models import MessageRequest, PathResult


class KnowledgePath:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def run(self, request: MessageRequest, context: dict[str, Any]) -> PathResult:
        result = self.retriever.search(
            request.text,
            limit=context.get("result_count", 3),
            min_similarity=context.get("min_similarity", 0.35),
        )
        return PathResult(path="knowledge", data={"results": result})
