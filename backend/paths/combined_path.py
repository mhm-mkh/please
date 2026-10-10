"""Combine outputs from applicable paths without repeating their logic."""

from typing import Any

from ..request_models import MessageRequest, PathResult


class CombinedPath:
    def run(self, request: MessageRequest, context: dict[str, Any]) -> PathResult:
        results = context.get("path_results", [])
        combined: dict[str, Any] = {}
        for result in results:
            combined[result.path] = result.data
        return PathResult(path="combined", data=combined)
