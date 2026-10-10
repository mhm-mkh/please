"""Shared message orchestration for all platform adapters."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from .request_models import AssistantResponse, MessageRequest, PathResult

logger = logging.getLogger(__name__)


class MessagePath(Protocol):
    def run(self, request: MessageRequest, context: dict[str, Any]) -> Any: ...


class MessageHandler:
    """Run applicable paths and pass their structured results to response creation.

    Path selection should allow combinations: one request may retrieve knowledge,
    update a profile, assess pathways, and include greeting behavior together.
    """

    def __init__(self, paths: dict[str, MessagePath], response_creator: Any) -> None:
        self.paths = paths
        self.response_creator = response_creator

    def handle(self, request: MessageRequest, context: dict[str, Any]) -> AssistantResponse:
        progress = context.get("progress", lambda _message: None)
        started = time.perf_counter()
        progress("در حال تشخیص نوع پیام و به‌روزرسانی نمایه…")
        conversation = self.paths["conversation"].run(request, context)
        logger.info("request stage=conversation duration=%.2fs", time.perf_counter() - started)
        context["turn"] = conversation
        path_results: list[PathResult] = [PathResult(path="conversation", data=conversation)]
        progress("در حال بررسی اطلاعات نمایه و مسیرها…")
        stage_started = time.perf_counter()
        path_results.append(self.paths["profile"].run(request, context))
        logger.info("request stage=profile duration=%.2fs", time.perf_counter() - stage_started)

        if conversation.get("should_retrieve") and context.get("records"):
            progress("در حال جست‌وجو در منابع دانش…")
            stage_started = time.perf_counter()
            path_results.append(self.paths["knowledge"].run(request, context))
            logger.info("request stage=retrieval duration=%.2fs", time.perf_counter() - stage_started)

        # The combined path packages outputs for downstream consumers without re-running work.
        combined_context = dict(context)
        combined_context["path_results"] = path_results
        path_results.append(self.paths["combined"].run(request, combined_context))
        response = self.response_creator.create(request, path_results, conversation, progress=progress)
        logger.info("request stage=complete duration=%.2fs", time.perf_counter() - started)
        return response
