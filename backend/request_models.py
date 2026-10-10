"""Platform-neutral request/result contracts for backend paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MessageRequest:
    text: str
    conversation_id: str
    user_id: str | None = None
    platform: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PathResult:
    path: str
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class AssistantResponse:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    profile_updates: list[dict[str, Any]] = field(default_factory=list)
    pathway_assessment: list[dict[str, Any]] | None = None
    follow_up_questions: list[str] = field(default_factory=list)
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
