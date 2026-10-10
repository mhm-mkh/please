"""Profile path: extract/update user facts and request pathway assessment."""

from typing import Any

from ..request_models import MessageRequest, PathResult


class ProfilePath:
    def __init__(self, pathway_assessor: Any) -> None:
        self.pathway_assessor = pathway_assessor

    def run(self, request: MessageRequest, context: dict[str, Any]) -> PathResult:
        turn = context["turn"]
        profile = turn["profile"]
        pathways = self.pathway_assessor.assess(profile, context.get("pathways", []))
        return PathResult(
            path="profile",
            data={
                "profile": profile,
                "changed_fields": turn["changed_fields"],
                "recommendations": pathways,
            },
        )
