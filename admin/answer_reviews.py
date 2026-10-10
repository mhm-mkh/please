"""Append minimal answer-review feedback as JSONL (caller owns access control)."""

import json
from datetime import datetime, timezone
from pathlib import Path


def record_review(path: str | Path, answer_id: str, rating: str, notes: str | None = None) -> dict:
    if rating not in {"helpful", "not_helpful", "incorrect", "unsafe"}:
        raise ValueError("Unsupported review rating.")
    record = {
        "answer_id": answer_id,
        "rating": rating,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as target:
        target.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
