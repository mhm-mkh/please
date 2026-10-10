"""Load structured pathway definitions from JSON."""

import json
from pathlib import Path


def load_pathways(path: str | Path) -> list[dict]:
    pathway_file = Path(path)
    if not pathway_file.exists():
        return []
    with pathway_file.open("r", encoding="utf-8-sig") as source:
        data = json.load(source)
    if isinstance(data, dict):
        data = data.get("pathways", [])
    if not isinstance(data, list):
        raise ValueError("pathways.json must contain a 'pathways' array.")
    if any(not isinstance(item, dict) for item in data):
        raise ValueError("Every pathway entry must be a JSON object.")
    return data
