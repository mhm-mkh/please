"""Small Streamlit admin view; mount behind authentication before deployment."""

import json
from pathlib import Path

import streamlit as st

from .health_checks import check_readiness


def render(services: dict, review_path: str | Path | None = None) -> None:
    st.title("Assistant administration")
    status = check_readiness(services)
    st.metric("Readiness", "ready" if status["ready"] else "needs attention")
    for name, healthy in status["checks"].items():
        st.write(f"{'✅' if healthy else '❌'} {name}")
    if st.button("Clear cached resources"):
        st.cache_resource.clear()
        st.success("Cache cleared. Resources will reload on the next app run.")
    if review_path and Path(review_path).is_file():
        st.subheader("Recent answer reviews")
        rows = []
        for line in Path(review_path).read_text(encoding="utf-8").splitlines()[-100:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        st.dataframe(rows, use_container_width=True)
