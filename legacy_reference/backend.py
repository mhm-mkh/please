"""Application backend: RAG coordination, pathway matching, and link selection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dialog_management import (
    PROFILE_FIELDS,
    PROFILE_PATHWAY_ALIASES,
    analyze_turn,
    normalize_persian,
)
from rag import retrieve


def load_pathways(path: str | Path) -> list[dict[str, Any]]:
    """Load the structured pathway JSON file."""
    data_path = Path(path)
    if not data_path.exists():
        return []
    with data_path.open("r", encoding="utf-8-sig") as source:
        data = json.load(source)
    if isinstance(data, dict):
        data = data.get("pathways", [])
    if not isinstance(data, list):
        raise ValueError("pathways.json must contain a 'pathways' array.")
    return [item for item in data if isinstance(item, dict)]


def _profile_value(profile: dict[str, Any], field: str) -> Any:
    return profile.get(PROFILE_PATHWAY_ALIASES.get(field, field))


def recommend_pathways(
    profile: dict[str, Any], pathways: list[dict[str, Any]], threshold: float = 0.30
) -> list[dict[str, Any]]:
    """Rank preliminary profile matches by criteria with related profile evidence."""
    recommendations = []
    for pathway in pathways:
        criteria = pathway.get("criteria", [])
        if not criteria:
            continue
        matched = 0
        missing: list[str] = []
        missing_fields: list[str] = []
        for criterion in criteria:
            field_names = criterion.get("profile_fields", [])
            if isinstance(field_names, str):
                field_names = [field_names]
            fields_with_evidence = [
                name for name in field_names
                if _profile_value(profile, name) not in (None, "", [], {})
            ]
            if fields_with_evidence:
                matched += 1
            else:
                missing.append(criterion.get("description_fa", criterion.get("id", "اطلاعات تکمیلی")))
                missing_fields.extend(field_names)
        score = matched / len(criteria)
        recommendations.append(
            {
                "id": pathway.get("id", ""),
                "name": pathway.get("name_fa", pathway.get("id", "مسیر")),
                "outcome": pathway.get("outcome", ""),
                "availability": pathway.get("availability", "unknown"),
                "score": score,
                "suggested": score > threshold,
                "matched": matched,
                "total": len(criteria),
                "missing": missing,
                "missing_fields": list(dict.fromkeys(missing_fields)),
            }
        )
    return sorted(recommendations, key=lambda item: item["score"], reverse=True)


def _intent_for_question(question: str) -> str | None:
    text = normalize_persian(question)
    if any(term in text for term in ("امتیاز", "crs", "محاسبه", "اکسپرس انتری")):
        return "express_entry_points"
    if any(
        term in text
        for term in (
            "آیا می توانم مهاجرت", "میتوانم مهاجرت", "واجد شرایط", "شرایط من",
            "ارزیابی", "کدام مسیر", "مسیر مناسب", "دنبال اقامت دائم",
        )
    ):
        return "assessment"
    return None


def collect_links(question: str, results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Collect reliable structured links and Markdown links from retrieved chunks."""
    requested_intent = _intent_for_question(question)
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    markdown_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    for result in results:
        for link in result.get("links", []):
            if not isinstance(link, dict) or not link.get("url"):
                continue
            if link.get("reliable") is not True:
                continue
            intent = link.get("intent")
            if requested_intent and intent and intent != requested_intent:
                continue
            url = str(link["url"])
            if url not in seen:
                seen.add(url)
                links.append({"label": str(link.get("label") or "اطلاعات بیشتر"), "url": url})
        for label, url in markdown_pattern.findall(result.get("text", "")):
            if url not in seen:
                seen.add(url)
                links.append({"label": label, "url": url})
    return links


def _updated_fields_text(fields: list[str]) -> str:
    labels = [PROFILE_FIELDS.get(field, field) for field in fields]
    return "، ".join(labels)


def handle_message(
    question: str,
    session: dict[str, Any],
    records: list[dict[str, Any]],
    embedder: Any,
    index: Any,
    pathways: list[dict[str, Any]],
    result_count: int = 3,
    min_similarity: float = 0.35,
) -> dict[str, Any]:
    """Interpret a turn, update session state, and coordinate the relevant actions."""
    turn = analyze_turn(question, session)
    profile = turn["profile"]
    state = turn["state"]
    recommendations = recommend_pathways(profile, pathways)
    prefix = "سلام، خوش آمدید. " if turn["first_response"] else ""

    if turn["greeting_only"]:
        answer = "سلام، خوش آمدید. لطفاً پرسش یا اطلاعاتی را که مایلید بررسی شود مطرح کنید."
        return {"answer": answer, "results": [], "links": [], "recommendations": recommendations, "profile": profile}

    if turn["thanks_only"]:
        answer = prefix + "از لطف شما سپاسگزارم. اگر پرسش دیگری دارید، بفرمایید."
        return {"answer": answer, "results": [], "links": [], "recommendations": recommendations, "profile": profile}

    if turn["farewell_only"]:
        answer = prefix + "از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
        state["pending_fields"] = []
        return {"answer": answer, "results": [], "links": [], "recommendations": recommendations, "profile": profile}

    if turn["profile_only"]:
        fields_text = _updated_fields_text(turn["changed_fields"])
        answer = prefix + f"اطلاعات مربوط به «{fields_text}» در نمایه موقت این گفت‌وگو ثبت شد."
        if "follow_up" in turn["intents"] and recommendations:
            best = recommendations[0]
            answer += f"\n\nبا اطلاعات جدید، تطبیق اولیه مسیر «{best['name']}» {best['score']:.0%} است؛ این عدد تأیید صلاحیت نیست."
            if best["missing"]:
                answer += "\nبرای ادامه بررسی، لطفاً این اطلاعات را هم ارائه کنید: " + "؛ ".join(best["missing"][:3]) + "."
                state["pending_fields"] = best["missing_fields"]
            else:
                state["pending_fields"] = []
        if turn["farewell"]:
            answer += " از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
            state["pending_fields"] = []
        elif turn["gratitude"]:
            answer += " از لطف شما هم سپاسگزارم."
        return {"answer": answer, "results": [], "links": [], "recommendations": recommendations, "profile": profile}

    if not turn["should_retrieve"]:
        answer = prefix + "متوجه شدم. اگر مایلید، پرسش یا اطلاعات بیشتری مطرح کنید تا راهنمایی‌تان کنم."
        if turn["farewell"]:
            answer += " از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
        return {"answer": answer, "results": [], "links": [], "recommendations": recommendations, "profile": profile}

    results = retrieve(question, records, embedder, index, result_count, min_similarity) if records else []
    if results:
        answer = prefix + "این مطالب از میان بخش‌های مرتبط داده‌های موجود بازیابی شده‌اند."
    else:
        answer = prefix + "اطلاعات مرتبط کافی برای پاسخ به این پرسش در داده‌های موجود پیدا نکردم. لطفاً موضوع را مشخص‌تر کنید."

    if turn["changed_fields"]:
        answer += "\n\nاطلاعات «" + _updated_fields_text(turn["changed_fields"]) + "» نیز در نمایه موقت ثبت شد."

    if turn["assessment"]:
        suggested = [item for item in recommendations if item["suggested"]]
        if suggested:
            best = suggested[0]
            answer += (
                f"\n\nبر اساس اطلاعات ثبت‌شده، مسیر «{best['name']}» می‌تواند برای بررسی اولیه مناسب باشد "
                f"({best['score']:.0%} از معیارهای فهرست‌شده با اطلاعات فعلی پوشش داده شده است). "
                "این ارزیابی تأیید صلاحیت نیست."
            )
            if best["missing"]:
                answer += "\nبرای بررسی دقیق‌تر، لطفاً این موارد را هم مشخص کنید: " + "؛ ".join(best["missing"][:3]) + "."
                state["pending_fields"] = best["missing_fields"]
        else:
            answer += "\nبرای پیشنهاد مسیر، لطفاً درباره سن، تحصیلات، زبان، سابقه کاری، وضعیت مالی و ارتباط با کانادا اطلاعات بیشتری بدهید."
            state["pending_fields"] = [
                "age", "education", "language", "work_experience", "financial_status", "canadian_connection"
            ]

    if turn["gratitude"]:
        answer += "\n\nاز لطف شما سپاسگزارم."
    if turn["farewell"]:
        answer += "\n\nاز گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
        state["pending_fields"] = []

    return {
        "answer": answer,
        "results": results,
        "links": collect_links(question, results),
        "recommendations": recommendations,
        "profile": profile,
        "intents": turn["intents"],
    }
