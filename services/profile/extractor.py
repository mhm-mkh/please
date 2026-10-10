"""LLM-assisted, evidence-checked extraction of profile facts from chat turns."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

PROFILE_FIELDS = {
    "destination_country": "کشور مقصد",
    "age": "سن متقاضی",
    "marital_status": "وضعیت تأهل",
    "family_members": "اعضای خانواده و اطلاعات همراهان",
    "occupation": "شغل و حوزه تخصصی متقاضی",
    "education_resume": "تحصیلات و رزومه متقاضی",
    "work_experience": "سابقه کاری متقاضی",
    "research_achievements": "سوابق پژوهشی، افتخارات و دستاوردها",
    "budget": "بودجه و توان مالی",
    "applying_with_family": "تنهایی یا همراه خانواده اقدام کردن",
    "immigration_type": "مهاجرت فردی یا کسب‌وکار/استارتاپ",
    "previous_immigration": "سوابق درخواست مهاجرت، ویزا و سفر",
    "timeline": "محدودیت و ترجیح زمانی",
    "permanent_residence_goal": "هدف دریافت اقامت دائم",
    "cost_preferences": "ترجیحات هزینه و هزینه زندگی",
    "language_profile": "زبان‌ها، مدرک زبان و تمایل به یادگیری",
    "destination_relatives": "اقوام و اعضای خانواده در کشور مقصد",
    "destination_connections": "ارتباط‌های دیگر در کشور مقصد مانند پیشنهاد کار یا سابقه ویزا",
    "work_permit_goal": "نیاز یا علاقه به اجازه کار",
    "partner_profile": "تحصیلات، شغل، تجربه و زبان همسر/شریک زندگی",
    "contact_phone": "شماره تماس برای پیگیری انسانی",
    "business_experience": "سابقه مالکیت، مدیریت یا راه‌اندازی کسب‌وکار",
    "business_startup_details": "ایده، برنامه یا حمایت مربوط به کسب‌وکار/استارتاپ",
}

FIELD_TYPES = {
    "destination_country": str,
    "age": int,
    "marital_status": str,
    "family_members": dict,
    "occupation": str,
    "education_resume": str,
    "work_experience": str,
    "research_achievements": str,
    "budget": dict,
    "applying_with_family": str,
    "immigration_type": str,
    "previous_immigration": dict,
    "timeline": dict,
    "permanent_residence_goal": str,
    "cost_preferences": dict,
    "language_profile": dict,
    "destination_relatives": str,
    "destination_connections": str,
    "work_permit_goal": str,
    "partner_profile": dict,
    "contact_phone": str,
    "business_experience": str,
    "business_startup_details": str,
}

_SYSTEM_PROMPT = """You extract only facts explicitly stated by a user for an immigration-advice profile.
Return one JSON object exactly in this shape:
{"updates":{"field":{"value":VALUE,"evidence":"EXACT SUBSTRING COPIED FROM USER MESSAGE"}}}
Use only these field names and meanings: """ + json.dumps(PROFILE_FIELDS, ensure_ascii=False) + """.
Do not guess, infer, fill missing details, or turn questions/hypotheticals into user facts.
For nested object fields, include only the subfields explicitly stated. Preserve the user's wording for descriptive values.
Use an integer for age. Set immigration_type to individual, business_or_startup, or either when explicit.
Use the standard English country name for destination_country. Set permanent_residence_goal and work_permit_goal to yes, no, or unsure when explicit.
Useful nested keys: family_members may contain applying_with, children_count, and notes; partner_profile may contain age, occupation, education_resume, work_experience, and language_profile; budget may contain currency, available_now, planned_later, and notes; cost_preferences may contain acceptable_cost, cost_of_living_concern, and willing_to_work; previous_immigration may contain applications, refusals, visas, and travel_history; timeline may contain urgency, target_months, and flexible; language_profile may contain languages, certificates, scores, and willing_to_learn.
If the message gives no new profile facts, return {"updates":{}}.
For every update, evidence must be an exact, contiguous quote from the user's message. Output JSON only."""


def _merge(old: Any, new: Any) -> Any:
    if isinstance(old, dict) and isinstance(new, dict):
        merged = dict(old)
        for key, value in new.items():
            merged[key] = _merge(merged[key], value) if key in merged else value
        return merged
    return new


class ProfileExtractor:
    """Ask the configured LLM for candidate facts; accept only typed, quoted facts."""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def extract_updates(self, text: str, pending_fields: list[str] | None = None) -> dict[str, Any]:
        if not self.llm_client.configured:
            return {}
        user_prompt = json.dumps(
            {"message": text, "fields_being_clarified": pending_fields or []},
            ensure_ascii=False,
        )
        raw = self.llm_client.generate({"system": _SYSTEM_PROMPT, "user": user_prompt})
        if not raw:
            return {}
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            parsed = json.loads(cleaned)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Profile extraction returned invalid JSON")
            return {}
        candidates = parsed.get("updates", {}) if isinstance(parsed, dict) else {}
        if not isinstance(candidates, dict):
            return {}

        accepted: dict[str, Any] = {}
        for field, item in candidates.items():
            expected_type = FIELD_TYPES.get(field)
            if expected_type is None or not isinstance(item, dict):
                continue
            value, evidence = item.get("value"), item.get("evidence")
            if not isinstance(evidence, str) or not evidence or evidence not in text:
                continue
            if not isinstance(value, expected_type) or isinstance(value, bool):
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, dict) and not value:
                continue
            if field == "age" and not 0 < value < 120:
                continue
            accepted[field] = value
        return accepted

    @staticmethod
    def apply(profile: dict[str, Any], updates: dict[str, Any]) -> list[str]:
        changed = []
        for field, value in updates.items():
            merged = _merge(profile.get(field), value)
            if merged != profile.get(field):
                profile[field] = merged
                changed.append(field)
        return changed