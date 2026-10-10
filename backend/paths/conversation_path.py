"""Persian-first multi-intent dialog and session-profile management.

This module uses deterministic rules and is intentionally easy to extend. It does
not call an LLM: the embedding model is reserved for knowledge retrieval.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any
import logging

from services.profile.extractor import PROFILE_FIELDS as NEW_PROFILE_FIELDS, ProfileExtractor

logger = logging.getLogger(__name__)


PROFILE_FIELDS = {
    **NEW_PROFILE_FIELDS,
    "children_count": "تعداد فرزند (قدیمی)",
    "education": "تحصیلات (قدیمی)",
    "language": "زبان (قدیمی)",
    "work_experience_years": "سال‌های سابقه کاری (قدیمی)",
    "employment_status": "وضعیت شغلی (قدیمی)",
    "international_experience": "سابقه بین‌المللی (قدیمی)",
    "financial_status": "وضعیت مالی (قدیمی)",
    "business_status": "وضعیت کسب‌وکار (قدیمی)",
    "immigration_goal": "هدف مهاجرت (قدیمی)",
    "canadian_connection": "ارتباط با کانادا (قدیمی)",
    "portfolio": "افتخارات (قدیمی)",
    "business_plan": "طرح کسب‌وکار (قدیمی)",
    "support_letter": "نامه حمایت (قدیمی)",
    "business_idea": "ایده کسب‌وکار (قدیمی)",
    "admission": "پذیرش تحصیلی (قدیمی)",
    "family_status": "رابطه خانوادگی (قدیمی)",
}

PROFILE_PATHWAY_ALIASES = {
    "company_status": "business_status",
    "job_role": "work_experience",
}

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
GREETING_PHRASES = (
    "سلام", "درود", "وقت بخیر", "صبح بخیر", "عصر بخیر", "سلام و عرض ادب",
)
FAREWELL_PHRASES = (
    "خداحافظ", "خدانگهدار", "بدرود", "فعلا", "فعلاً", "روز خوش",
)
THANKS_PHRASES = ("ممنون", "متشکرم", "سپاس", "مرسی", "قدردانم")


def normalize_persian(text: str) -> str:
    """Normalize common Arabic/Persian spelling and punctuation variants."""
    text = text.translate(PERSIAN_DIGITS).lower()
    text = text.replace("ي", "ی").replace("ك", "ک").replace("ۀ", "ه")
    text = text.replace("ـ", "").replace("\u200c", " ")
    text = re.sub(r"[=،.!؟?؛:|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _has_fuzzy_farewell(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if _has_phrase(text, FAREWELL_PHRASES):
        return True
    # Covers common small typos such as «خداحاق=فظ» without requiring exact input.
    return any(
        SequenceMatcher(None, compact, "خداحافظ").ratio() >= 0.72
        or SequenceMatcher(None, token, "خداحافظ").ratio() >= 0.72
        for token in text.split()
    )


def _set(profile: dict[str, Any], field: str, value: Any) -> None:
    if value is not None and str(value).strip():
        profile[field] = value


def _extract_occupation(text: str, normalized: str) -> str | None:
    known_roles = (
        "مهندسی نرم افزار", "مهندس نرم افزار", "برنامه نویس", "توسعه دهنده نرم افزار",
        "هنرمند", "ورزشکار", "پزشک", "پرستار", "معلم", "استاد", "حسابدار",
        "وکیل", "طراح", "مدیر", "کارآفرین", "محقق", "دانشجو", "فریلنسر",
    )
    for role in known_roles:
        if role in normalized:
            if role == "مهندسی نرم افزار":
                return "مهندسی نرم‌افزار"
            if role == "مهندس نرم افزار":
                return "مهندس نرم‌افزار"
            return role

    # General forms such as «شغل من X است» or «من X کار می‌کنم».
    patterns = (
        r"شغل(?:م)?\s*(?:من)?\s*[:：]?\s*(.{2,60}?)(?:\s+است|\s+می باشد|$)",
        r"(?:من\s+)?(.{2,60}?)\s+کار\s*می\s*کنم",
        r"(?:من\s+)?(.{2,60}?)\s+فعالیت\s*می\s*کنم",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" ،؛:")
            # In a multi-clause message, prefer the clause containing the occupation.
            value = re.split(r"\s+و\s+", value)[-1].strip()
            value = re.sub(r"^(?:من\s+)?\d+\s*(?:ساله|سال)\s*", "", value).strip()
            if 1 < len(value) <= 60:
                return value
    return None


def extract_profile_updates(text: str, profile: dict[str, Any]) -> list[str]:
    """Extract explicitly stated profile facts; later statements replace earlier ones."""
    normalized = normalize_persian(text)
    raw = text.translate(PERSIAN_DIGITS)
    changed: list[str] = []

    def put(field: str, value: Any) -> None:
        before = profile.get(field)
        _set(profile, field, value)
        if profile.get(field) != before:
            changed.append(field)

    age_patterns = (
        r"(?:سن(?: من)?|من)\s*(?:من\s*)?(\d{1,2})\s*(?:ساله|سال دارم|سال(?:ه)? هستم|هستم|هست|است)",
        r"(?:من\s+)?(\d{1,2})\s*سالمه",
    )
    for pattern in age_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            put("age", int(match.group(1)))
            break

    child_match = re.search(r"(\d{1,2})\s*(?:فرزند|بچه|کودک)\s*(?:دارم)?", raw)
    if child_match:
        put("children_count", int(child_match.group(1)))
    elif "بدون فرزند" in normalized or "فرزندی ندارم" in normalized:
        put("children_count", 0)

    work_years = re.search(r"(\d{1,2})\s*سال\s*(?:سابقه|تجربه)\s*(?:کاری|کار)?", raw)
    if work_years:
        put("work_experience_years", int(work_years.group(1)))
        put("work_experience", f"{work_years.group(1)} سال سابقه کاری")

    if any(term in normalized for term in ("متاهل", "متأهل", "ازدواج کرده", "همسر دارم")):
        put("marital_status", "متأهل")
    elif any(term in normalized for term in ("مجرد", "ازدواج نکرده")):
        put("marital_status", "مجرد")
    elif any(term in normalized for term in ("طلاق گرفته", "مطلقه", "طلاق")):
        put("marital_status", "مطلقه/جداشده")

    education_terms = ("دیپلم", "کاردانی", "کارشناسی", "لیسانس", "ارشد", "دکتری", "دکترا")
    if any(term in normalized for term in education_terms):
        put("education", text.strip()[:200])

    exam = re.search(r"\b(ielts|tef|celpip|clb)\b\s*(?:نمره|score)?\s*[:=]?\s*(\d+(?:\.\d+)?)", raw, re.I)
    if exam:
        put("language", f"{exam.group(1).upper()} {exam.group(2)}")
    elif any(term in normalized for term in ("زبان انگلیسی", "زبان فرانسه", "زبان فرانسوی", "سطح زبان", "زبانم")):
        put("language", text.strip()[:180])

    occupation = _extract_occupation(raw, normalized)
    if occupation:
        put("occupation", occupation)
        if "work_experience" not in profile or isinstance(profile.get("work_experience"), str):
            put("work_experience", occupation)

    if any(term in normalized for term in ("فریلنسر", "آزادکار")):
        put("employment_status", "فریلنسر")
    elif any(term in normalized for term in ("صاحب بیزینس", "صاحب کسب و کار", "کارآفرین", "شرکت دارم")):
        put("employment_status", "صاحب کسب‌وکار")
        put("business_status", text.strip()[:180])
    elif any(term in normalized for term in ("بیکار", "شاغل نیستم")):
        put("employment_status", "بیکار")
    elif any(term in normalized for term in ("کارمند", "استخدام هستم", "شاغل هستم")):
        put("employment_status", "کارمند")

    if any(term in normalized for term in ("بیزینس پلن", "طرح تجاری", "طرح کسب و کار")):
        put("business_plan", text.strip()[:240])
    if any(term in normalized for term in ("ایده کسب و کار", "ایده بیزینس", "استارتاپ من", "ایده نوآورانه")):
        put("business_idea", text.strip()[:240])
    if any(term in normalized for term in ("نامه حمایت", "حمایت سازمان", "designated organization")):
        put("support_letter", text.strip()[:180])
    if any(term in normalized for term in ("نمونه کار", "افتخار", "مدال", "رزومه هنری", "رزومه ورزشی")):
        put("portfolio", text.strip()[:240])
    if any(term in normalized for term in ("سرمایه", "تمکن مالی", "دارایی", "بودجه")):
        put("financial_status", text.strip()[:180])
    if any(term in normalized for term in ("شرکت", "مدیر", "مدیریتی", "کارآفرینی")):
        put("business_experience", text.strip()[:180])
    if any(term in normalized for term in ("جاب آفر", "پیشنهاد شغلی", "فامیل در کانادا", "اقامت کانادا", "شهروند کانادا")):
        put("canadian_connection", text.strip()[:180])
    if any(term in normalized for term in ("کانادا کار", "کار در کانادا", "تحصیل در کانادا", "خارج از کشور")):
        put("international_experience", text.strip()[:180])
    if any(term in normalized for term in ("اقامت دائم", "گرفتن pr", "کارمند شدن", "راه اندازی بیزینس", "فریلنس", "مسیر مهاجرت")):
        put("immigration_goal", text.strip()[:180])
    if any(term in normalized for term in ("پذیرش دانشگاه", "پذیرش تحصیلی", "نامه پذیرش")):
        put("admission", text.strip()[:180])
    if any(term in normalized for term in ("همسرم کانادا", "فرزندم کانادا", "والدینم کانادا", "اسپانسر", "خواهرم کانادا", "برادرم کانادا")):
        put("family_status", text.strip()[:180])
    if any(term in normalized for term in ("شغل من", "عنوان شغلی", "حوزه تخصصی", "occupation")) and not occupation:
        put("occupation", text.strip()[:180])

    # De-duplicate while preserving field order.
    return list(dict.fromkeys(changed))


def analyze_turn(text: str, session: dict[str, Any]) -> dict[str, Any]:
    """Update dialog state/profile and return multi-label intents for this turn."""
    state = session.setdefault("dialog_state", {})
    profile = session.setdefault("profile", {})
    normalized = normalize_persian(text)
    changed_fields = extract_profile_updates(text, profile)

    greeting = _has_phrase(normalized, GREETING_PHRASES)
    farewell = _has_fuzzy_farewell(normalized)
    gratitude = _has_phrase(normalized, THANKS_PHRASES)
    question = bool(
        re.search(r"[؟?]", text)
        or any(token in normalized.split() for token in ("چطور", "چگونه", "چی", "چرا", "کدام", "آیا", "چه"))
    )
    information_request = any(
        phrase in normalized
        for phrase in (
            "راجب", "راجع به", "درباره", "اطلاعات", "توضیح بده", "کمک کن",
            "راهنمایی کن", "بگو", "میخوام بدونم", "می خواهم بدانم", "میشه",
            "می تونی", "می توانی", "نظرت درباره",
        )
    )
    assessment = any(
        phrase in normalized
        for phrase in (
            "مسیر مهاجرت", "مسیر مناسب", "دنبال اقامت دائم", "اقامت دائم کانادا",
            "میتونم مهاجرت", "می توانم مهاجرت", "شرایط من", "ارزیابی", "کدام مسیر",
            "مناسبم", "برای من مناسبه", "چه مسیری",
        )
    )

    intents: list[str] = []
    if greeting:
        intents.append("greeting")
    if changed_fields:
        intents.append("profile_update")
    if question or information_request:
        intents.append("information_request")
    if assessment:
        intents.append("pathway_assessment")
    if gratitude:
        intents.append("gratitude")
    if farewell:
        intents.append("farewell")
    if state.get("pending_fields") and changed_fields:
        intents.append("follow_up")

    has_substantive_task = bool(question or information_request or assessment)
    profile_only = bool(changed_fields) and not has_substantive_task
    greeting_only = greeting and not (changed_fields or question or information_request or assessment)
    thanks_only = gratitude and not (changed_fields or question or information_request or assessment or farewell)
    farewell_only = farewell and not (changed_fields or question or information_request or assessment)
    should_retrieve = bool(question or information_request or assessment)

    first_response = not state.get("greeted", False)
    state["greeted"] = True
    state["last_intents"] = intents
    state["status"] = "closing" if farewell else "active"

    # A user has replied to the current follow-up; clear only fields now present.
    if state.get("pending_fields") and changed_fields:
        state["pending_fields"] = [
            field for field in state["pending_fields"] if profile.get(field) in (None, "", [], {})
        ]

    return {
        "intents": intents,
        "changed_fields": changed_fields,
        "greeting": greeting,
        "farewell": farewell,
        "gratitude": gratitude,
        "question": question,
        "information_request": information_request,
        "assessment": assessment,
        "should_retrieve": should_retrieve,
        "profile_only": profile_only,
        "greeting_only": greeting_only,
        "thanks_only": thanks_only,
        "farewell_only": farewell_only,
        "first_response": first_response,
        "profile": profile,
        "state": state,
    }


class ConversationPath:
    """Run intent recognition and evidence-checked LLM profile extraction."""

    def __init__(self, profile_extractor: ProfileExtractor | None = None) -> None:
        self.profile_extractor = profile_extractor

    def run(self, request, context: dict[str, Any]) -> dict[str, Any]:
        turn = analyze_turn(request.text, context["session"])
        profile = turn["profile"]
        legacy_migrations = {
            "education": "education_resume",
            "portfolio": "research_achievements",
            "business_plan": "business_startup_details",
            "business_idea": "business_startup_details",
            "financial_status": "budget",
            "canadian_connection": "destination_connections",
            "business_status": "business_experience",
            "support_letter": "business_startup_details",
            "admission": "education_resume",
            "family_status": "family_members",
            "international_experience": "previous_immigration",
        }
        for old, new in legacy_migrations.items():
            value = profile.pop(old, None)
            if value and new not in profile:
                profile[new] = {"summary": value} if new == "budget" else value
                turn["changed_fields"].append(new)
        old_language = profile.pop("language", None)
        if old_language and "language_profile" not in profile:
            profile["language_profile"] = {"summary": old_language}
            turn["changed_fields"].append("language_profile")
        old_children = profile.pop("children_count", None)
        if old_children is not None:
            if not isinstance(profile.get("family_members"), dict):
                profile["family_members"] = {"summary": profile["family_members"]} if profile.get("family_members") else {}
            profile["family_members"]["children_count"] = old_children
            turn["changed_fields"].append("family_members")
        old_years = profile.pop("work_experience_years", None)
        if old_years is not None and not profile.get("work_experience"):
            profile["work_experience"] = f"{old_years} سال سابقه کاری"
            turn["changed_fields"].append("work_experience")
        old_employment = profile.pop("employment_status", None)
        if old_employment and not profile.get("work_experience"):
            profile["work_experience"] = old_employment
            turn["changed_fields"].append("work_experience")
        # The legacy goal field mixed several concepts; don't mislabel it as fact.
        profile.pop("immigration_goal", None)

        if self.profile_extractor and not any(
            turn.get(flag) for flag in ("greeting_only", "thanks_only", "farewell_only")
        ):
            try:
                updates = self.profile_extractor.extract_updates(
                    request.text, turn["state"].get("pending_fields", [])
                )
                changed = self.profile_extractor.apply(profile, updates)
                turn["changed_fields"] = list(dict.fromkeys(turn["changed_fields"] + changed))
            except Exception as exc:
                logger.warning("LLM profile extraction failed (%s): %s", type(exc).__name__, exc)

        if turn["changed_fields"]:
            if "profile_update" not in turn["intents"]:
                turn["intents"].append("profile_update")
            turn["profile_only"] = bool(turn["changed_fields"]) and not bool(
                turn.get("question") or turn.get("information_request") or turn.get("assessment")
            )
            if turn["state"].get("pending_fields"):
                turn["state"]["pending_fields"] = [
                    field for field in turn["state"]["pending_fields"]
                    if profile.get(field) in (None, "", [], {})
                ]
        return turn
