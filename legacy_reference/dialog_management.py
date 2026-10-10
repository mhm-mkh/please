"""Persian-first multi-intent dialog and session-profile management.

This module uses deterministic rules and is intentionally easy to extend. It does
not call an LLM: the embedding model is reserved for knowledge retrieval.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


PROFILE_FIELDS = {
    "age": "سن",
    "marital_status": "وضعیت تأهل",
    "children_count": "تعداد فرزند",
    "education": "تحصیلات (مدرک، رشته و کشور)",
    "language": "زبان (انگلیسی/فرانسوی و نمره آزمون)",
    "work_experience": "سابقه کاری و عنوان شغلی",
    "work_experience_years": "تعداد سال سابقه کاری",
    "employment_status": "وضعیت شغلی فعلی",
    "international_experience": "سابقه کار یا تحصیل بین‌المللی/کانادا",
    "financial_status": "وضعیت مالی و سرمایه",
    "business_status": "وضعیت و نقش در شرکت",
    "immigration_goal": "هدف مهاجرت",
    "canadian_connection": "ارتباط با کانادا",
    "portfolio": "نمونه‌کار یا افتخارات",
    "business_experience": "سابقه بیزینسی/مدیریتی",
    "business_plan": "ایده یا بیزینس پلن",
    "occupation": "شغل و حوزه تخصصی",
    "support_letter": "نامه حمایت استارتاپ",
    "business_idea": "ایده کسب‌وکار",
    "admission": "پذیرش تحصیلی",
    "family_status": "رابطه خانوادگی واجد شرایط",
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
