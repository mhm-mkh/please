"""Score pathway data coverage; this is not a legal eligibility decision."""

from typing import Any


PROFILE_PATHWAY_ALIASES = {
    "company_status": "business_experience",
    "job_role": "occupation",
    "education": "education_resume",
    "language": "language_profile",
    "financial_status": "budget",
    "portfolio": "research_achievements",
    "business_plan": "business_startup_details",
    "business_idea": "business_startup_details",
    "family_status": "family_members",
    "canadian_connection": "destination_relatives",
}


class PathwayAssessor:
    def __init__(self, threshold: float = 0.30) -> None:
        self.threshold = threshold

    def assess(self, profile: dict[str, Any], pathways: list[dict[str, Any]]) -> list[dict[str, Any]]:
        destination = str(profile.get("destination_country", "")).strip().casefold()
        destination_is_canada = "canada" in destination or "کانادا" in destination
        destination_missing = not destination
        # This project currently has pathway definitions for Canada only.
        if destination and not destination_is_canada:
            return []

        recommendations = []
        for pathway in pathways:
            countries = pathway.get("destination_countries", ["Canada"])
            if isinstance(countries, str):
                countries = [countries]
            if destination_is_canada and not any(str(country).casefold() in {"canada", "کانادا"} for country in countries):
                continue

            applicant_type_raw = str(profile.get("immigration_type", "")).casefold()
            if any(word in applicant_type_raw for word in ("business", "startup", "کسب", "استارتاپ", "کارآفرین")):
                applicant_type = "business_or_startup"
            elif any(word in applicant_type_raw for word in ("individual", "فردی", "شخصی")):
                applicant_type = "individual"
            else:
                applicant_type = "either"
            supported_types = pathway.get("applicant_types", [])
            if isinstance(supported_types, str):
                supported_types = [supported_types]
            if applicant_type in {"individual", "business_or_startup"} and supported_types:
                if applicant_type not in supported_types and "either" not in supported_types:
                    continue

            criteria = pathway.get("criteria", [])
            if not isinstance(criteria, list) or not criteria:
                continue
            matched_weight = 0.0
            total_weight = 0.0
            missing = []
            missing_fields = []
            for criterion in criteria:
                if not isinstance(criterion, dict):
                    continue
                weight = float(criterion.get("weight", 1.0))
                total_weight += weight
                field_names = criterion.get("profile_fields", [])
                if isinstance(field_names, str):
                    field_names = [field_names]
                if not isinstance(field_names, list):
                    field_names = []
                has_value = any(
                    profile.get(PROFILE_PATHWAY_ALIASES.get(name, name)) not in (None, "", [], {})
                    for name in field_names
                    if isinstance(name, str)
                )
                if has_value:
                    matched_weight += weight
                else:
                    missing.append(criterion.get("description_fa", criterion.get("id", "اطلاعات تکمیلی")))
                    missing_fields.extend(name for name in field_names if isinstance(name, str))
            coverage = matched_weight / total_weight if total_weight else 0.0
            # Preferences affect ranking, while missing profile facts affect the
            # separate data-coverage score. This is triage, never eligibility.
            preference_score = 0.5
            pr_goal = str(profile.get("permanent_residence_goal", "")).casefold()
            direct_pr = bool(pathway.get("direct_pr", False))
            if pr_goal in {"yes", "بله", "حتما", "حتماً"}:
                preference_score = 1.0 if direct_pr else 0.35
            elif pr_goal in {"no", "نه", "خیر"}:
                preference_score = 0.35 if direct_pr else 1.0
            score = 0.8 * coverage + 0.2 * preference_score
            recommendations.append({
                "id": pathway.get("id", ""),
                "name": pathway.get("name_fa", pathway.get("id", "مسیر")),
                "outcome": pathway.get("outcome", ""),
                "availability": pathway.get("availability", "unknown"),
                "score": score,
                "coverage_score": coverage,
                "preference_score": preference_score,
                "suggested": destination_is_canada and score > self.threshold,
                "needs_destination": destination_missing,
                "matched": matched_weight,
                "total": total_weight,
                "missing": missing,
                "missing_fields": list(dict.fromkeys(missing_fields)),
            })
        return sorted(recommendations, key=lambda item: item["score"], reverse=True)
