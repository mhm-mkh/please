"""Create the final response using an optional configured LLM and retrieved evidence."""

import re
import logging
from typing import Any

from ..paths.conversation_path import PROFILE_FIELDS
from ..request_models import AssistantResponse, MessageRequest, PathResult

logger = logging.getLogger(__name__)


class ResponseCreator:
    def __init__(self, llm_client: Any, prompt_builder: Any, validator: Any) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.validator = validator

    @staticmethod
    def _link_intent(question: str) -> str | None:
        from ..paths.conversation_path import normalize_persian
        text = normalize_persian(question)
        if any(term in text for term in ("امتیاز", "crs", "محاسبه", "اکسپرس انتری")):
            return "express_entry_points"
        if any(term in text for term in (
            "آیا می توانم مهاجرت", "میتوانم مهاجرت", "واجد شرایط", "شرایط من",
            "ارزیابی", "کدام مسیر", "مسیر مناسب", "دنبال اقامت دائم",
        )):
            return "assessment"
        return None

    def _links(self, question: str, results: list[dict]) -> list[dict[str, str]]:
        requested_intent = self._link_intent(question)
        links, seen = [], set()
        pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
        for result in results:
            for link in result.get("links", []):
                if not isinstance(link, dict) or not link.get("url") or link.get("reliable") is not True:
                    continue
                intent = link.get("intent")
                if requested_intent and intent and intent != requested_intent:
                    continue
                url = str(link["url"])
                if url not in seen:
                    seen.add(url)
                    links.append({"label": str(link.get("label") or "اطلاعات بیشتر"), "url": url})
            for label, url in pattern.findall(result.get("text", "")):
                if url not in seen:
                    seen.add(url)
                    links.append({"label": label, "url": url})
        return links

    @staticmethod
    def _field_labels(fields: list[str]) -> str:
        return "، ".join(PROFILE_FIELDS.get(field, field) for field in fields)

    def _template_answer(self, request, turn, results, profile_data) -> str:
        prefix = "سلام، خوش آمدید. " if turn.get("first_response") else ""
        if turn.get("greeting_only"):
            return "سلام، خوش آمدید. لطفاً پرسش یا اطلاعاتی را که مایلید بررسی شود مطرح کنید."
        if turn.get("thanks_only"):
            return prefix + "از لطف شما سپاسگزارم. اگر پرسش دیگری دارید، بفرمایید."
        if turn.get("farewell_only"):
            return prefix + "از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."

        recommendations = profile_data.get("recommendations", [])
        profile = profile_data.get("profile", {})
        changed = turn.get("changed_fields", [])
        if turn.get("profile_only"):
            answer = prefix + f"اطلاعات مربوط به «{self._field_labels(changed)}» در نمایه موقت این گفت‌وگو ثبت شد."
            if "follow_up" in turn.get("intents", []) and recommendations:
                best = recommendations[0]
                answer += f"\n\nبا اطلاعات جدید، تطبیق اولیه مسیر «{best['name']}» {best['score']:.0%} است؛ این عدد تأیید صلاحیت نیست."
                if best["missing"]:
                    answer += "\nبرای ادامه بررسی، لطفاً این اطلاعات را هم ارائه کنید: " + "؛ ".join(best["missing"][:3]) + "."
                    turn["state"]["pending_fields"] = best["missing_fields"]
                else:
                    turn["state"]["pending_fields"] = []
            if turn.get("farewell"):
                answer += " از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
                turn["state"]["pending_fields"] = []
            elif turn.get("gratitude"):
                answer += " از لطف شما هم سپاسگزارم."
            return answer

        if not turn.get("should_retrieve"):
            answer = prefix + "متوجه شدم. اگر مایلید، پرسش یا اطلاعات بیشتری مطرح کنید تا راهنمایی‌تان کنم."
            if turn.get("farewell"):
                answer += " از گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
            return answer

        answer = (prefix + "این مطالب از میان بخش‌های مرتبط داده‌های موجود بازیابی شده‌اند.") if results else (
            prefix + "اطلاعات مرتبط کافی برای پاسخ به این پرسش در داده‌های موجود پیدا نکردم. لطفاً موضوع را مشخص‌تر کنید."
        )
        if changed:
            answer += "\n\nاطلاعات «" + self._field_labels(changed) + "» نیز در نمایه موقت ثبت شد."
        if turn.get("assessment"):
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
                    turn["state"]["pending_fields"] = best["missing_fields"]
            else:
                answer += "\nبرای پیشنهاد مسیر، لطفاً درباره سن، تحصیلات، زبان، سابقه کاری، وضعیت مالی و ارتباط با کانادا اطلاعات بیشتری بدهید."
                turn["state"]["pending_fields"] = ["age", "education", "language", "work_experience", "financial_status", "canadian_connection"]
        if turn.get("gratitude"):
            answer += "\n\nاز لطف شما سپاسگزارم."
        if turn.get("farewell"):
            answer += "\n\nاز گفت‌وگو با شما سپاسگزارم. روز خوبی داشته باشید."
            turn["state"]["pending_fields"] = []
        return answer

    def create(self, request: MessageRequest, path_results: list[PathResult], turn: dict[str, Any], progress=None) -> AssistantResponse:
        profile_data = next((item.data for item in path_results if item.path == "profile"), {})
        knowledge_data = next((item.data for item in path_results if item.path == "knowledge"), {})
        results = knowledge_data.get("results", [])
        answer = None
        status = "template_fallback"
        if turn.get("should_retrieve") and self.llm_client.configured:
            evidence = results
            try:
                if progress:
                    progress("در حال ساخت پاسخ با مدل زبانی…")
                prompt = self.prompt_builder.build(request, path_results)
                candidate = self.llm_client.generate(prompt)
                evidence_for_validation = [{"id": row.get("id")} for row in evidence]
                if self.validator.validate(candidate, evidence_for_validation):
                    answer = candidate
                    status = "llm"
                else:
                    logger.warning("LLM response failed citation validation; using template fallback")
                    status = "llm_fallback"
            except Exception as exc:
                # Retrieval evidence remains available; response falls back to the safe template.
                logger.warning("LLM generation failed (%s): %s; using template fallback", type(exc).__name__, exc)
                status = "llm_fallback"
        if answer is None:
            answer = self._template_answer(request, turn, results, profile_data)

        return AssistantResponse(
            answer=answer,
            citations=[{"id": item.get("id"), "topic": item.get("topic")} for item in results],
            intents=turn.get("intents", []),
            profile_updates=[{"field": field, "value": profile_data.get("profile", {}).get(field)} for field in turn.get("changed_fields", [])],
            pathway_assessment=profile_data.get("recommendations", [])[:1] if turn.get("assessment") else None,
            status=status,
            results=results,
            links=self._links(request.text, results),
            recommendations=profile_data.get("recommendations", []),
            profile=profile_data.get("profile", {}),
        )
