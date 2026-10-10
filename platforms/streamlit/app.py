"""Streamlit platform adapter and UI for the shared modular backend."""

from __future__ import annotations

import uuid
import time
import json

import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from backend.message_handler import MessageHandler
from backend.paths.combined_path import CombinedPath
from backend.paths.conversation_path import ConversationPath, PROFILE_FIELDS
from backend.paths.knowledge_path import KnowledgePath
from backend.paths.profile_path import ProfilePath
from backend.request_models import MessageRequest
from backend.response.prompt_builder import PromptBuilder
from backend.response.response_creator import ResponseCreator
from backend.response.response_validator import ResponseValidator
from config.settings import (
    DEFAULT_MIN_SIMILARITY, DEFAULT_RESULT_COUNT, EMBEDDING_MODEL,
    FAISS_INDEX_PATH, FAISS_METADATA_PATH, KNOWLEDGE_PATH, PATHWAYS_PATH,
    PATHWAY_SCORE_THRESHOLD,
)
from services.llm.client import LLMClient
from services.profile.extractor import ProfileExtractor
from services.pathways.assessor import PathwayAssessor
from services.pathways.pathway_loader import load_pathways
from services.rag.corpus_loader import load_corpus
from services.rag.index_loader import load_index
from services.rag.retriever import Retriever


@st.cache_resource(show_spinner=False)
def load_shared_resources():
    records = load_corpus(KNOWLEDGE_PATH)
    pathways = load_pathways(PATHWAYS_PATH)
    if not records:
        return {"records": [], "retriever": None, "pathways": pathways}
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    index = load_index(
        records, EMBEDDING_MODEL, FAISS_INDEX_PATH, FAISS_METADATA_PATH,
        expected_dimension=embedder.get_sentence_embedding_dimension(),
    )
    return {"records": records, "retriever": Retriever(records, embedder, index), "pathways": pathways}


def create_handler(resources: dict) -> MessageHandler:
    llm_client = LLMClient()
    response_creator = ResponseCreator(llm_client, PromptBuilder(), ResponseValidator())
    return MessageHandler(
        paths={
            "conversation": ConversationPath(ProfileExtractor(llm_client)),
            "profile": ProfilePath(PathwayAssessor(PATHWAY_SCORE_THRESHOLD)),
            "knowledge": KnowledgePath(resources["retriever"]),
            "combined": CombinedPath(),
        },
        response_creator=response_creator,
    )


def _render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for link in message.get("links", []):
            st.markdown(f"[{link['label']}]({link['url']})")
        for result in message.get("results", []):
            title = f"{result.get('topic', 'موضوع مرتبط')} · شباهت {result.get('similarity', 0):.3f}"
            with st.expander(title):
                path = result.get("section_path", [])
                if path:
                    st.caption(" > ".join(path))
                if result.get("requires_current_verification"):
                    st.warning("این بخش برای اطمینان از به‌روز بودن نیاز به بررسی دارد.")
                st.markdown("**بخش بازیابی‌شده**")
                st.write(result.get("text", ""))
                if result.get("context_group_id"):
                    st.markdown("**متن کامل گروه زمینه**")
                    st.write(result.get("context_text", ""))


def _profile_export_text(profile: dict) -> str:
    lines = ["خلاصه اطلاعات متقاضی برای بررسی مشاور", "", "این خلاصه بر اساس گفته‌های کاربر تهیه شده و ارزیابی صلاحیت قانونی نیست.", ""]
    for field, value in profile.items():
        label = PROFILE_FIELDS.get(field, field)
        rendered = json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)
        lines.extend((f"{label}:", rendered, ""))
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="راهنمای مهاجرت کانادا", page_icon="🇨🇦", layout="wide")
    st.markdown(
        """<style>
        [data-testid="stMarkdownContainer"], [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {direction:rtl;text-align:right;unicode-bidi:plaintext;}
        [data-testid="stMarkdownContainer"] :is(p,li,h1,h2,h3,h4,blockquote,td,th),
        [data-testid="stChatMessage"] :is(p,li,h1,h2,h3,h4,blockquote) {direction:rtl;text-align:right;unicode-bidi:plaintext;}
        [data-testid="stMarkdownContainer"] :is(code,pre), [data-testid="stChatMessage"] :is(code,pre) {direction:ltr;text-align:left;unicode-bidi:isolate;}
        [data-testid="stMarkdownContainer"] a, [data-testid="stChatMessage"] a {unicode-bidi:isolate;}
        [data-testid="stChatInput"] textarea {direction:rtl;text-align:right;unicode-bidi:plaintext;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("دستیار اطلاعات مهاجرت کانادا")
    st.caption("پاسخ‌ها بر پایه داده‌های بارگذاری‌شده هستند؛ برای تصمیم نهایی، شرایط را از منبع رسمی نیز بررسی کنید.")

    try:
        with st.spinner("در حال بارگذاری داده‌ها و مدل جست‌وجو…"):
            resources = load_shared_resources()
        handler = create_handler(resources)
    except Exception as exc:
        st.error(f"بارگذاری برنامه انجام نشد: {exc}")
        st.info("بررسی کنید فایل‌های دانش، FAISS و metadata در مسیر data/ قرار داشته باشند و requirements.txt نصب شده باشد.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "dialog_state" not in st.session_state:
        st.session_state.dialog_state = {}
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = uuid.uuid4().hex

    with st.sidebar:
        st.subheader("تنظیمات بازیابی")
        min_similarity = st.slider("حداقل شباهت بازیابی", 0.0, 1.0, DEFAULT_MIN_SIMILARITY, 0.01,
                                   help="امتیاز شباهت برای رتبه‌بندی است و احتمال صحت پاسخ نیست.")
        count = len(resources["records"])
        result_count = st.slider("تعداد بخش‌های بازیابی‌شده", 1, min(10, count), min(DEFAULT_RESULT_COUNT, count)) if count else DEFAULT_RESULT_COUNT
        st.caption(f"تعداد بخش‌های دانش: {count}")
        if not LLMClient().configured:
            st.info("LLM تنظیم نشده است؛ پاسخ پایه همراه با منابع بازیابی‌شده نمایش داده می‌شود. برای تولید پاسخ، LLM_API_KEY و LLM_MODEL را در تنظیمات استقرار وارد کنید.")

        st.divider()
        st.subheader("آزمایش مستقیم LLM")
        st.caption("این آزمون RAG و بازیابی دانش را دور می‌زند و فقط اتصال مدل را بررسی می‌کند.")
        if st.button("ارسال پیام آزمایشی به LLM", key="llm_connection_test"):
            test_client = LLMClient()
            if not test_client.configured:
                st.error("LLM_API_KEY یا LLM_MODEL تنظیم نشده است.")
            else:
                started = time.perf_counter()
                try:
                    with st.spinner("در حال آزمایش اتصال مستقیم به LLM…"):
                        # Match the provider's published OpenAI SDK call while
                        # bypassing this app's custom urllib-based LLM client.
                        sdk_client = OpenAI(
                            base_url=test_client.base_url,
                            api_key=test_client.api_key,
                            timeout=60.0,
                            max_retries=0,
                        )
                        sdk_response = sdk_client.chat.completions.create(
                            model=test_client.model,
                            messages=[
                                {"role": "user", "content": "فقط بنویس: اتصال مدل برقرار است."}
                            ],
                        )
                        test_answer = sdk_response.choices[0].message.content
                    st.success(f"پاسخ دریافت شد · {time.perf_counter() - started:.1f} ثانیه")
                    st.write(test_answer or "مدل پاسخ خالی برگرداند.")
                except Exception as exc:
                    st.error(f"آزمون مستقیم ناموفق بود · {time.perf_counter() - started:.1f} ثانیه")
                    st.code(f"{type(exc).__name__}: {exc}")

        st.divider()
        st.subheader("نمایه موقت کاربر")
        st.caption("برای استخراج اطلاعات، پیام‌های شما به مدل زبانی تنظیم‌شده فرستاده می‌شوند؛ فقط اطلاعاتی پذیرفته می‌شود که با نقل‌قولی از همان پیام همراه باشد.")
        profile = st.session_state.profile
        populated = [(label, profile[key]) for key, label in PROFILE_FIELDS.items() if profile.get(key) not in (None, "", [], {})]
        if populated:
            for label, value in populated:
                st.markdown(f"**{label}:**")
                if isinstance(value, (dict, list)):
                    st.code(json.dumps(value, ensure_ascii=False, indent=2), language="json")
                else:
                    st.write(value)
            st.download_button(
                "دانلود خلاصه برای مشاور",
                data=_profile_export_text(profile).encode("utf-8"),
                file_name="immigration-profile-summary.txt",
                mime="text/plain; charset=utf-8",
                help="فایل فقط روی دستگاه شما دانلود می‌شود؛ ایمیلی ارسال نمی‌شود.",
            )
        else:
            st.caption("هنوز اطلاعاتی از گفت‌وگو ثبت نشده است.")
        st.caption("نمایه فعلاً فقط در نشست جاری ذخیره می‌شود؛ با پایان نشست، ذخیره دائمی یا ارسال ایمیل انجام نمی‌شود.")

        pathways = resources["pathways"]
        if pathways:
            recommendations = PathwayAssessor(PATHWAY_SCORE_THRESHOLD).assess(profile, pathways)
            st.divider()
            st.subheader("مسیرهای قابل بررسی")
            displayed = [item for item in recommendations if item["suggested"]]
            if displayed:
                for item in displayed[:3]:
                    st.markdown(f"**{item['name']}** · تناسب اولیه با اطلاعات و ترجیحات {item['score']:.0%}")
                    st.caption(f"پوشش اطلاعات پروفایل: {item['coverage_score']:.0%}")
                    st.caption(f"خروجی: {item['outcome']} | وضعیت داده: {item['availability']}")
                    if item["missing"]:
                        st.caption("اطلاعات تکمیلی: " + "، ".join(item["missing"][:3]))
            elif not profile.get("destination_country"):
                st.caption("برای شروع پیشنهاد مسیر، ابتدا کشور مقصد را مشخص کنید.")
            elif str(profile.get("destination_country", "")).casefold() not in {"canada", "کانادا"}:
                st.caption("در حال حاضر مسیرهای تعریف‌شده این برنامه فقط برای کانادا هستند.")
            else:
                st.caption("برای پیشنهاد اولیه، اطلاعات بیشتری از کاربر لازم است.")
            st.caption("این امتیاز فقط غربال اولیه بر اساس پوشش اطلاعات و ترجیح اقامت دائم است؛ بودجه، هزینه و زمان پردازش هنوز با مقادیر واقعی هر مسیر مقایسه نمی‌شوند و این نتیجه تأیید صلاحیت نیست.")

    for message in st.session_state.messages:
        _render_message(message)

    question = st.chat_input("پرسش خود را به فارسی بنویسید…")
    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        _render_message(user_message)
        request = MessageRequest(text=question, conversation_id=st.session_state.conversation_id, platform="streamlit")
        status_box = st.status("در حال پردازش پیام…", expanded=True)
        try:
            response = handler.handle(request, {
                "session": st.session_state,
                "records": resources["records"],
                "pathways": resources["pathways"],
                "result_count": result_count,
                "min_similarity": min_similarity,
                "progress": lambda message: status_box.update(label=message, state="running"),
            })
        except Exception as exc:
            status_box.update(label="پردازش پیام ناموفق بود", state="error")
            st.error(f"پردازش پیام با خطا مواجه شد: {exc}")
            st.stop()
        status_box.update(label="پاسخ آماده شد", state="complete", expanded=False)
        assistant_message = {"role": "assistant", "content": response.answer,
                             "results": response.results, "links": response.links}
        st.session_state.messages.append(assistant_message)
        _render_message(assistant_message)
        st.rerun()