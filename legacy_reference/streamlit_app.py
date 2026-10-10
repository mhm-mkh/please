"""Streamlit UI for the modular Persian-first Canada immigration assistant."""

from __future__ import annotations

from pathlib import Path
import time

import streamlit as st

from backend import handle_message, load_pathways
from dialog_management import PROFILE_FIELDS
from rag import DEFAULT_EMBEDDING_MODEL, load_chunks, load_precomputed_index
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "canada.jsonl"
PATHWAYS_PATH = ROOT / "pathways.json"

st.set_page_config(page_title="راهنمای مهاجرت کانادا", page_icon="🇨🇦", layout="wide")
st.markdown(
    """
    <style>
    /* Keep layout components intact; apply RTL to text containers only. */
    [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
        unicode-bidi: plaintext;
    }

    [data-testid="stMarkdownContainer"] :is(p, li, h1, h2, h3, h4, blockquote, td, th),
    [data-testid="stChatMessage"] :is(p, li, h1, h2, h3, h4, blockquote) {
        direction: rtl;
        text-align: right;
        unicode-bidi: plaintext;
    }

    /* Isolate inline LTR content so English terms, numbers, and URLs don't
       reorder adjacent Persian punctuation. */
    [data-testid="stMarkdownContainer"] :is(code, pre),
    [data-testid="stChatMessage"] :is(code, pre) {
        direction: ltr;
        text-align: left;
        unicode-bidi: isolate;
    }
    [data-testid="stMarkdownContainer"] a,
    [data-testid="stChatMessage"] a {
        unicode-bidi: isolate;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        direction: rtl;
        text-align: right;
        unicode-bidi: plaintext;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("دستیار اطلاعات مهاجرت کانادا")
st.caption("پاسخ‌ها بر پایه داده‌های بارگذاری‌شده هستند؛ برای تصمیم نهایی، شرایط را از منبع رسمی نیز بررسی کنید.")


@st.cache_resource(show_spinner="در حال بارگذاری داده‌ها و مدل زبانی…")
def load_rag(data_path: str, model_name: str):
    started = time.perf_counter()
    status = st.empty()
    progress_bar = st.progress(0, text="خواندن داده‌های دانش…")
    chunks = load_chunks(data_path)
    print(f"[startup] Loaded {len(chunks)} text chunks in {time.perf_counter() - started:.1f}s", flush=True)
    if not chunks:
        progress_bar.empty()
        status.warning("فایل دانش رکورد متنی ندارد؛ برنامه بدون جست‌وجوی دانش ادامه می‌یابد.")
        return chunks, None, None
    status.info("بارگذاری مدل embedding برای پرسش‌ها…")
    progress_bar.progress(0.1, text="بارگذاری مدل embedding برای پرسش‌ها…")
    embedder = SentenceTransformer(model_name)
    print(f"[startup] Embedding model ready in {time.perf_counter() - started:.1f}s", flush=True)
    status.info("بارگذاری نمایه آماده جست‌وجو…")
    progress_bar.progress(0.9, text="بارگذاری نمایه آماده جست‌وجو…")
    index = load_precomputed_index(
        chunks,
        model_name,
        ROOT / "data" / "canada.faiss",
        ROOT / "data" / "canada.index.json",
        expected_dimension=embedder.get_sentence_embedding_dimension(),
    )
    print(f"[startup] Loaded precomputed FAISS index ({index.ntotal} chunks)", flush=True)
    progress_bar.empty()
    status.empty()
    return chunks, embedder, index


try:
    records, embedder, vector_index = load_rag(str(DATA_PATH), DEFAULT_EMBEDDING_MODEL)
    pathways = load_pathways(PATHWAYS_PATH)
except Exception as exc:
    st.error(f"بارگذاری برنامه انجام نشد: {exc}")
    st.info("فایل JSONL را در مسیر data/canada.jsonl قرار دهید و وابستگی‌های requirements.txt را نصب کنید.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = {}
if "dialog_state" not in st.session_state:
    st.session_state.dialog_state = {}

with st.sidebar:
    st.subheader("تنظیمات بازیابی")
    min_similarity = st.slider(
        "حداقل شباهت بازیابی", 0.0, 1.0, 0.35, 0.01,
        help="امتیاز شباهت برای رتبه‌بندی است و احتمال صحت پاسخ نیست.",
    )
    if records:
        result_count = st.slider("تعداد بخش‌های بازیابی‌شده", 1, min(10, len(records)), min(3, len(records)))
    else:
        result_count = 3
        st.info("فایل دانش هنوز خالی است؛ گفت‌وگو، تشخیص نیت و نمایه موقت فعال‌اند، اما جست‌وجوی اطلاعات تا افزودن داده کار نمی‌کند.")
    st.caption(f"تعداد بخش‌های دانش: {len(records)}")

    st.divider()
    st.subheader("نمایه موقت کاربر")
    st.caption("اطلاعات فقط در نشست فعلی نگهداری می‌شود و با تازه‌سازی نشست پاک می‌شود.")
    profile = st.session_state.profile
    populated = [(key, label, profile[key]) for key, label in PROFILE_FIELDS.items() if profile.get(key) not in (None, "", [], {})]
    if populated:
        for _, label, value in populated:
            st.markdown(f"**{label}:** {value}")
    else:
        st.caption("هنوز اطلاعاتی از گفت‌وگو ثبت نشده است.")

    if pathways:
        from backend import recommend_pathways
        recommendations = recommend_pathways(profile, pathways)
        st.divider()
        st.subheader("مسیرهای قابل بررسی")
        displayed = [item for item in recommendations if item["suggested"]]
        if displayed:
            for item in displayed[:3]:
                st.markdown(f"**{item['name']}** · تطبیق اولیه {item['score']:.0%}")
                st.caption(f"خروجی: {item['outcome']} | وضعیت داده: {item['availability']}")
                if item["missing"]:
                    st.caption("اطلاعات تکمیلی: " + "، ".join(item["missing"][:3]))
        else:
            st.caption("برای پیشنهاد اولیه، اطلاعات بیشتری از کاربر لازم است.")
        st.caption("درصد تطبیق، بررسی اولیه اطلاعات است و به معنی احراز شرایط نیست.")


def render_message(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for link in message.get("links", []):
            st.markdown(f"[{link['label']}]({link['url']})")
        for result in message.get("results", []):
            title = f"{result['topic']} · شباهت {result['similarity']:.3f}"
            with st.expander(title):
                path = result.get("section_path", [])
                if path:
                    st.caption(" > ".join(path))
                if result.get("requires_current_verification"):
                    st.warning("این بخش برای اطمینان از به‌روز بودن نیاز به بررسی دارد.")
                st.markdown("**بخش بازیابی‌شده**")
                st.write(result["text"])
                if result.get("context_group_id"):
                    st.markdown("**متن کامل گروه زمینه**")
                    st.write(result["context_text"])


for saved_message in st.session_state.messages:
    render_message(saved_message)

question = st.chat_input("پرسش خود را به فارسی بنویسید…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])
    response = handle_message(
        question=question,
        session=st.session_state,
        records=records,
        embedder=embedder,
        index=vector_index,
        pathways=pathways,
        result_count=result_count,
        min_similarity=min_similarity,
    )
    assistant_message = {
        "role": "assistant",
        "content": response["answer"],
        "results": response["results"],
        "links": response["links"],
    }
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)
    # Re-render the sidebar immediately with profile fields extracted from this turn.
    st.rerun()
