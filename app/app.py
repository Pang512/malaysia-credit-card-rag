"""Streamlit app for the Malaysian credit card assistant."""
import sys
import threading
import time
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import Agent  # noqa: E402

st.set_page_config(page_title="Malaysian Credit Card Assistant", page_icon="💳", layout="wide")

PRIORITY_OPTIONS = {"Not sure yet": None, "💰 Lowest cost": "lowest_cost", "🛒 Cashback & everyday spending": "cashback",
                    "✈️ Travel": "travel", "📉 Low interest (may not always pay in full)": "low_interest"}
QUICK_STARTS = [  # (button label, priority label or None, card type or None, question)
    ("💰 Lowest cost", "💰 Lowest cost", None, "Which card should I get? I want to keep costs as low as possible."),
    ("🛒 Cashback", "🛒 Cashback & everyday spending", None, "Which card should I get for everyday spending and cashback?"),
    ("✈️ Travel", "✈️ Travel", None, "Which card should I get for spending overseas?"),
    ("🌙 Islamic", None, "Islamic", "Which Islamic card should I get?"),
]
EXAMPLES = [
    "How does a credit card work, and what fees should I watch out for?",
    "What is Ta'widh on Islamic credit cards?",
    "Compare the annual fees of the OCBC 365 Mastercard and PB Quantum.",
]
STEP_LABELS = {"query_catalog": "Searching the card catalog", "search_documents": "Reading the PDS documents",
               "recommend_cards": "Ranking cards for your priority", "rewrite": "Understood your question as"}
TABLE_COLUMNS = ["bank", "card_name", "card_type", "annual_fee_principal_rm", "annual_fee_supplementary_rm",
                 "annual_fee_note", "min_annual_income_rm", "overseas_fee_pct", "best_rate_pct",
                 "highest_rate_pct", "rate_note", "late_payment_cap_rm", "benefits_text", "valid_as_at"]


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_agent():
    return Agent()  # embedding model, FAISS index, BM25 and catalog: loaded once, shared by every visitor


@st.cache_resource
def daily_usage():
    return {"date": None, "count": 0, "lock": threading.Lock()}  # shared across all visitors


def take_daily_slot(limit):
    """Count one question against the app-wide daily cap. False if today's cap is used up."""
    usage = daily_usage()
    with usage["lock"]:
        if usage["date"] != date.today():
            usage["date"], usage["count"] = date.today(), 0
        if usage["count"] >= limit:
            return False
        usage["count"] += 1
        return True


def source_line(n, s):
    pages = ", ".join(map(str, s["pages"]))
    return f"[{n}] {s['bank']} | {s['title']} | page {pages} | valid as at {s.get('valid_as_at') or 'unknown'}"


def show_sources(result, agent):
    cited = result["cited"]
    if not cited:
        return
    with st.expander(f"Sources ({len(cited)} cited)"):
        for n in cited:
            s = result["sources"][n - 1]
            st.markdown(f"**{source_line(n, s)}**")
            if s["kind"] == "document":
                text = next(c["text"] for c in agent.retriever.chunks if c["chunk_id"] == s["chunk_id"])
                st.caption(text[:600] + ("..." if len(text) > 600 else ""))
            else:
                st.caption("From the verified card catalog (extracted from the PDS and checked against the PDF).")


def transcript_markdown(history):
    lines = ["# Malaysian Credit Card Assistant: conversation", ""]
    for turn in history:
        result = turn["result"]
        lines += [f"**You:** {turn['question']}", "", f"**Assistant:** {result['answer']}", ""]
        lines += [f"- {source_line(n, result['sources'][n - 1])}" for n in result["cited"]]
        lines += [f"\n_({result['timing']['total_s']:.1f}s)_", "", "---", ""]
    return "\n".join(lines)


def quick_start(priority_label, card_type, question):
    """Button callback: runs before the rerun, so it may set the 'About you' widgets."""
    if priority_label:
        st.session_state.priority = priority_label
    if card_type:
        st.session_state.card_type = card_type
    st.session_state.pending = question


def ask(question):
    st.session_state.pending = question


def clear_chat():
    st.session_state.history = []


agent = load_agent()
limits = agent.cfg.get("app", {})
per_session = limits.get("max_questions_per_session", 15)
per_day = limits.get("max_questions_per_day", 100)
st.session_state.setdefault("history", [])
st.session_state.setdefault("asked", 0)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About you")
    st.caption("Optional. The assistant uses these in every answer.")
    income = st.number_input("Monthly income (RM)", min_value=0, step=100, key="income", help="0 = not saying")
    card_type = st.radio("Card type", ["Any", "Conventional", "Islamic"], key="card_type", horizontal=True)
    priority = st.selectbox("Main priority", list(PRIORITY_OPTIONS), key="priority")
    profile = {"monthly_income_rm": income or None,
               "card_type": None if card_type == "Any" else card_type.lower(),
               "priority": PRIORITY_OPTIONS[priority]}

    st.subheader("Help me choose")
    cols = st.columns(2)
    for i, (label, priority_label, type_, question) in enumerate(QUICK_STARTS):
        cols[i % 2].button(label, use_container_width=True, on_click=quick_start,
                           args=(priority_label, type_, question))

    with st.expander("Example questions"):
        for example in EXAMPLES:
            st.button(example, use_container_width=True, on_click=ask, args=(example,))

    st.divider()
    if st.session_state.history:
        st.download_button("⬇️ Download conversation", transcript_markdown(st.session_state.history),
                           file_name="credit-card-chat.md", use_container_width=True)
        st.button("🗑️ Clear conversation", on_click=clear_chat, use_container_width=True)
    st.caption("General information from official Product Disclosure Sheets. Not financial advice. "
               "Always confirm details with the bank. Please don't enter personal details such as your name, "
               "IC number or account numbers.")

tab_ask, tab_cards, tab_about = st.tabs(["💬 Ask", "🗂️ Card explorer", "ℹ️ About"])

# ---------------- Ask ----------------
with tab_ask:
    st.title("💳 Malaysian Credit Card Assistant")
    st.caption("Answers grounded in official Product Disclosure Sheets from 8 banks, with citations.")
    chat = st.container()  # messages go here, so the input box below always sits under the latest answer
    question = st.chat_input("Ask about fees, eligibility, Islamic cards...") or st.session_state.pop("pending", None)
    questions_left = st.empty()  # filled in at the end, after this question is counted

    with chat:
        for turn in st.session_state.history:  # redraw earlier turns (Streamlit reruns the script each time)
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["result"]["answer"])
                show_sources(turn["result"], agent)

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                if st.session_state.asked >= per_session:
                    st.warning("You've reached the question limit for this session. Thanks for trying the assistant!")
                elif not take_daily_slot(per_day):
                    st.warning("The assistant has reached today's question limit. Please come back tomorrow.")
                else:
                    st.session_state.asked += 1
                    result = None
                    with st.status("Thinking...", expanded=True) as status:
                        def on_step(tool, args):
                            status.write(f"{STEP_LABELS.get(tool, tool)}: `{args}`")

                        t0 = time.perf_counter()
                        try:
                            history = [{"question": t["question"], "answer": t["result"]["answer"]}
                                       for t in st.session_state.history]
                            result = agent.answer(question, history=history, profile=profile, on_step=on_step)
                            calls = " + ".join(f"{s:g}s" for s in result["llm_call_s"])
                            status.update(label=f"Done in {time.perf_counter() - t0:.1f}s (Gemini calls: {calls})",
                                          state="complete", expanded=False)
                        except Exception as e:
                            status.update(label="Something went wrong", state="error")
                            st.error(f"The assistant is unavailable right now. Please try again in a minute. ({e})")
                    if result:
                        st.markdown(result["answer"])
                        show_sources(result, agent)
                        st.session_state.history.append({"question": question, "result": result})
    questions_left.caption(f"{max(per_session - st.session_state.asked, 0)} questions left in this session.")

# ---------------- Card explorer ----------------
with tab_cards:
    st.subheader("All cards in the catalog")
    df = agent.catalog.df
    c1, c2, c3, c4 = st.columns(4)
    banks = c1.multiselect("Bank", sorted(df["bank"].unique()))
    types = c2.multiselect("Type", sorted(df["card_type"].unique()))
    annual_income = c3.number_input("My annual income (RM)", min_value=0, value=0, step=1000,
                                    help="0 = no income filter")
    max_fee = c4.number_input("Max principal annual fee (RM)", min_value=0, value=2000, step=50)

    view = df
    if banks:
        view = view[view["bank"].isin(banks)]
    if types:
        view = view[view["card_type"].isin(types)]
    if annual_income:
        view = view[view["min_annual_income_rm"].isna() | (view["min_annual_income_rm"] <= annual_income)]
    view = view[view["annual_fee_principal_rm"].fillna(0) <= max_fee]

    st.dataframe(view[TABLE_COLUMNS], hide_index=True, use_container_width=True)
    st.caption(f"{len(view)} of {len(df)} cards. A blank cell means the PDS does not state that value. "
               "Annual fee 0 means waived or free: check the note for conditions.")

# ---------------- About ----------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "- An **agent** (Gemini with tool calling) decides how to answer each question, using the last few "
        "turns of the conversation and your optional 'About you' settings.\n"
        "- **Card catalog:** 51 cards with fees, income rules and rates, extracted from the PDS with an LLM "
        "and verified against the PDFs. Used for lists, filters and comparisons.\n"
        "- **Recommendations:** eligible cards are filtered and ranked **in code** by your priority "
        "(annual fee, overseas fee or finance rate); the AI only explains the ranking.\n"
        "- **Document search:** hybrid search (BM25 + embeddings) over table-aware PDS text. "
        "Used for explanations and conditions.\n"
        "- Every fact is cited to a bank document and page."
    )
    st.subheader("Coverage")
    coverage = (df.groupby("bank").agg(cards=("card_name", "count"), valid_as_at=("valid_as_at", "first"))
                .reset_index())
    st.dataframe(coverage, hide_index=True)
    st.subheader("Limitations")
    st.markdown(
        "- Covers 8 banks only (no Hong Leong, RHB or AmBank yet).\n"
        "- PDS documents rarely state rewards, cashback rates, lounge access or miles, so 'cashback' and "
        "'travel' recommendations are ranked on costs and fees only.\n"
        "- Minimum income is often not stated in the PDS.\n"
        "- Fees change; check each document's 'valid as at' date and confirm with the bank.\n"
        "- This is general information, not financial advice."
    )
