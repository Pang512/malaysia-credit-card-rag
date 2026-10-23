"""Streamlit app for the Malaysian credit card assistant."""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import Agent  # noqa: E402

st.set_page_config(page_title="Malaysian Credit Card Assistant", page_icon="💳", layout="wide")

EXAMPLES = [
    "I earn RM4,000 a month. Which Islamic cards can I apply for, and which have no annual fee?",
    "Which cards have the lowest overseas transaction fee?",
    "What is Ta'widh on the Bank Muamalat Credit Card-i?",
    "Compare the annual fees of the OCBC 365 Mastercard and PB Quantum.",
]
STEP_LABELS = {"query_catalog": "Searching the card catalog", "search_documents": "Reading the PDS documents"}
TABLE_COLUMNS = ["bank", "card_name", "card_type", "annual_fee_principal_rm", "annual_fee_supplementary_rm",
                 "annual_fee_note", "min_annual_income_rm", "overseas_fee_pct", "best_rate_pct",
                 "highest_rate_pct", "late_payment_cap_rm", "benefits_text", "valid_as_at"]


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_agent():
    return Agent()  # embedding model, FAISS index, BM25 and catalog: loaded once, reused for every question


def show_sources(result, agent):
    cited = result["cited"]
    if not cited:
        return
    with st.expander(f"Sources ({len(cited)} cited)"):
        for n in cited:
            s = result["sources"][n - 1]
            pages = ", ".join(map(str, s["pages"]))
            st.markdown(f"**[{n}] {s['bank']} | {s['title']}** | page {pages} | "
                        f"valid as at {s.get('valid_as_at') or 'unknown'}")
            if s["kind"] == "document":
                text = next(c["text"] for c in agent.retriever.chunks if c["chunk_id"] == s["chunk_id"])
                st.caption(text[:600] + ("..." if len(text) > 600 else ""))
            else:
                st.caption("From the verified card catalog (extracted from the PDS and checked against the PDF).")


agent = load_agent()

with st.sidebar:
    st.header("Try an example")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.divider()
    st.caption("General information from official Product Disclosure Sheets. "
               "Not financial advice. Always confirm details with the bank.")

tab_ask, tab_cards, tab_about = st.tabs(["💬 Ask", "🗂️ Card explorer", "ℹ️ About"])

# ---------------- Ask ----------------
with tab_ask:
    st.title("💳 Malaysian Credit Card Assistant")
    st.caption("Answers grounded in official Product Disclosure Sheets from 8 banks, with citations.")

    if "history" not in st.session_state:
        st.session_state.history = []
    for turn in st.session_state.history:  # redraw earlier turns (Streamlit reruns the script each time)
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["result"]["answer"])
            show_sources(turn["result"], agent)

    question = st.chat_input("Ask about fees, eligibility, Islamic cards...") or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                def on_step(tool, args):
                    status.write(f"{STEP_LABELS.get(tool, tool)}: `{args}`")

                t0 = time.perf_counter()
                try:
                    result = agent.answer(question, on_step=on_step)
                except Exception as e:
                    status.update(label="Something went wrong", state="error")
                    st.error(f"The assistant is unavailable right now. Please try again in a minute. ({e})")
                    st.stop()
                status.update(label=f"Done in {time.perf_counter() - t0:.1f}s", state="complete", expanded=False)
            st.markdown(result["answer"])
            show_sources(result, agent)
        st.session_state.history.append({"question": question, "result": result})

# ---------------- Card explorer ----------------
with tab_cards:
    st.subheader("All cards in the catalog")
    df = agent.catalog.df
    c1, c2, c3, c4 = st.columns(4)
    banks = c1.multiselect("Bank", sorted(df["bank"].unique()))
    types = c2.multiselect("Type", sorted(df["card_type"].unique()))
    income = c3.number_input("My annual income (RM)", min_value=0, value=0, step=1000, help="0 = no income filter")
    max_fee = c4.number_input("Max principal annual fee (RM)", min_value=0, value=2000, step=50)

    view = df
    if banks:
        view = view[view["bank"].isin(banks)]
    if types:
        view = view[view["card_type"].isin(types)]
    if income:
        view = view[view["min_annual_income_rm"].isna() | (view["min_annual_income_rm"] <= income)]
    view = view[view["annual_fee_principal_rm"].fillna(0) <= max_fee]

    st.dataframe(view[TABLE_COLUMNS], hide_index=True, use_container_width=True)
    st.caption(f"{len(view)} of {len(df)} cards. A blank cell means the PDS does not state that value. "
               "Annual fee 0 means waived or free: check the note for conditions.")

# ---------------- About ----------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        "- An **agent** (Gemini with tool calling) decides how to answer each question.\n"
        "- **Card catalog:** 51 cards with fees, income rules and rates, extracted from the PDS with an LLM "
        "and verified against the PDFs. Used for lists, filters and comparisons.\n"
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
        "- PDS documents rarely state rewards, cashback rates, lounge access or miles.\n"
        "- Fees change; check each document's 'valid as at' date and confirm with the bank.\n"
        "- This is general information, not financial advice."
    )