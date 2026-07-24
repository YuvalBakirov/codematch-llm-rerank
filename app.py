"""Interactive demo: does an LLM judge improve embedding-based code-clone search?

Run with:
    streamlit run app.py

Reuses the exact same llm_rerank/ modules as the CLI experiment (data
loading, judge client, reranking, metrics) - this is a viewer on top of
the same tested logic, not a separate reimplementation.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm_rerank.data import (
    build_clone_display_names,
    build_clone_groups,
    build_original_display_names,
    candidates_for_clone,
    load_clone_lookup,
    load_code_lookup,
)
from llm_rerank.evaluate import clone_type_lookup as build_clone_type_lookup
from llm_rerank.judge_client import ClaudeJudgeClient
from llm_rerank.rerank import rerank_one

load_dotenv()

ROOT = Path(__file__).resolve().parent
SCORES_CSV = ROOT / "output/Qwen2.5-Coder-0.5B-pe/global-clone/Qwen2.5-Coder-0.5B-pe_global_clone_search_scores_20.11.2024_12-02-25.csv"
ORIGINAL_CODE_CSV = ROOT / "data/evaluation-datasets/original_code_benchmark_fixed.csv"
TEST_CODE_CSV = ROOT / "data/evaluation-datasets/test_code_benchmark_fixed.csv"
OUTCOMES_CSV = ROOT / "output/llm_rerank_final_results/rerank_outcomes.csv"

st.set_page_config(page_title="CodeMatch: LLM Rerank", layout="wide")


@st.cache_data
def load_everything():
    scores_df = build_clone_groups(str(SCORES_CSV))
    code_lookup = load_code_lookup(str(ORIGINAL_CODE_CSV))
    clone_lookup = load_clone_lookup(str(TEST_CODE_CSV))
    outcomes_df = pd.read_csv(OUTCOMES_CSV)
    original_names = build_original_display_names(str(ORIGINAL_CODE_CSV))
    clone_names = build_clone_display_names(str(TEST_CODE_CSV))
    return scores_df, code_lookup, clone_lookup, outcomes_df, original_names, clone_names


scores_df, code_lookup, clone_lookup, outcomes_df, original_names, clone_names = load_everything()


def show_original(base_code_id: str) -> str:
    return original_names.get(base_code_id, base_code_id)


def show_clone(clone_code_id: str) -> str:
    return clone_names.get(clone_code_id, clone_code_id)


st.title("CodeMatch: does an LLM judge improve embedding-based clone search?")
st.caption(
    "Embedding search (Qwen2.5-Coder-0.5B-pe) already retrieves a top-5 for each query. "
    "This asks Claude to judge each candidate as genuine-clone vs. false-positive and reranks accordingly."
)

with st.expander("New here? How this whole thing works (read this first)", expanded=True):
    st.markdown(
        """
**The problem.** A code-clone search tool takes a snippet of code and tries to find other code
that's really "the same thing" - the same algorithm, possibly renamed, reformatted, or rewritten
in another language. It does this with **embedding similarity**: code is converted into a vector,
and the 5 closest vectors in a database are returned as candidates. The catch: "closest vector"
sometimes means *superficially* similar (same imports, same boilerplate) rather than *actually*
the same logic - this tool measures and tries to fix exactly that gap.

**The 4 difficulty levels (clone types), easiest to hardest:**
- **T1** - near-identical copy (renamed whitespace/comments/formatting only)
- **T2** - renamed variables, functions, or data types, same structure
- **T3** - restructured (statements added/removed/reordered), same logic
- **T4** - same logic, but rewritten from scratch or in a different language - hardest to catch by embedding similarity alone

**What this page shows, in order:**
1. Aggregate accuracy (Hit@1 / Hit@5) - embedding search alone vs. embedding search + an LLM judge on top, split by difficulty level
2. One specific example you pick, so you can see exactly what changed
3. A live button that calls the real Claude API right now, on whichever example you're looking at
        """
    )

# --- Aggregate result -------------------------------------------------------
st.header("1. Aggregate result (160-clone stratified sample)")

outcomes_df["clone_type"] = outcomes_df["clone_code_id"].map(build_clone_type_lookup(scores_df)).fillna("Unknown")

before_hit1 = outcomes_df.groupby("clone_type")["original_hit_at_1"].mean()
after_hit1 = outcomes_df.groupby("clone_type")["reranked_hit_at_1"].mean()

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = range(len(before_hit1))
    width = 0.35
    ax.bar([i - width / 2 for i in x], before_hit1.values * 100, width, label="Embedding-only")
    ax.bar([i + width / 2 for i in x], after_hit1.values * 100, width, label="+ LLM rerank")
    ax.set_xticks(list(x))
    ax.set_xticklabels(before_hit1.index)
    ax.set_ylabel("Hit@1 (%)")
    ax.set_title("Hit@1 by clone type")
    ax.legend()
    st.pyplot(fig)

with col2:
    overall_before = outcomes_df["original_hit_at_1"].mean() * 100
    overall_after = outcomes_df["reranked_hit_at_1"].mean() * 100
    st.metric("Overall Hit@1 (before)", f"{overall_before:.1f}%")
    st.metric("Overall Hit@1 (after rerank)", f"{overall_after:.1f}%", delta=f"{overall_after - overall_before:+.1f}pp")
    st.caption(f"n = {len(outcomes_df)} clones, 0 judge errors (after the tool-use fix)")

# --- Inspect one clone -------------------------------------------------------
st.header("2. Inspect one clone")

improved_ids = outcomes_df[
    (~outcomes_df["original_hit_at_1"]) & outcomes_df["reranked_hit_at_1"]
]["clone_code_id"].tolist()
still_wrong_ids = outcomes_df[
    (~outcomes_df["original_hit_at_1"]) & (~outcomes_df["reranked_hit_at_1"])
]["clone_code_id"].tolist()
all_ids = outcomes_df["clone_code_id"].tolist()

bucket = st.radio(
    "Show me a clone where...",
    ["rerank fixed a miss", "rerank still got it wrong", "any of the 160 sampled clones"],
    horizontal=True,
)
if bucket == "rerank fixed a miss":
    options = improved_ids
elif bucket == "rerank still got it wrong":
    options = still_wrong_ids
else:
    options = all_ids

selected = st.selectbox("Clone", options, format_func=show_clone)
st.caption(
    "Label format: **task name - what varies in this clone [clone type] (internal dataset id)**. "
    "The internal id (e.g. `8m72_2_1`) is only kept for cross-referencing the raw CSV files - "
    "the readable part is what actually matters."
)

row = outcomes_df[outcomes_df["clone_code_id"] == selected].iloc[0]
original_order = row["original_order"].split("|")
reranked_order = row["reranked_order"].split("|")

st.subheader("Query code")
st.code(clone_lookup.get(selected, "<not found>"), language="python")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Embedding search order (top-5)")
    for i, bid in enumerate(original_order, 1):
        marker = " ⬅ true match" if bid == row["desired_base_code_id"] else ""
        st.text(f"{i}. {show_original(bid)}{marker}")
with c2:
    st.subheader("After LLM rerank")
    for i, bid in enumerate(reranked_order, 1):
        marker = " ⬅ true match" if bid == row["desired_base_code_id"] else ""
        st.text(f"{i}. {show_original(bid)}{marker}")

with st.expander("Show top embedding candidate's code"):
    st.code(code_lookup.get(original_order[0], "<not found>"), language="python")

# --- Live call ---------------------------------------------------------------
st.header("Ask Claude now")

st.markdown(f"**Query:** {show_clone(selected)}")
st.markdown("**Embedding search's current top pick (before any LLM judgment):**")
top_before = show_original(original_order[0])
marker_before = " ⬅ true match" if original_order[0] == row["desired_base_code_id"] else ""
st.markdown(f"> #1: **{top_before}**{marker_before}")

if st.button("Judge this clone with Claude now", type="primary", use_container_width=True):
    group = scores_df[scores_df["clone_code_id"] == selected]
    candidates = candidates_for_clone(group, code_lookup)
    start_time = time.time()
    with st.spinner("Calling Claude..."):
        client = ClaudeJudgeClient()
        outcome = rerank_one(
            client, selected, row["desired_base_code_id"], clone_lookup[selected], candidates
        )
    elapsed = time.time() - start_time

    if outcome.judge_error:
        st.error(f"Judge error: {outcome.judge_error}")
    else:
        st.success(f"Claude responded in {elapsed:.1f}s.")

        top_after = show_original(outcome.reranked_order[0])
        marker_after = " ⬅ true match" if outcome.reranked_order[0] == row["desired_base_code_id"] else ""
        changed = outcome.reranked_order[0] != original_order[0]

        arrow_color = "#16a34a" if changed else "#6b7280"
        arrow_label = "changed the #1 pick" if changed else "kept the same #1 pick"
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:14px; margin: 10px 0;
                        padding: 14px; border-radius: 10px; background:#f8fafc; border:1px solid #e2e8f0;">
                <div style="flex:1; text-align:center;">
                    <div style="font-size:12px; color:#64748b;">EMBEDDING SAID</div>
                    <div style="font-weight:700;">{top_before}{marker_before}</div>
                </div>
                <div style="font-size:24px; color:{arrow_color};">&#8594;</div>
                <div style="flex:1; text-align:center;">
                    <div style="font-size:12px; color:#64748b;">CLAUDE SAYS</div>
                    <div style="font-weight:700; color:{arrow_color};">{top_after}{marker_after}</div>
                </div>
            </div>
            <div style="text-align:center; font-size:13px; color:{arrow_color}; margin-bottom:10px;">
                Claude {arrow_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Claude's reasoning per candidate, in its new order:**")
        for bid in outcome.reranked_order:
            reasoning = outcome.reasonings.get(bid, "(kept from original order - no judgment returned)")
            is_true = bid == row["desired_base_code_id"]
            st.markdown(f"- **{show_original(bid)}**{' ✅ true match' if is_true else ''} — {reasoning}")
