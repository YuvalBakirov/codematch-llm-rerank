"""End-to-end test: fixture CSVs -> sample -> rerank (mocked judge) -> evaluate.

Uses tiny hand-written fixtures instead of the real ~9k-row benchmark data
so the test is fast and its expected outcome is easy to reason about: the
embedding search ranks an unrelated snippet ("orig_loop") above the true
clone ("orig_sum") for `clone_a`, and the LLM judge is scripted to correctly
say the opposite - this is the exact false-positive pattern the rerank
step is meant to fix.
"""

from pathlib import Path

from llm_rerank.data import build_clone_groups, candidates_for_clone, load_clone_lookup, load_code_lookup
from llm_rerank.evaluate import summarize
from llm_rerank.judge_client import MockJudgeClient
from llm_rerank.rerank import rerank_one
from llm_rerank.sampling import sample_clone_ids

FIXTURES = Path(__file__).parent / "fixtures"


def test_full_pipeline_on_fixture_data():
    scores_df = build_clone_groups(str(FIXTURES / "global_scores.csv"))
    code_lookup = load_code_lookup(str(FIXTURES / "original_code.csv"))
    clone_lookup = load_clone_lookup(str(FIXTURES / "test_code.csv"))

    sampled_ids = sample_clone_ids(scores_df, n_per_type=10)
    assert sampled_ids == ["clone_a"]

    group = scores_df[scores_df["clone_code_id"] == "clone_a"]
    desired = group["desired_base_code_id"].iloc[0]
    candidates = candidates_for_clone(group, code_lookup)
    assert [c["base_code_id"] for c in candidates] == ["orig_loop", "orig_sum"]

    judge = MockJudgeClient(
        '{"judgments": ['
        '{"base_code_id": "orig_loop", "is_clone": false, "confidence": 0.9, "reasoning": "unrelated"}, '
        '{"base_code_id": "orig_sum", "is_clone": true, "confidence": 0.95, "reasoning": "same computation"}'
        ']}'
    )

    outcome = rerank_one(judge, "clone_a", desired, clone_lookup["clone_a"], candidates)

    # Embedding search alone got Hit@1 wrong (orig_loop ranked first).
    assert outcome.original_hit_at_1 is False
    assert outcome.original_hit_at_5 is True

    # LLM rerank fixes it.
    assert outcome.reranked_hit_at_1 is True

    before = summarize([outcome], reranked=False)
    after = summarize([outcome], reranked=True)
    assert before.hit_at_1 == 0.0
    assert after.hit_at_1 == 1.0
