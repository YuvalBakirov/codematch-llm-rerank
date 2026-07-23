import pandas as pd
import pytest

from llm_rerank.evaluate import clone_type_lookup, summarize
from llm_rerank.rerank import RerankOutcome


def _outcome(original_order, reranked_order, desired):
    return RerankOutcome(
        clone_code_id="c",
        original_order=original_order,
        reranked_order=reranked_order,
        desired_base_code_id=desired,
    )


def test_summarize_hit_rates_before_and_after():
    outcomes = [
        _outcome(["b", "a"], ["a", "b"], desired="a"),  # rerank fixes hit@1
        _outcome(["x", "y"], ["x", "y"], desired="y"),  # hit@5 only, unchanged
        _outcome(["p", "q"], ["p", "q"], desired="z"),  # miss entirely
    ]

    before = summarize(outcomes, reranked=False)
    after = summarize(outcomes, reranked=True)

    assert before.n == 3
    assert before.hit_at_1 == 0.0  # neither "b,a" nor "x,y" nor "p,q" hit at position 1 for their desired id
    assert before.hit_at_5 == pytest.approx(2 / 3)

    assert after.hit_at_1 == pytest.approx(1 / 3)
    assert after.hit_at_5 == pytest.approx(2 / 3)


def test_summarize_empty_outcomes():
    result = summarize([], reranked=True)
    assert result.n == 0
    assert result.hit_at_1 == 0.0
    assert result.hit_at_5 == 0.0


def test_clone_type_lookup_handles_five_rows_per_clone():
    # A real global-clone-search scores df has 5 rows per clone_code_id
    # (one per candidate). Building the lookup without de-duplicating first
    # returns a pandas Series instead of a scalar for every id, which blew
    # up with `TypeError: unhashable type: 'Series'` the moment it was used
    # as a groupby key in the demo app.
    scores_df = pd.DataFrame(
        {
            "clone_code_id": ["c1", "c1", "c1", "c1", "c1", "c2", "c2"],
            "clone_type": ["T1", "T1", "T1", "T1", "T1", "T4", "T4"],
        }
    )

    lookup = clone_type_lookup(scores_df)

    assert lookup == {"c1": "T1", "c2": "T4"}
    # The actual regression: this must not raise, and must return a plain str.
    assert isinstance(lookup["c1"], str)
