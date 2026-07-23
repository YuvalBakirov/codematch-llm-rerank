from llm_rerank.judge_client import MockJudgeClient
from llm_rerank.rerank import rerank_one

CANDIDATES = [
    {"base_code_id": "wrong1", "code": "def x(): pass"},
    {"base_code_id": "wrong2", "code": "def y(): pass"},
    {"base_code_id": "correct", "code": "def z(): return 42"},
]


def test_rerank_promotes_true_clone_above_embedding_top1():
    # Embedding search put "wrong1" first by cosine similarity, but the
    # judge correctly identifies "correct" as the only genuine clone -
    # this is the exact failure mode (superficial similarity beating
    # true logical similarity) the rerank step exists to catch.
    scripted = (
        '{"judgments": ['
        '{"base_code_id": "wrong1", "is_clone": false, "confidence": 0.9, "reasoning": "different logic"}, '
        '{"base_code_id": "wrong2", "is_clone": false, "confidence": 0.8, "reasoning": "different logic"}, '
        '{"base_code_id": "correct", "is_clone": true, "confidence": 0.95, "reasoning": "same algorithm"}'
        ']}'
    )
    client = MockJudgeClient(scripted)

    outcome = rerank_one(client, "clone1", "correct", "def q(): return 42", CANDIDATES)

    assert outcome.original_order == ["wrong1", "wrong2", "correct"]
    assert outcome.original_hit_at_1 is False
    assert outcome.reranked_order[0] == "correct"
    assert outcome.reranked_hit_at_1 is True


def test_rerank_keeps_original_order_on_judge_error():
    client = MockJudgeClient("not valid json at all")

    outcome = rerank_one(client, "clone1", "correct", "def q(): return 42", CANDIDATES)

    assert outcome.judge_error is not None
    assert outcome.reranked_order == outcome.original_order


def test_rerank_appends_unjudged_candidates_after_judged_ones():
    # Judge only returned a verdict for one of the three candidates.
    scripted = '{"judgments": [{"base_code_id": "correct", "is_clone": true, "confidence": 0.9, "reasoning": "x"}]}'
    client = MockJudgeClient(scripted)

    outcome = rerank_one(client, "clone1", "correct", "def q(): return 42", CANDIDATES)

    assert outcome.reranked_order[0] == "correct"
    assert set(outcome.reranked_order) == {"wrong1", "wrong2", "correct"}
    assert len(outcome.reranked_order) == 3
