import pytest

from llm_rerank.judge_client import JudgeClientError, build_judge_prompt, parse_judge_response


KNOWN_IDS = {"aaa1", "bbb2", "ccc3"}


def test_parse_clean_json():
    raw = '{"judgments": [{"base_code_id": "aaa1", "is_clone": true, "confidence": 0.9, "reasoning": "same algorithm"}]}'
    result = parse_judge_response(raw, KNOWN_IDS)
    assert len(result.judgments) == 1
    assert result.judgments[0].base_code_id == "aaa1"
    assert result.judgments[0].is_clone is True
    assert result.judgments[0].confidence == 0.9


def test_parse_json_wrapped_in_prose():
    # Real LLMs (including Claude, occasionally) prepend commentary before the JSON
    # even when told to return only JSON. The parser must recover from that.
    raw = (
        "Sure, here is my analysis of the candidates:\n\n"
        '{"judgments": [{"base_code_id": "aaa1", "is_clone": false, "confidence": 0.2, "reasoning": "different logic"}]}'
        "\n\nLet me know if you need anything else!"
    )
    result = parse_judge_response(raw, KNOWN_IDS)
    assert len(result.judgments) == 1
    assert result.judgments[0].is_clone is False


def test_parse_drops_hallucinated_candidate_id():
    # The LLM must only judge candidates it was actually shown; an id that
    # was never in the candidate set (hallucinated or copy-paste error from
    # the model) is dropped rather than silently trusted.
    raw = '{"judgments": [{"base_code_id": "zzz9", "is_clone": true, "confidence": 0.9, "reasoning": "x"}]}'
    result = parse_judge_response(raw, KNOWN_IDS)
    assert result.judgments == []


def test_parse_missing_optional_fields_defaults_safely():
    raw = '{"judgments": [{"base_code_id": "bbb2"}]}'
    result = parse_judge_response(raw, KNOWN_IDS)
    assert len(result.judgments) == 1
    assert result.judgments[0].is_clone is False
    assert result.judgments[0].confidence == 0.0
    assert result.judgments[0].reasoning == ""


def test_parse_raises_on_non_json():
    with pytest.raises(JudgeClientError):
        parse_judge_response("I cannot help with that request.", KNOWN_IDS)


def test_parse_raises_on_missing_judgments_key():
    with pytest.raises(JudgeClientError):
        parse_judge_response('{"result": "ok"}', KNOWN_IDS)


def test_parse_raises_on_malformed_json_braces():
    # Truncated/mid-stream response (e.g. hit max_tokens) - unbalanced braces.
    raw = '{"judgments": [{"base_code_id": "aaa1", "is_clone": true'
    with pytest.raises(JudgeClientError):
        parse_judge_response(raw, KNOWN_IDS)


def test_build_judge_prompt_includes_all_candidates_and_query():
    prompt = build_judge_prompt(
        "def f(): pass",
        [
            {"base_code_id": "aaa1", "code": "def a(): return 1"},
            {"base_code_id": "bbb2", "code": "def b(): return 2"},
        ],
    )
    assert "def f(): pass" in prompt
    assert "aaa1" in prompt and "def a(): return 1" in prompt
    assert "bbb2" in prompt and "def b(): return 2" in prompt
