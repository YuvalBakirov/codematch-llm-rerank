"""Tests for ClaudeJudgeClient's tool-use response handling, using a stub
Anthropic client (no network / no API key needed).

Regression context: v1 asked Claude to write JSON as free text and parsed
it with regex + json.loads. In a 160-clone live run, 8 calls (5%) failed
because malformed escaping around quotation characters in a "reasoning"
sentence broke the surrounding JSON string. Moved to forced tool use to
push escaping onto Anthropic's side; these tests pin the new behavior.
"""

from types import SimpleNamespace

from llm_rerank.judge_client import ClaudeJudgeClient, JudgeClientError

CANDIDATES = [
    {"base_code_id": "aaa1", "code": "def a(): pass"},
    {"base_code_id": "bbb2", "code": "def b(): pass"},
]


def _client_with_stub_response(content_blocks):
    client = ClaudeJudgeClient.__new__(ClaudeJudgeClient)  # bypass __init__ (no real API client)
    client._model = "stub-model"

    class _StubMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=content_blocks)

    client._client = SimpleNamespace(messages=_StubMessages())
    return client


def test_extracts_judgments_from_tool_use_block():
    tool_use_block = SimpleNamespace(
        type="tool_use",
        input={
            "judgments": [
                {"base_code_id": "aaa1", "is_clone": True, "confidence": 0.9, "reasoning": "it's the same logic"},
                {"base_code_id": "bbb2", "is_clone": False, "confidence": 0.1, "reasoning": "unrelated"},
            ]
        },
    )
    client = _client_with_stub_response([tool_use_block])

    result = client.judge_candidates("def q(): pass", CANDIDATES)

    assert len(result.judgments) == 2
    assert result.judgments[0].base_code_id == "aaa1"
    assert result.judgments[0].is_clone is True
    # Tool input is already structured, so punctuation in reasoning does not
    # create a hand-written JSON escaping burden in the application.
    assert "it's the same logic" == result.judgments[0].reasoning


def test_drops_hallucinated_id_from_tool_use_block():
    tool_use_block = SimpleNamespace(
        type="tool_use",
        input={"judgments": [{"base_code_id": "zzz9", "is_clone": True, "confidence": 0.9, "reasoning": "x"}]},
    )
    client = _client_with_stub_response([tool_use_block])

    result = client.judge_candidates("def q(): pass", CANDIDATES)

    assert result.judgments == []


def test_raises_if_no_tool_use_block_returned():
    text_block = SimpleNamespace(type="text", text="I refuse to use the tool.")
    client = _client_with_stub_response([text_block])

    try:
        client.judge_candidates("def q(): pass", CANDIDATES)
        assert False, "expected JudgeClientError"
    except JudgeClientError:
        pass


def test_wraps_provider_or_network_exception_as_judge_error():
    client = ClaudeJudgeClient.__new__(ClaudeJudgeClient)
    client._model = "stub-model"

    class _FailingMessages:
        def create(self, **kwargs):
            raise TimeoutError("provider timed out")

    client._client = SimpleNamespace(messages=_FailingMessages())

    try:
        client.judge_candidates("def q(): pass", CANDIDATES)
        assert False, "expected JudgeClientError"
    except JudgeClientError as error:
        assert "TimeoutError" in str(error)
        assert "provider timed out" in str(error)
