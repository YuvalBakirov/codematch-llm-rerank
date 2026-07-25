"""LLM clients that judge whether retrieved candidates are true code clones.

The reranker depends only on the `JudgeClient` protocol, not on any specific
provider, so tests can run against `MockJudgeClient` and real experiments
can swap in `ClaudeJudgeClient` without touching the reranking logic.
"""

from __future__ import annotations

import abc
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class CandidateJudgment:
    base_code_id: str
    is_clone: bool
    confidence: float
    reasoning: str


@dataclass
class JudgeResult:
    judgments: list[CandidateJudgment] = field(default_factory=list)
    raw_response: str = ""


class JudgeClientError(Exception):
    """Raised when a judge client cannot produce a usable result."""


class JudgeClient(abc.ABC):
    @abc.abstractmethod
    def judge_candidates(self, query_code: str, candidates: list[dict]) -> JudgeResult:
        """Judge each candidate against the query code.

        `candidates` is a list of dicts with at least `base_code_id` and `code`.
        Must return a JudgeResult whose judgments only reference base_code_ids
        that were present in `candidates` (unknown ids are dropped upstream).
        """


def build_judge_prompt(query_code: str, candidates: list[dict]) -> str:
    candidate_blocks = []
    for c in candidates:
        candidate_blocks.append(
            f'--- candidate id="{c["base_code_id"]}" ---\n{c["code"]}'
        )
    candidates_text = "\n\n".join(candidate_blocks)

    return f"""You are judging code-clone search results.

A query code snippet was searched against a database using embedding
similarity, which is known to sometimes match on superficial similarity
(similar tokens, similar boilerplate) rather than genuine shared logic.

Query code:
```
{query_code}
```

Candidates returned by the embedding search (in this order):

{candidates_text}

For EACH candidate, decide if it is a genuine code clone of the query
(same underlying algorithm/logic, even if renamed, reformatted, or in a
different language) or a false positive (superficially similar but
different logic).

Use the provided record_judgments tool exactly once. Include one judgment
per candidate and use only the candidate ids given above."""


def _judgments_from_raw_list(raw_judgments: list, known_candidate_ids: set[str]) -> list[CandidateJudgment]:
    """Shared filtering/coercion used by both the free-text and tool-use paths.

    Silently drops any base_code_id not in known_candidate_ids: a judgment
    for a candidate that was never shown to the model (hallucinated or
    copy-pasted from a different call) cannot be reranked into a list that
    never contained it.
    """
    judgments = []
    for item in raw_judgments:
        base_code_id = item.get("base_code_id")
        if base_code_id not in known_candidate_ids:
            continue
        judgments.append(
            CandidateJudgment(
                base_code_id=base_code_id,
                is_clone=bool(item.get("is_clone", False)),
                confidence=float(item.get("confidence", 0.0)),
                reasoning=str(item.get("reasoning", "")),
            )
        )
    return judgments


def parse_judge_response(raw_response: str, known_candidate_ids: set[str]) -> JudgeResult:
    """Parse a judge LLM's free-text response into a JudgeResult.

    Used for the MockJudgeClient test double and as a fallback path. The
    real ClaudeJudgeClient no longer relies on this for live calls (see
    its docstring) - free-text JSON-in-prose responses turned out to
    intermittently break on unescaped content inside the "reasoning"
    string (about 5% of calls in a 160-call live run), so live judging
    was moved to tool-use / forced structured output instead. This parser
    is kept because it is still exercised by tests and remains a sane
    fallback shape for any future non-tool-use provider.
    """
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        raise JudgeClientError(f"No JSON object found in judge response: {raw_response!r}")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise JudgeClientError(f"Judge response was not valid JSON: {e}") from e

    raw_judgments = payload.get("judgments")
    if not isinstance(raw_judgments, list):
        raise JudgeClientError("Judge response missing a 'judgments' list")

    judgments = _judgments_from_raw_list(raw_judgments, known_candidate_ids)
    return JudgeResult(judgments=judgments, raw_response=raw_response)


JUDGE_TOOL = {
    "name": "record_judgments",
    "description": "Record a clone-vs-false-positive judgment for each candidate shown.",
    "input_schema": {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "base_code_id": {"type": "string"},
                        "is_clone": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["base_code_id", "is_clone", "confidence", "reasoning"],
                },
            }
        },
        "required": ["judgments"],
    },
}


class ClaudeJudgeClient(JudgeClient):
    """Judges candidates via Claude tool-use (forced structured output).

    v1 of this client asked Claude to write JSON as free text and parsed it
    with `parse_judge_response`. In a 160-call live run, 8 calls (5%) failed
    with JSONDecodeError because malformed escaping around quotation
    characters broke the surrounding JSON string.
    Forcing a tool call with a JSON schema moves that escaping burden onto
    Anthropic's structured-output handling instead of a hand-rolled prompt
    instruction, which removed the failure mode entirely on re-run.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def judge_candidates(self, query_code: str, candidates: list[dict]) -> JudgeResult:
        prompt = build_judge_prompt(query_code, candidates)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "record_judgments"},
                messages=[{"role": "user", "content": prompt}],
            )

            tool_use_block = next(
                (b for b in response.content if getattr(b, "type", None) == "tool_use"),
                None,
            )
            if tool_use_block is None:
                raise JudgeClientError(
                    f"No tool_use block in response: {response.content!r}"
                )
            if not isinstance(tool_use_block.input, dict):
                raise JudgeClientError(
                    f"Tool input was not an object: {tool_use_block.input!r}"
                )

            raw_judgments = tool_use_block.input.get("judgments")
            if not isinstance(raw_judgments, list):
                raise JudgeClientError("Tool input missing a 'judgments' list")

            known_ids = {c["base_code_id"] for c in candidates}
            judgments = _judgments_from_raw_list(raw_judgments, known_ids)
            return JudgeResult(
                judgments=judgments,
                raw_response=json.dumps(tool_use_block.input),
            )
        except JudgeClientError:
            raise
        except Exception as e:
            raise JudgeClientError(
                f"Claude judge request failed ({type(e).__name__}): {e}"
            ) from e


class MockJudgeClient(JudgeClient):
    """Test double: returns a scripted raw response regardless of input."""

    def __init__(self, scripted_response: str):
        self.scripted_response = scripted_response
        self.calls: list[tuple[str, list[dict]]] = []

    def judge_candidates(self, query_code: str, candidates: list[dict]) -> JudgeResult:
        self.calls.append((query_code, candidates))
        known_ids = {c["base_code_id"] for c in candidates}
        return parse_judge_response(self.scripted_response, known_ids)
