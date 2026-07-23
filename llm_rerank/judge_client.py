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

Respond with ONLY a JSON object, no other text, in this exact shape:
{{
  "judgments": [
    {{"base_code_id": "<id>", "is_clone": true|false, "confidence": 0.0-1.0, "reasoning": "<one sentence>"}}
  ]
}}

Include exactly one judgment object per candidate, using the candidate ids
given above."""


def parse_judge_response(raw_response: str, known_candidate_ids: set[str]) -> JudgeResult:
    """Parse a judge LLM's raw text response into a JudgeResult.

    Tolerates: prose wrapped around the JSON block, missing/extra fields,
    and hallucinated candidate ids (silently dropped, since they cannot be
    reranked into a candidate list that never contained them).
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

    return JudgeResult(judgments=judgments, raw_response=raw_response)


class ClaudeJudgeClient(JudgeClient):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        import anthropic

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def judge_candidates(self, query_code: str, candidates: list[dict]) -> JudgeResult:
        prompt = build_judge_prompt(query_code, candidates)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        known_ids = {c["base_code_id"] for c in candidates}
        return parse_judge_response(raw_text, known_ids)


class MockJudgeClient(JudgeClient):
    """Test double: returns a scripted raw response regardless of input."""

    def __init__(self, scripted_response: str):
        self.scripted_response = scripted_response
        self.calls: list[tuple[str, list[dict]]] = []

    def judge_candidates(self, query_code: str, candidates: list[dict]) -> JudgeResult:
        self.calls.append((query_code, candidates))
        known_ids = {c["base_code_id"] for c in candidates}
        return parse_judge_response(self.scripted_response, known_ids)
