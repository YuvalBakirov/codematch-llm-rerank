"""Rerank existing embedding-search candidates using an LLM judge."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_rerank.judge_client import JudgeClient, JudgeClientError


@dataclass
class RerankOutcome:
    clone_code_id: str
    original_order: list[str]
    reranked_order: list[str]
    desired_base_code_id: str
    judge_error: str | None = None
    reasonings: dict[str, str] = field(default_factory=dict)

    @property
    def original_hit_at_1(self) -> bool:
        return bool(self.original_order) and self.original_order[0] == self.desired_base_code_id

    @property
    def original_hit_at_5(self) -> bool:
        return self.desired_base_code_id in self.original_order

    @property
    def reranked_hit_at_1(self) -> bool:
        return bool(self.reranked_order) and self.reranked_order[0] == self.desired_base_code_id

    @property
    def reranked_hit_at_5(self) -> bool:
        return self.desired_base_code_id in self.reranked_order


def rerank_one(
    judge_client: JudgeClient,
    clone_code_id: str,
    desired_base_code_id: str,
    query_code: str,
    candidates: list[dict],
) -> RerankOutcome:
    """Rerank one clone's top-5 candidates using the judge client.

    JudgeClientError is the provider boundary: malformed output, API errors,
    timeouts, and other handled provider failures are normalized to it. On
    that failure the original embedding-search order is kept unchanged and
    the error is recorded rather than producing a made-up ranking.
    """
    original_order = [c["base_code_id"] for c in candidates]

    try:
        result = judge_client.judge_candidates(query_code, candidates)
    except JudgeClientError as e:
        return RerankOutcome(
            clone_code_id=clone_code_id,
            original_order=original_order,
            reranked_order=original_order,
            desired_base_code_id=desired_base_code_id,
            judge_error=str(e),
        )

    judged_ids = {j.base_code_id for j in result.judgments}
    ranked_judgments = sorted(
        result.judgments, key=lambda j: (not j.is_clone, -j.confidence)
    )
    reranked_order = [j.base_code_id for j in ranked_judgments]
    # Candidates the judge didn't return a judgment for keep their relative
    # original order and are appended after the judged ones.
    reranked_order += [cid for cid in original_order if cid not in judged_ids]

    reasonings = {j.base_code_id: j.reasoning for j in result.judgments}

    return RerankOutcome(
        clone_code_id=clone_code_id,
        original_order=original_order,
        reranked_order=reranked_order,
        desired_base_code_id=desired_base_code_id,
        reasonings=reasonings,
    )
