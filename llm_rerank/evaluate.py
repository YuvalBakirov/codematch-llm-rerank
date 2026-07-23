"""Aggregate Hit@1 / Hit@5 before vs. after LLM reranking.

Mirrors the Hit@1 / Hit@5 definitions already used in
`core/metrics.py::calculate_global_metrics` (first-position hit / anywhere
in the returned set), applied to a list of RerankOutcome instead of a raw
scores dataframe, so the numbers here are directly comparable to the
existing per-model benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_rerank.rerank import RerankOutcome


@dataclass
class HitRates:
    n: int
    hit_at_1: float
    hit_at_5: float


def summarize(outcomes: list[RerankOutcome], reranked: bool) -> HitRates:
    n = len(outcomes)
    if n == 0:
        return HitRates(n=0, hit_at_1=0.0, hit_at_5=0.0)

    if reranked:
        hits1 = sum(o.reranked_hit_at_1 for o in outcomes)
        hits5 = sum(o.reranked_hit_at_5 for o in outcomes)
    else:
        hits1 = sum(o.original_hit_at_1 for o in outcomes)
        hits5 = sum(o.original_hit_at_5 for o in outcomes)

    return HitRates(n=n, hit_at_1=hits1 / n, hit_at_5=hits5 / n)


def summarize_by_clone_type(
    outcomes: list[RerankOutcome], clone_types: dict[str, str], reranked: bool
) -> dict[str, HitRates]:
    by_type: dict[str, list[RerankOutcome]] = {}
    for o in outcomes:
        ct = clone_types.get(o.clone_code_id, "Unknown")
        by_type.setdefault(ct, []).append(o)

    return {ct: summarize(group, reranked=reranked) for ct, group in by_type.items()}


def comparison_report(outcomes: list[RerankOutcome], clone_types: dict[str, str]) -> str:
    before = summarize(outcomes, reranked=False)
    after = summarize(outcomes, reranked=True)
    errors = sum(1 for o in outcomes if o.judge_error is not None)

    lines = [
        "# LLM Rerank vs. Embedding-Only: Hit Rate Comparison",
        "",
        f"Sample size: {before.n} clones ({errors} judge errors, kept at original order)",
        "",
        "| Metric | Embedding-only (before) | + LLM rerank (after) |",
        "|---|---|---|",
        f"| Hit@1 | {before.hit_at_1:.1%} | {after.hit_at_1:.1%} |",
        f"| Hit@5 | {before.hit_at_5:.1%} | {after.hit_at_5:.1%} |",
        "",
        "## By clone type",
        "",
        "| Clone type | n | Hit@1 before | Hit@1 after | Hit@5 before | Hit@5 after |",
        "|---|---|---|---|---|---|",
    ]

    before_by_type = summarize_by_clone_type(outcomes, clone_types, reranked=False)
    after_by_type = summarize_by_clone_type(outcomes, clone_types, reranked=True)
    for ct in sorted(before_by_type):
        b = before_by_type[ct]
        a = after_by_type[ct]
        lines.append(f"| {ct} | {b.n} | {b.hit_at_1:.1%} | {a.hit_at_1:.1%} | {b.hit_at_5:.1%} | {a.hit_at_5:.1%} |")

    return "\n".join(lines)
