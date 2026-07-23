"""Entry point: run the LLM-rerank experiment against an existing benchmark run.

Usage:
    python -m llm_rerank.cli --scores-csv output/Qwen2.5-Coder-0.5B-pe/global-clone/<file>.csv --n-per-type 40
    python -m llm_rerank.cli --scores-csv ... --n-per-type 40 --dry-run   # no API calls, sanity-checks wiring only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from llm_rerank.data import build_clone_groups, candidates_for_clone, load_clone_lookup, load_code_lookup
from llm_rerank.evaluate import comparison_report
from llm_rerank.judge_client import ClaudeJudgeClient, JudgeClient, JudgeResult, CandidateJudgment
from llm_rerank.rerank import RerankOutcome, rerank_one
from llm_rerank.sampling import sample_clone_ids

ROOT_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_CODE_CSV = ROOT_DIR / "data/evaluation-datasets/original_code_benchmark_fixed.csv"
TEST_CODE_CSV = ROOT_DIR / "data/evaluation-datasets/test_code_benchmark_fixed.csv"


class _DryRunJudgeClient(JudgeClient):
    """Keeps embedding order as-is; used to sanity-check the pipeline without an API key."""

    def judge_candidates(self, query_code, candidates):
        judgments = [
            CandidateJudgment(base_code_id=c["base_code_id"], is_clone=True, confidence=0.5, reasoning="dry-run")
            for c in candidates
        ]
        return JudgeResult(judgments=judgments, raw_response="dry-run")


def run(scores_csv: str, n_per_type: int, dry_run: bool, sleep_between_calls: float, out_dir: str) -> None:
    load_dotenv()

    scores_df = build_clone_groups(scores_csv)
    code_lookup = load_code_lookup(str(ORIGINAL_CODE_CSV))
    clone_code_lookup = load_clone_lookup(str(TEST_CODE_CSV))

    sampled_ids = sample_clone_ids(scores_df, n_per_type=n_per_type)
    clone_types = dict(
        scores_df.drop_duplicates(subset=["clone_code_id"])[["clone_code_id", "clone_type"]].values
    )

    judge_client: JudgeClient = _DryRunJudgeClient() if dry_run else ClaudeJudgeClient()

    outcomes: list[RerankOutcome] = []
    for clone_code_id in tqdm(sampled_ids, desc="Reranking"):
        group = scores_df[scores_df["clone_code_id"] == clone_code_id]
        if group.empty:
            continue

        desired_base_code_id = group["desired_base_code_id"].iloc[0]
        query_code = clone_code_lookup.get(clone_code_id)
        if query_code is None:
            continue

        candidates = candidates_for_clone(group, code_lookup)
        if not candidates:
            continue

        outcome = rerank_one(judge_client, clone_code_id, desired_base_code_id, query_code, candidates)
        outcomes.append(outcome)

        if not dry_run and sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    report = comparison_report(outcomes, clone_types)
    print(report)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "rerank_report.md").write_text(report, encoding="utf-8")

    rows = [
        {
            "clone_code_id": o.clone_code_id,
            "desired_base_code_id": o.desired_base_code_id,
            "original_order": "|".join(o.original_order),
            "reranked_order": "|".join(o.reranked_order),
            "original_hit_at_1": o.original_hit_at_1,
            "reranked_hit_at_1": o.reranked_hit_at_1,
            "original_hit_at_5": o.original_hit_at_5,
            "reranked_hit_at_5": o.reranked_hit_at_5,
            "judge_error": o.judge_error,
        }
        for o in outcomes
    ]
    pd.DataFrame(rows).to_csv(out_path / "rerank_outcomes.csv", index=False)
    print(f"\nSaved report + raw outcomes to {out_path}/")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-csv", required=True, help="Existing global-clone-search scores CSV to rerank")
    parser.add_argument("--n-per-type", type=int, default=40, help="Clones to sample per clone_type")
    parser.add_argument("--dry-run", action="store_true", help="Skip real LLM calls; sanity-check wiring only")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between real API calls")
    parser.add_argument("--out-dir", default="output/llm_rerank", help="Where to write the report + outcomes CSV")
    args = parser.parse_args(argv)

    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (and --dry-run not passed). Set it in a .env file.", file=sys.stderr)
        sys.exit(1)

    run(args.scores_csv, args.n_per_type, args.dry_run, args.sleep, args.out_dir)


if __name__ == "__main__":
    main()
