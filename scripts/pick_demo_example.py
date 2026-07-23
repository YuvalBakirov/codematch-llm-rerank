"""Pick one clone where the LLM rerank fixed a Hit@1 miss, for the demo recording.

Prints the query code, the embedding search's (wrong) top pick, the true
match, and where the rerank put it - concrete evidence, not just an
aggregate percentage.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main(outcomes_csv: str):
    df = pd.read_csv(outcomes_csv)
    fixed = df[(df["original_hit_at_1"] == False) & (df["reranked_hit_at_1"] == True)]  # noqa: E712

    if fixed.empty:
        print("No 'rerank fixed it' examples in this run.")
        return

    row = fixed.iloc[0]
    clone_lookup = dict(
        pd.read_csv(ROOT / "data/evaluation-datasets/test_code_benchmark_fixed.csv")[
            ["clone_code_id", "code"]
        ].values
    )
    code_lookup = dict(
        pd.read_csv(ROOT / "data/evaluation-datasets/original_code_benchmark_fixed.csv")[
            ["base_code_id", "code"]
        ].values
    )

    original_order = row["original_order"].split("|")
    reranked_order = row["reranked_order"].split("|")

    print(f"Clone: {row['clone_code_id']}  (true match: {row['desired_base_code_id']})")
    print()
    print("Query code:")
    print(clone_lookup.get(row["clone_code_id"], "<not found>")[:400])
    print()
    print(f"Embedding search top-1 (WRONG): {original_order[0]}")
    print(code_lookup.get(original_order[0], "<not found>")[:200])
    print()
    print(f"Full embedding order: {original_order}")
    print(f"Full rerank order:    {reranked_order}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/llm_rerank_full/rerank_outcomes.csv")
