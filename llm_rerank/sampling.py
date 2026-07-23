"""Stratified sampling of clones for the (costly, rate-limited) LLM rerank experiment.

Running the full ~8k-clone benchmark through an LLM judge is not worth the
time or API cost for this experiment; we sample a stratified subset across
clone_type so the comparison still reflects all four clone categories
(identical / renamed / near-miss / semantic-or-cross-language), not just
whichever type happens to dominate the dataset.
"""

from __future__ import annotations

import pandas as pd


def sample_clone_ids(
    global_scores_df: pd.DataFrame,
    n_per_type: int,
    random_state: int = 42,
) -> list[str]:
    """Pick up to n_per_type distinct clone_code_ids per clone_type."""
    per_clone = global_scores_df.drop_duplicates(subset=["clone_code_id"])[
        ["clone_code_id", "clone_type"]
    ]

    sampled_ids: list[str] = []
    for clone_type, group in per_clone.groupby("clone_type"):
        n = min(n_per_type, len(group))
        sampled = group.sample(n=n, random_state=random_state)
        sampled_ids.extend(sampled["clone_code_id"].tolist())

    return sampled_ids
