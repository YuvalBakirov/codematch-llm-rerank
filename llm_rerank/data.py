"""Load existing benchmark artifacts and join them into rerank-ready inputs."""

from __future__ import annotations

import pandas as pd


def load_code_lookup(original_code_csv: str) -> dict[str, str]:
    """base_code_id -> code text, for building candidate snippets."""
    df = pd.read_csv(original_code_csv)
    return dict(zip(df["base_code_id"], df["code"]))


def load_clone_lookup(test_code_csv: str) -> dict[str, str]:
    """clone_code_id -> code text, for the query snippet."""
    df = pd.read_csv(test_code_csv)
    return dict(zip(df["clone_code_id"], df["code"]))


def build_clone_groups(global_scores_csv: str) -> pd.DataFrame:
    """Load an existing global-clone-search scores CSV.

    Returns the raw dataframe; group by clone_code_id to get each clone's
    top-5 candidates in embedding-search rank order (the CSV rows are
    already written in that order by the original benchmark).
    """
    df = pd.read_csv(global_scores_csv)
    return df


def candidates_for_clone(group: pd.DataFrame, code_lookup: dict[str, str]) -> list[dict]:
    candidates = []
    for _, row in group.iterrows():
        base_code_id = row["base_code_id"]
        code = code_lookup.get(base_code_id)
        if code is None:
            continue
        candidates.append({"base_code_id": base_code_id, "code": code})
    return candidates
