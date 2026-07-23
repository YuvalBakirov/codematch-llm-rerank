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


def build_original_display_names(original_code_csv: str) -> dict[str, str]:
    """base_code_id -> human-readable label, e.g. "Chernick's Carmichael numbers (8m72)".

    The raw ids (e.g. "8m72") are opaque dataset keys with no inherent
    meaning; the "task" column already contains a real human-readable name
    for what the code does. The raw id is kept in parentheses only so it's
    still traceable back to the underlying data for debugging.
    """
    df = pd.read_csv(original_code_csv)
    deduped = df.drop_duplicates(subset=["base_code_id"])
    return {row["base_code_id"]: f'{row["task"]} ({row["base_code_id"]})' for _, row in deduped.iterrows()}


def build_clone_display_names(test_code_csv: str) -> dict[str, str]:
    """clone_code_id -> human-readable label, e.g.
    "Chernick's Carmichael numbers - Different Variable Names [T2] (8m72_2_1)".

    A clone id like "8m72_2_1" encodes: base task "8m72", clone_type "2"
    (T2 = renamed-but-equivalent), variant "1" within that type - but none
    of that is readable without knowing the dataset's own conventions. The
    "clone_sub_type" column already spells out in English exactly what
    varies for that specific id, so we use it instead of the raw numbers.

    clone_code_id is not always unique in the source data (the same id can
    appear once per clone_language for T4/cross-language clones), so this
    de-duplicates first - same caveat as clone_type_lookup() in evaluate.py.
    """
    df = pd.read_csv(test_code_csv)
    deduped = df.drop_duplicates(subset=["clone_code_id"])
    labels = {}
    for _, row in deduped.iterrows():
        labels[row["clone_code_id"]] = (
            f'{row["task"]} - {row["clone_sub_type"]} [{row["clone_type"]}] ({row["clone_code_id"]})'
        )
    return labels
