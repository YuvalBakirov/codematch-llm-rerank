from pathlib import Path

from llm_rerank.data import build_clone_display_names, build_original_display_names

FIXTURES = Path(__file__).parent / "fixtures"


def test_original_display_name_uses_task_not_raw_id():
    names = build_original_display_names(str(FIXTURES / "original_code.csv"))
    assert names["orig_sum"] == "sum list (orig_sum)"
    assert names["orig_loop"] == "print loop (orig_loop)"


def test_clone_display_name_uses_task_and_sub_type_not_raw_numbers():
    names = build_clone_display_names(str(FIXTURES / "test_code.csv"))
    # "clone_a" encodes base task "orig_sum", clone_type "T1", but that's only
    # readable via clone_sub_type - the label must spell it out in English.
    assert names["clone_a"] == "sum list - Identical Clone [T1] (clone_a)"
