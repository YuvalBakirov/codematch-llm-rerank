# Demo recording script (CLI, ~2-3 min)

Record with Win+Alt+R (Xbox Game Bar) or any screen recorder. Terminal in
the `codematch-benchmark/` folder with the `.venv` activated.

## 1. Show the test suite (proves the logic is verified, not just "trust me")

```bash
./.venv/Scripts/python.exe -m pytest tests/test_llm_rerank/ -v
```

Say: "19 tests, no API calls needed - the parsing, reranking, and metrics
logic are all verified against a mocked judge before touching real money
or real API calls."

## 2. Show a --dry-run (proves the wiring works without spending anything)

```bash
./.venv/Scripts/python.exe -m llm_rerank.cli \
  --scores-csv "output/Qwen2.5-Coder-0.5B-pe/global-clone/Qwen2.5-Coder-0.5B-pe_global_clone_search_scores_20.11.2024_12-02-25.csv" \
  --n-per-type 3 --dry-run --out-dir output/llm_rerank_demo_dryrun
```

## 3. Show the real result (the actual number that matters)

Open `output/llm_rerank_full_v2/rerank_report.md` (or print it):

```bash
cat output/llm_rerank_full_v2/rerank_report.md
```

Say: "160 clones sampled, stratified across all four clone types. Overall
Hit@1 went from 78.1% to 81.2% with LLM reranking on top of the existing
best embedding model (Qwen2.5-Coder-0.5B-pe). The gain is concentrated in
T4 - the hardest category, semantic/cross-language clones - which went
from 52.5% to 57.5%. That's exactly where embedding similarity alone is
weakest and an LLM judge has the most to add. Hit@5 is unchanged by
design - reranking can only reorder the top-5 the embedding model already
retrieved, it can't recover a true match that embedding search missed
entirely."

## 4. Show the concrete example (this is the part that actually convinces people)

```bash
./.venv/Scripts/python.exe scripts/pick_demo_example.py output/llm_rerank_full_v2/rerank_outcomes.csv
```

Say: "Concretely: for this query about Chernick numbers, the embedding
search's top pick was a snippet that's also about primality testing but a
different algorithm - superficially similar tokens, wrong logic. The true
match was sitting at rank 2. The LLM judge caught that and promoted it to
rank 1."

## 5. (Optional) Show the bug-fix commit

```bash
git log --oneline
git show c0aa40d --stat
```

Say: "First live run had a 5% JSON-parse failure rate (8/160 calls) -
free-text JSON broke when the model's own reasoning text contained a
quote or apostrophe. Fixed by moving to Claude's tool-use / forced
structured output instead of prompt-engineered JSON-in-prose. Re-ran
after the fix: 0/160 errors."
