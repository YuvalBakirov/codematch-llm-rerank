# LLM Rerank

Extension to the CodeMatch benchmark: takes the top-5 candidates an
embedding model already retrieved for a clone (from an existing
`global-clone` scores CSV) and asks an LLM judge to decide, per candidate,
whether it's a genuine clone or a superficially-similar false positive,
then reranks accordingly.

Motivation: embedding similarity search is known to match on surface
features (variable names, boilerplate, token overlap) that don't always
track genuine shared logic. This measures whether adding an LLM-judgment
pass on top of the existing best embedding model's results improves
Hit@1 / Hit@5, using the same metric definitions as `core/metrics.py`.

## Usage

```bash
# Sanity-check wiring without calling any API
python -m llm_rerank.cli --scores-csv <path-to-existing-global-clone-scores.csv> --n-per-type 5 --dry-run

# Real run (requires ANTHROPIC_API_KEY in a .env file at repo root)
python -m llm_rerank.cli --scores-csv <path> --n-per-type 40
```

## Design notes

- `judge_client.py` defines a `JudgeClient` protocol so the reranking logic
  is testable against `MockJudgeClient` and swappable to any provider;
  `ClaudeJudgeClient` is the only real implementation.
- Live Claude calls use forced tool use with a defined schema. The original
  free-text parser remains only for the mock/provider-neutral path and tests.
- Hallucinated candidate ids are dropped. Malformed tool input and provider,
  network, or timeout exceptions are normalized to `JudgeClientError`.
- On `JudgeClientError`, `rerank_one` keeps the original embedding-search
  order instead of guessing - a failed rerank must not look successful.
- Sampling is stratified by `clone_type` (see `sampling.py`) because
  running the full ~8k-clone set through an LLM judge wasn't worth the
  time/cost for this experiment; see the PDF write-up for what this
  tradeoff means for the results' reliability.

## Result and limits

The 160-clone experiment improved Hit@1 from 78.1% to 81.2% while Hit@5
remained 85.6%. The sample was intended to screen for directional value, not
to establish statistical significance. Production-scale latency, concurrency,
cost, prompt-injection resistance, and code-data governance were not measured.
