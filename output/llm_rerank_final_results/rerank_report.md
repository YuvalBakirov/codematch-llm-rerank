# LLM Rerank vs. Embedding-Only: Hit Rate Comparison

Sample size: 160 clones (0 judge errors, kept at original order)

| Metric | Embedding-only (before) | + LLM rerank (after) |
|---|---|---|
| Hit@1 | 78.1% | 81.2% |
| Hit@5 | 85.6% | 85.6% |

## By clone type

| Clone type | n | Hit@1 before | Hit@1 after | Hit@5 before | Hit@5 after |
|---|---|---|---|---|---|
| T1 | 40 | 92.5% | 92.5% | 92.5% | 92.5% |
| T2 | 40 | 82.5% | 87.5% | 87.5% | 87.5% |
| T3 | 40 | 85.0% | 87.5% | 90.0% | 90.0% |
| T4 | 40 | 52.5% | 57.5% | 72.5% | 72.5% |