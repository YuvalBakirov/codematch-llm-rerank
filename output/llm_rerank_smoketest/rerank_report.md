# LLM Rerank vs. Embedding-Only: Hit Rate Comparison

Sample size: 12 clones (0 judge errors, kept at original order)

| Metric | Embedding-only (before) | + LLM rerank (after) |
|---|---|---|
| Hit@1 | 66.7% | 66.7% |
| Hit@5 | 91.7% | 91.7% |

## By clone type

| Clone type | n | Hit@1 before | Hit@1 after | Hit@5 before | Hit@5 after |
|---|---|---|---|---|---|
| T1 | 3 | 100.0% | 100.0% | 100.0% | 100.0% |
| T2 | 3 | 66.7% | 66.7% | 100.0% | 100.0% |
| T3 | 3 | 66.7% | 66.7% | 100.0% | 100.0% |
| T4 | 3 | 33.3% | 33.3% | 66.7% | 66.7% |