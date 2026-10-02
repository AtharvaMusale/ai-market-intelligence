---
description: Testing approach and evaluation harness requirements
paths:
  - "tests/**"
  - "src/market_intel/eval/**"
---

# Testing and evaluation

## Tests
- Fully offline: synthetic data, in-memory DuckDB (`connect(":memory:")`), mocked LLM and Pinecone. No network, no API keys.
- Test behavior at thresholds (e.g. VIX 14.99 / 15.0) and edge cases (missing data, empty frames).

## Evaluation harness (Phase 4)
- Parse numbers and claims in a generated brief and verify them against DuckDB/analytics output within a stated tolerance.
- Report numeric match rate, unsupported-claim rate, and citation coverage.
- Keep a small fixed question set with expected tool usage.
- Fail the run if numeric accuracy falls below a configurable threshold.
- Never tune thresholds just to make a run pass.
