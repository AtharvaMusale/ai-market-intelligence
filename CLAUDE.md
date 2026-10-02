@AGENTS.md

## About the user
- On macOS with zsh. Give copy-pasteable commands and say which folder to run them in.
- Learning LangGraph and Pinecone: explain what each new file is for and what changed, in simple terms.
- Add code comments only where logic is non-obvious.
- Keep changes minimal and focused.
- Never delete files. Warn before commands that install packages, change config, or are destructive, and offer a safer alternative.

## Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Numeric store | DuckDB |
| Text retrieval | Pinecone free tier (one serverless index) |
| Orchestration | LangGraph |
| LLM | claude-haiku-4-5-20251001 only |
| UI | Streamlit (Phase 5) |
| Tests | pytest |

## Build and test
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Phase workflow
After each phase: run pytest, summarize files changed, suggest the user review `git diff`, then STOP and wait for their go-ahead. Phases: 0 scaffolding, 1 ingestion + regime analytics, 2 text ingestion + retrieval, 3 LangGraph agents, 4 evaluation harness, 5 Streamlit delivery.

## Rules files
Detailed rules live in `.claude/rules/`: `architecture.md`, `python-style.md`, `data-and-sql.md`, `llm-and-cost.md`, `testing-and-eval.md`, `git-and-safety.md`.
