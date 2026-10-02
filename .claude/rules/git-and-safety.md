---
description: Git hygiene and safety rules
---

# Git and safety

- Never delete files. If something must go, ask the user to do it.
- Warn before any destructive command or one that installs packages or changes config; offer a safer alternative.
- Never commit secrets (`.env`) or the `.duckdb` file.
- After each phase: summarize files changed and suggest the user review `git diff`.
- Commit or push only when the user asks.
