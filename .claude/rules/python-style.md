---
description: Python style conventions for this repo
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

# Python style

- Type hints on all function signatures; `from __future__ import annotations` at the top.
- Docstrings on public functions: say what it returns and any non-obvious rule.
- Use `pathlib.Path`, not string paths.
- No bare `except:`. Catch the specific exception you expect.
- Library code (`ingest`, `analytics`, `db`, future agents) uses `logging.getLogger(__name__)`. `print()` is only for `scripts/`.
- Comments only for non-obvious logic. Prefer clear names over clever code.
