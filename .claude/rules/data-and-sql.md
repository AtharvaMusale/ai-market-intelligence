---
description: DuckDB, SQL, and market-data correctness rules
paths:
  - "src/market_intel/db.py"
  - "src/market_intel/ingest/**"
  - "src/market_intel/analytics/**"
---

# Data and SQL

- Parameterized queries only. Table/column names that must be interpolated go through an allow-list.
- Every table has a primary key; writes are `INSERT OR REPLACE` upserts so re-runs are idempotent.
- Ingestion is incremental: fetch from the last stored date minus an overlap window.
- Handle missing values explicitly (drop or return `None`, never silently fill).
- Approved sources only: Yahoo (yfinance), SEC EDGAR, GDELT, Treasury FiscalData. Check a source's terms before adding it. Never commit raw vendor data.
- EDGAR: declared User-Agent (name + email) and at most 10 requests/second. GDELT: cite the GDELT Project with a link to gdeltproject.org; store titles and URLs only, never scraped article text.
- Use `adj_close` for returns. Use `close` only when the raw price level matters.
- Windows are trading days (rows), not calendar days.
- Point-in-time discipline: anything computed "as of" a date must use only data dated on or before it. No look-ahead.
- Yahoo Finance data is unofficial and can be revised or missing; warn on empty results and never treat it as authoritative.
