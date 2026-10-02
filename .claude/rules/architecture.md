---
description: Data flow, module layering, and the phase plan
---

# Architecture

## Data flow
approved free sources (Yahoo/yfinance, SEC EDGAR, GDELT) -> `ingest/` -> DuckDB (numbers + raw documents) and Pinecone (text chunks) -> `analytics/` (numbers) and retrieval (text) -> LangGraph agents -> Haiku narration -> cited briefing / Q&A -> eval -> Streamlit.

## Layering (imports flow downward only)
- `config` and `db` are leaves: they import nothing else from the project.
- `ingest` never imports `analytics` or `agents`.
- `retrieval` (chunking, Pinecone, sync) imports only `config` and `db`; it never imports `analytics` or `agents`.
- `llm` (Haiku client) imports only `config`.
- `analytics` has no LLM and no network access. Pure functions on DataFrames.
- `agents` call `analytics` and retrieval, never the reverse.
- `eval` may read everything.

## Phase plan
0. Scaffolding and context files
1. Price/yield ingestion and regime analytics
2. News + SEC filings ingestion, chunking, embeddings, Pinecone retrieval
3. LangGraph agents: tool nodes, Haiku writer, validator, Q&A
4. Evaluation harness (numeric match, unsupported claims, citation coverage)
5. Streamlit app and final README
