"""Render a validated briefing or answer as Markdown with clickable citations."""
from __future__ import annotations

import duckdb


def _citation(con: duckdb.DuckDBPyConnection, source_id: str) -> str:
    """Document IDs become links (looked up in DuckDB); data paths stay as plain `code`."""
    row = con.execute("SELECT url, title FROM documents WHERE doc_id = ?", [source_id]).fetchone()
    return f"[{source_id}]({row[0]})" if row else f"`{source_id}`"


def _claim_line(con: duckdb.DuckDBPyConnection, claim: dict) -> str:
    sources = ", ".join(_citation(con, s) for s in claim["source_ids"])
    return f"- {claim['text']} _({sources})_"


def render_markdown(output: dict, con: duckdb.DuckDBPyConnection) -> str:
    if not output:
        return "_No briefing was produced (see the validation line for the reason)._"
    lines: list[str] = []
    if "sections" in output:
        lines.append(f"# Daily market briefing (as of {output.get('as_of') or 'latest data'})")
        for section in output["sections"]:
            if section["claims"]:
                lines += ["", f"## {section['name'].replace('_', ' ').title()}"]
                lines += [_claim_line(con, c) for c in section["claims"]]
    else:
        lines.append(f"# {output['question']}")
        lines += [_claim_line(con, c) for c in output.get("claims", [])] or ["_The available data cannot answer this._"]
    lines += ["", "_News headlines from the GDELT Project (https://www.gdeltproject.org/). Not investment advice._"]
    return "\n".join(lines)
