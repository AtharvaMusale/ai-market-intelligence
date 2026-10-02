"""Claude Haiku 4.5 client: disk-cached by input hash, capped output, token and cost logging.

Haiku is the only model used. Responses are cached in data/cache/llm (git-ignored) so repeating
an identical request costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_intel.config import (
    LLM_CACHE_DIR,
    LLM_INPUT_USD_PER_MTOK,
    LLM_MODEL,
    LLM_OUTPUT_USD_PER_MTOK,
    get_anthropic_key,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    from_cache: bool = False


@dataclass
class CostTracker:
    """Running totals for one run."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, result: LLMResult) -> None:
        self.calls += 1
        self.cache_hits += int(result.from_cache)
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.cost_usd += result.cost_usd

    def summary(self) -> dict:
        return {**self.__dict__, "cost_usd": round(self.cost_usd, 6)}


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * LLM_INPUT_USD_PER_MTOK + output_tokens * LLM_OUTPUT_USD_PER_MTOK) / 1_000_000


def cache_key(system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({"model": LLM_MODEL, "system": system, "user": user, "max_tokens": max_tokens})
    return hashlib.sha256(payload.encode()).hexdigest()


class HaikuClient:
    def __init__(self, cache_dir: Path | None = None, client: Any = None) -> None:
        self._cache_dir = Path(cache_dir or LLM_CACHE_DIR)
        self._client = client
        self.tracker = CostTracker()

    def _api(self) -> Any:
        if self._client is None:
            import anthropic  # imported here so tests and --dry-run need no key

            self._client = anthropic.Anthropic(api_key=get_anthropic_key())
        return self._client

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        path = self._cache_dir / f"{cache_key(system, user, max_tokens)}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            result = LLMResult(cached["text"], cached["input_tokens"], cached["output_tokens"], 0.0, True)
            logger.info("LLM cache hit (saved ~$%.4f)", estimate_cost(result.input_tokens, result.output_tokens))
        else:
            response = self._api().messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            if response.stop_reason == "max_tokens":
                logger.warning("LLM output hit max_tokens=%d; JSON may be truncated", max_tokens)
            text = "".join(b.text for b in response.content if b.type == "text")
            usage = response.usage
            result = LLMResult(
                text, usage.input_tokens, usage.output_tokens,
                estimate_cost(usage.input_tokens, usage.output_tokens),
            )
            if response.stop_reason != "max_tokens":  # never cache a truncated answer
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"text": text, "input_tokens": result.input_tokens,
                                            "output_tokens": result.output_tokens}))
            logger.info("LLM call: %d in / %d out tokens, ~$%.4f", result.input_tokens, result.output_tokens, result.cost_usd)
        self.tracker.add(result)
        return result
