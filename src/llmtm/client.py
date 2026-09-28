"""A small client for any OpenAI-compatible chat endpoint, with retries and a usage ledger.

The key and endpoint come from the environment (LLM_API_KEY, LLM_BASE_URL), never from code.
Every call appends a line to a JSONL ledger with its model, token counts and latency, and the
client refuses new calls once `max_tokens_total` has been spent.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reply:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency: float


class ChatClient:
    def __init__(self, model: str, ledger: Path, max_tokens_total: int = 5_000_000, timeout: int = 300):
        self.model = model
        self.base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        self.key = os.environ.get("LLM_API_KEY")
        if not self.key:
            raise RuntimeError("set LLM_API_KEY (and LLM_BASE_URL for an endpoint other than the default)")
        self.ledger = ledger
        self.max_tokens_total = max_tokens_total
        self.timeout = timeout
        self._lock = threading.Lock()
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.spent = sum(r["prompt_tokens"] + r["completion_tokens"] for r in read_ledger(ledger))

    def chat(self, system: str, user: str, tag: str, attempts: int = 5) -> Reply:
        if self.spent >= self.max_tokens_total:
            raise BudgetExceeded(f"{self.spent:,} tokens spent, budget {self.max_tokens_total:,}")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, ValueError) as err:
                if attempt == attempts - 1:
                    raise
                wait = 2**attempt * 5
                print(f"  {tag}: {err!s:.120} - retrying in {wait}s", flush=True)
                time.sleep(wait)
        usage = data.get("usage", {})
        reply = Reply(
            content=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency=time.perf_counter() - start,
        )
        with self._lock:
            self.spent += reply.prompt_tokens + reply.completion_tokens
            with self.ledger.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "tag": tag,
                            "model": reply.model,
                            "prompt_tokens": reply.prompt_tokens,
                            "completion_tokens": reply.completion_tokens,
                            "latency_s": round(reply.latency, 2),
                        }
                    )
                    + "\n"
                )
        return reply


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
