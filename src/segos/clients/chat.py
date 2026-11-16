"""Explicitly configured Chat Completions client for offline rewriting."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any





def benchmark_model_from_jobs(jobs_roots: list[Path]) -> str | None:
    """Read the executed model from Harbor's saved job configuration."""
    models: set[str] = set()
    for root in jobs_roots:
        config_path = root / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        models.update(
            agent["model_name"] for agent in config.get("agents", [])
            if agent.get("model_name")
        )
    if len(models) > 1:
        raise ValueError(f"Evolution traces mix benchmark models: {sorted(models)}")
    return next(iter(models), None)


def resolve_rewriter_model(
    signals: dict[str, Any], benchmark_model: str | None = None,
) -> str:
    """Follow the benchmark model; never silently select a different rewriter."""
    recorded = signals.get("benchmark_model")
    job_model = benchmark_model_from_jobs(
        [Path(root) for root in signals.get("source_jobs", [])]
    )
    models = {value for value in (recorded, job_model, benchmark_model) if value}
    if len(models) > 1:
        raise ValueError(f"Benchmark model disagrees with evolution signals: {sorted(models)}")
    if not models:
        raise ValueError(
            "Cannot determine the benchmark model for L3. Preserve the source "
            "job config.json or signals.benchmark_model, or pass --model for "
            "legacy signals. EVO_LLM_MODEL is not a benchmark-model record."
        )
    model = models.pop()
    if not isinstance(model, str) or not model.strip():
        raise ValueError("The benchmark model must be a nonempty string")
    return model


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("EVO_LLM_BASE") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("EVO_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set EVO_LLM_API_KEY or OPENAI_API_KEY")
        self.model = model or os.getenv("EVO_LLM_MODEL")
        if not self.model:
            raise ValueError("An explicit model is required for the LLM client")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        max_retries: int = 4,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/chat/completions"
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last_error = f"status={exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}"
                if exc.code in (400, 401, 404):
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            else:
                try:
                    return body["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError) as exc:
                    last_error = f"unexpected response shape: {json.dumps(body)[:300]}"

            if attempt < max_retries:
                time.sleep(min(8.0, 1.5 * attempt))

        raise RuntimeError(f"LLM chat failed after {max_retries} attempts: {last_error}")
