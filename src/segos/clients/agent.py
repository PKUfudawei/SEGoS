"""Run the public Codex CLI with an explicitly configured model endpoint."""
from __future__ import annotations

import json
import os
import shutil
import subprocess


def run_agent(messages, model, *, timeout=240):
    executable = shutil.which(os.getenv("SEGOS_AGENT_BIN", "codex"))
    if executable is None:
        raise RuntimeError("Install the public Codex CLI or set SEGOS_AGENT_BIN")
    if not model:
        raise ValueError("An explicit model is required")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY for the configured endpoint")
    settings = {
        "model_provider": "segos",
        "model_providers.segos.name": "Configured endpoint",
        "model_providers.segos.base_url": base_url,
        "model_providers.segos.env_key": "OPENAI_API_KEY",
        "model_providers.segos.wire_api": "responses",
    }
    command = [executable, "exec", "--skip-git-repo-check", "--json", "-m", model]
    for key, value in settings.items():
        command.extend(["-c", f"{key}={json.dumps(value)}"])
    command.append("-")
    prompt = "\n\n".join(f"[{m.get('role', 'user')}]\n{m.get('content', '')}" for m in messages)
    process = subprocess.run(command, input=prompt, text=True, capture_output=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f"Agent exited with status {process.returncode}")
    text = None
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for line in process.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text")
        if event.get("type") == "turn.completed":
            for key in usage:
                usage[key] += int((event.get("usage") or {}).get(key) or 0)
    if not text:
        raise RuntimeError("Agent returned no assistant message")
    return text, usage
