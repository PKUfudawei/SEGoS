"""Write harbor job configs for the 7 evolved-condition eval runs.

Each config targets the 37-task eval subset and the condition's generated
dataset directory, with the same codex agent protocol as the N=1000 baseline.
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Any

CONDITIONS = ["L1", "L2", "L3", "L1L2", "L1L3", "L2L3", "L1L2L3"]


def build_config(condition: str, eval_tasks: list[str], out_root: Path) -> dict[str, Any]:
    return {
        "job_name": f"graphskills-codex-n1000-evo-{condition}",
        "jobs_dir": "jobs",
        "n_attempts": 2,
        "timeout_multiplier": 2.5,
        "debug": False,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": 8,
            "quiet": False,
            "retry": {
                "max_retries": 3,
                "include_exceptions": None,
                "exclude_exceptions": [
                    "BadRequestError",
                    "RateLimitError",
                    "AgentTimeoutError",
                ],
                "wait_multiplier": 1.0,
                "min_wait_sec": 1.0,
                "max_wait_sec": 60.0,
            },
            "kwargs": {},
        },
        "environment": {
            "type": "docker",
            "import_path": None,
            "force_build": False,
            "delete": True,
            "override_cpus": None,
            "override_memory_mb": None,
            "override_storage_mb": None,
            "override_gpus": None,
            "kwargs": {},
        },
        "verifier": {
            # Heavy verifiers (e.g. fix-erlang-ssh-cve full rebuilds) exceeded
            # the harbor default 3000s and got classified as infra; raise the
            # cap and stop excluding VerifierTimeoutError from retries so the
            # final infra-refill pass can actually complete those attempts.
            "override_timeout_sec": 5400,
            "max_timeout_sec": None,
            "disable": False,
        },
        "metrics": [],
        "agents": [
            {
                "name": "codex",
                "import_path": None,
                "model_name": os.environ.get("SEGOS_MODEL", "openai/gpt-5.2-codex"),
                "override_timeout_sec": None,
                "override_setup_timeout_sec": None,
                "max_timeout_sec": 7200,
                "kwargs": {},
                "env": {

                },
            }
        ],
        "datasets": [
            {
                "task_names": eval_tasks,
                "exclude_task_names": None,
                "path": str((out_root / condition / "tasks_graph_skills").resolve()),
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 7 evolved-condition harbor configs.")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="config output directory")
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    eval_tasks = split["eval"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for condition in CONDITIONS:
        config = build_config(condition, eval_tasks, args.out_root)
        path = args.out_dir / f"codex-evo-{condition}.yaml"
        import yaml  # noqa: PLC0415

        path.write_text(
            yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"wrote {path} ({len(eval_tasks)} tasks)")


if __name__ == "__main__":
    main()
