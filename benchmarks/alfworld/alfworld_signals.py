"""Extract SE-GoS evolution signals from ALFWorld run results.

Each episode result (benchmarks/alfworld/jobs/<model>/<split>_<exp>_mode_<mode>/idx_*.json)
carries:
  - reward / task_done
  - runtime_skill_events: [ {step, trigger, query, skill_names} ]
      trigger == "agent_request:read_skill"  -> skill actually used by the agent
      retrieval triggers (gos_retrieve / initial retrieve / ...) -> retrieved skills

Output format matches evo/levers expectations: trials: {task: {reward,
queries, used_skills}} so the existing L1/L2/L3 machinery can be reused.

Usage (from the repository root):
    python benchmarks/alfworld/alfworld_signals.py \
      --results-dir benchmarks/alfworld/jobs/deepseek-v4-flash-0731 \
      --out benchmarks/alfworld/generated/alfworld_signals.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RETRIEVAL_TRIGGERS = {"agent_request:gos_retrieve", "agent_request:vector_retrieve",
                      "initial_retrieve", "auto_retrieve"}
USE_TRIGGERS = {"agent_request:read_skill"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None,
                    help="Benchmark model for legacy episode records without model_name.")
    args = ap.parse_args()

    trials: dict[str, dict] = {}
    models = {args.model} if args.model else set()
    for f in sorted(args.results_dir.rglob("idx_*.json")):
        data = json.loads(f.read_text())
        if data.get("model_name"):
            models.add(data["model_name"])
        task = data.get("name") or f.stem
        if task in trials:
            raise ValueError(f"Multiple attempts for {task}; select one training attempt directory")
        reward = float(data.get("reward") or 0.0)
        events = data.get("runtime_skill_events") or []

        # Initial retrieval is captured directly on the result record (and, since
        # the runtime-events fix, also as an `initial_retrieve` event).
        retrieved: list[str] = [n for n in (data.get("relevant_skill_names") or []) if n]
        used: list[str] = []
        queries: list[str] = [data.get("retrieval_query") or ""] if retrieved else []
        for evt in events:
            trigger = evt.get("trigger", "")
            names = [n for n in (evt.get("skill_names") or []) if n]
            q = evt.get("query") or ""
            if trigger == "agent_request:read_skill":
                used.extend(names)
            elif (
                trigger in RETRIEVAL_TRIGGERS
                or trigger.startswith("agent_request:")
                or trigger.startswith("runtime_")
            ):
                retrieved.extend(names)
                if q:
                    queries.append(q)

        trials[task] = {
            "task_name": task,
            "attempt_id": data.get("attempt_id", "1"),
            "reward": reward,
            "queries": queries,
            "retrieved_skills": sorted(set(retrieved)),
            "used_skills": sorted(set(used)),
            "used_skills_ordered": list(dict.fromkeys(used)),
        }

    if len(models) > 1:
        raise ValueError(f"ALFWorld signals mix benchmark models: {sorted(models)}")
    signals = {
        "signal_schema_version": 2,
        "benchmark_model": next(iter(models), None),
        "source": str(args.results_dir),
        "trial_count": len(trials),
        "trials": trials,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(signals, indent=2), encoding="utf-8")

    n_success = sum(1 for t in trials.values() if t["reward"] > 0)
    n_used = sum(1 for t in trials.values() if t["used_skills"])
    print(f"[alfsignals] {len(trials)} trials, {n_success} success, "
          f"{n_used} with used_skills -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
