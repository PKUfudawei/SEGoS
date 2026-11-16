"""Phase E: offline retrieval metrics across all 8 condition bundles on the eval subset.

Zero agent cost: for each eval task we replay the agent's Step-1 retrieval query
against each condition bundle and measure whether the skills the agent actually
USED make it into the retrieved top-n (hit@5) and at which rank.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graphskills_assets"))

from segos.retrieval import query as gq  # noqa: E402

from benchmarks.skillsbench.evo.levers import retrieve_bundle  # noqa: E402


def evaluate_condition(
    bundle: dict[str, Any],
    signals: dict[str, Any],
    eval_tasks: list[str],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    hits = 0
    tasks_with_used = 0
    total_retrieved = 0
    rank_sum = 0
    hit_ranks = 0

    for task_name in eval_tasks:
        trial = signals["trials"].get(task_name)
        if trial is None or not (trial.get("queries") or []):
            per_task[task_name] = {"status": "no_query"}
            continue
        used_skills = set(trial.get("used_skills") or [])
        query = trial["queries"][0]
        result = retrieve_bundle(bundle, query)
        retrieved_names = [skill["name"] for skill in result.get("skills") or []]
        total_retrieved += len(retrieved_names)

        best_rank = None
        for rank, name in enumerate(retrieved_names, start=1):
            if name in used_skills:
                best_rank = rank
                break

        per_task[task_name] = {
            "query": query,
            "used_skills": sorted(used_skills),
            "retrieved": retrieved_names,
            "best_rank": best_rank,
            "hit": best_rank is not None,
        }
        if used_skills:
            tasks_with_used += 1
        if best_rank is not None:
            hits += 1
            hit_ranks += 1
            rank_sum += best_rank

    return {
        "eval_tasks": len(eval_tasks),
        "tasks_with_used_skills": tasks_with_used,
        "hit@5": hits,
        "hit_rate": hits / max(len(eval_tasks), 1),
        "hit_rate_used": hits / max(tasks_with_used, 1),
        "mean_best_rank_of_hits": rank_sum / max(hit_ranks, 1),
        "mean_retrieved": total_retrieved / max(len(eval_tasks), 1),
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval metrics for 8 condition bundles.")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True, help="out-root of bundles.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    signals = json.loads(args.signals.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    eval_tasks = split["eval"]

    conditions = [
        "static", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3", "L1L2L3",
    ]
    results: dict[str, Any] = {}
    for condition in conditions:
        bundle_path = args.bundles / "_bundles" / f"{condition}.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        results[condition] = evaluate_condition(bundle, signals, eval_tasks)
        summary = results[condition]
        print(
            f"{condition:8s} hit@5={summary['hit@5']}/{summary['eval_tasks']} "
            f"({summary['hit_rate']:.3f}) used-hit={summary['hit_rate_used']:.3f} "
            f"mean_best_rank={summary['mean_best_rank_of_hits']:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
