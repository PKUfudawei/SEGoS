"""Offline retrieval-quality comparison across candidate semantic bases.

Zero agent cost: replay the static-baseline retrieval queries against each
bundle and measure whether the skills the agent actually used are retrieved
(hit@8 / best rank / mean retrieved). Uses the same signals as the main
experiment (evo_data/signals.json).

Usage (from benchmarks/skillsbench):
    PYTHONPATH=$PWD python -m evo.offline_compare
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "graphskills_assets"))

from benchmarks.skillsbench.evo.offline import evaluate_condition  # noqa: E402

BUNDLES = {
    "embedding_sem": "generated/shared/graphskills_bundle_semonly.json",
    "token_sem": "generated/shared/graphskills_bundle_semonly_token.json",
}


def main() -> None:
    signals = json.loads(Path("evo_data/signals.json").read_text())
    split = json.loads(Path("evo_data/split.json").read_text())
    eval_tasks = split["eval"]

    print(f"eval tasks: {len(eval_tasks)}", flush=True)
    for name, path in BUNDLES.items():
        bundle = json.loads(Path(path).read_text())
        res = evaluate_condition(bundle, signals, eval_tasks)
        print(f"\n=== {name} ===", flush=True)
        print(f"  edges: {len(bundle['edges'])}", flush=True)
        print(f"  hit@top: {res['hit@5']}/{res['eval_tasks']} "
              f"({res['hit_rate']:.3f})", flush=True)
        print(f"  hit_rate_used: {res['hit_rate_used']:.3f} "
              f"(tasks_with_used={res['tasks_with_used_skills']})", flush=True)
        print(f"  mean_best_rank_of_hits: {res['mean_best_rank_of_hits']:.2f}", flush=True)
        print(f"  mean_retrieved: {res['mean_retrieved']:.1f}", flush=True)


if __name__ == "__main__":
    main()
