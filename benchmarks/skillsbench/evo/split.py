"""Phase B: 50/37 train/eval split, stratified by task category and reward.

Two-level stratification:
  1. category (from task.toml metadata.category)  -> fixed per-category train targets
  2. reward bucket (1.0 / partial / 0.0 / missing) -> proportional eval allocation
     within each category (largest-remainder method) so the eval subset keeps a
     representative static-performance mix.
Seeded for reproducibility.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

SEED = 2026

# Train targets per category (train 50 / eval 37 overall).
TRAIN_TARGETS = {
    "software-engineering": 9,
    "industrial-physical-systems": 8,
    "office-white-collar": 8,
    "natural-science": 8,
    "finance-economics": 5,
    "mathematics-or-formal-reasoning": 5,
    "cybersecurity": 4,
    "media-content-production": 3,
}


def reward_bucket(reward: float | None) -> str:
    if reward is None:
        return "missing"
    if reward >= 0.9:
        return "1.0"
    if reward > 0.0:
        return "partial"
    return "0.0"


def task_category(task_root: Path, task_name: str) -> str:
    task_toml = task_root / task_name / "task.toml"
    if task_toml.exists():
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib  # type: ignore

            with task_toml.open("rb") as f:
                data = tomllib.load(f)
            category = (data.get("metadata") or {}).get("category")
            if category:
                return str(category)
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def _largest_remainder_alloc(total: int, weights: list[int], rng: random.Random) -> list[int]:
    if total <= 0:
        return [0] * len(weights)
    weight_sum = sum(weights)
    raw = [total * w / weight_sum for w in weights]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    if remainder < 0:
        raise RuntimeError("negative remainder in largest-remainder allocation")
    order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - base[i], rng.random()),
        reverse=True,
    )
    for i in order[:remainder]:
        base[i] += 1
    return base


def make_split(
    signals: dict[str, Any],
    tasks_root: Path,
    *,
    seed: int = SEED,
    out_path: Path | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    trials = signals["trials"]

    # group tasks by category, then by reward bucket
    by_category: dict[str, dict[str, list[str]]] = {}
    for task_name, trial in trials.items():
        category = task_category(tasks_root, task_name)
        bucket = reward_bucket(trial.get("reward"))
        by_category.setdefault(category, {}).setdefault(bucket, []).append(task_name)

    eval_tasks: list[str] = []
    train_tasks: list[str] = []
    eval_meta: dict[str, Any] = {}

    categories = sorted(by_category)
    for category in categories:
        buckets = by_category[category]
        total = sum(len(t) for t in buckets.values())
        train_target = TRAIN_TARGETS.get(category, max(0, total // 2))
        eval_target = total - train_target
        if eval_target < 0:
            eval_target, train_target = train_target, total - train_target

        bucket_names = sorted(buckets)
        weights = [len(buckets[b]) for b in bucket_names]
        alloc = _largest_remainder_alloc(eval_target, weights, rng)

        chosen_eval: list[str] = []
        for bucket, count in zip(bucket_names, alloc):
            pool = buckets[bucket]
            count = min(count, len(pool))
            rng.shuffle(pool)
            chosen_eval.extend(pool[:count])

        # guard against rounding drift: exact-match eval_target from remaining pool
        if len(chosen_eval) < eval_target:
            remaining = [
                t for bucket in bucket_names for t in buckets[bucket] if t not in chosen_eval
            ]
            rng.shuffle(remaining)
            chosen_eval.extend(remaining[: eval_target - len(chosen_eval)])
        elif len(chosen_eval) > eval_target:
            chosen_eval = rng.sample(chosen_eval, eval_target)

        chosen_eval_set = set(chosen_eval)
        chosen_train = [
            t for bucket in bucket_names for t in buckets[bucket] if t not in chosen_eval_set
        ]

        eval_tasks.extend(sorted(chosen_eval))
        train_tasks.extend(sorted(chosen_train))
        eval_meta[category] = {
            "total": total,
            "train": len(chosen_train),
            "eval": len(chosen_eval),
        }

    split = {
        "seed": seed,
        "train": sorted(train_tasks),
        "eval": sorted(eval_tasks),
        "train_count": len(train_tasks),
        "eval_count": len(eval_tasks),
        "eval_meta": eval_meta,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


if __name__ == "__main__":
    signals_path = Path(sys.argv[1])
    tasks_root = Path(sys.argv[2])
    out = Path(sys.argv[3])
    signals = json.loads(signals_path.read_text(encoding="utf-8"))
    split = make_split(signals, tasks_root, out_path=out)
    print(f"train={split['train_count']} eval={split['eval_count']}")
