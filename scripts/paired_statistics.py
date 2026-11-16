#!/usr/bin/env python3
"""Task-paired inference from the released, two-attempt reward records.

Run from any directory: python3 scripts/paired_statistics.py
Requires NumPy. Does not modify source scores or invoke an evaluation model.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper" / "data"
OUT = ROOT / "build" / "paired_statistics"
SEED = 42
RESAMPLES = 200_000


def format_reward(value):
    """Round displayed percentages consistently at decimal halfway values."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load_scores(path, expected_tasks):
    groups = defaultdict(list)
    seen = set()
    models = set()
    for line in path.read_text().splitlines():
        row = json.loads(line, parse_float=Decimal)
        key = (row["task_name"], row["attempt_id"])
        assert key not in seen, f"Duplicate attempt: {key}"
        seen.add(key)
        value = Decimal(row["reward"])
        assert value.is_finite() and 0 <= value <= 1
        specification = row["allowed_reward_values"]
        if specification.startswith("round(p/n,3), 0<=p<=n<=48, n>=1;"):
            allowed = {Decimal(str(round(p / n, 3))) for n in range(1, 49) for p in range(n + 1)}
        else:
            allowed = [Decimal(x.strip()) for x in specification.split(";")]
        assert value in allowed, f"Undeclared score: {key}: {value}"
        groups[key[0]].append(value)
        models.add(row["model"])
    assert len(groups) == expected_tasks
    assert all(len(v) == 2 for v in groups.values())
    return {k: sum(v) / 2 for k, v in groups.items()}, sorted(models)


def exact_signed_rank(differences):
    # Decimal arithmetic preserves exact zeros and absolute-value ties in
    # the stored scores. Twice the average rank is always an integer.
    nonzero = [d for d in differences if d != 0]
    rank2, offset = {}, 0
    for value, count in sorted(Counter(abs(d) for d in nonzero).items()):
        rank2[value] = 2 * offset + count + 1
        offset += count
    ranks = [rank2[abs(d)] for d in nonzero]
    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    total = sum(ranks)
    distribution = Counter({0: 1})
    for rank in ranks:
        updated = distribution.copy()
        for score, count in distribution.items():
            updated[score + rank] += count
        distribution = updated
    assert sum(distribution.values()) == 2 ** len(nonzero)
    tail = min(positive, total - positive)
    p = min(1.0, 2 * sum(c for s, c in distribution.items() if s <= tail) / 2 ** len(nonzero))
    # Independent enumeration validates the conditional distribution,
    # including ties, for these small nonzero samples.
    if len(ranks) <= 20:
        sums = np.zeros(1, dtype=np.int64)
        for rank in ranks:
            sums = np.concatenate((sums, sums + rank))
        assert Counter(map(int, sums)) == distribution
    return {"nonzero_tasks": len(nonzero), "W_plus": positive / 2,
            "W_minus": (total - positive) / 2, "p_two_sided": p}


def main():
    global DATA, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    DATA, OUT = args.data_root.resolve(), args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    inputs = []
    for protocol, n in [("full87", 87), ("heldout", 37)]:
        means, models = [], []
        for method in ("segos", "skilldag"):
            path = DATA / "experiments" / "skillsbench" / protocol / "scores" / f"{method}_attempt_scores.jsonl"
            values, labels = load_scores(path, n)
            means.append(values)
            models.append(labels)
            inputs.append({"path": str(path.relative_to(DATA)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "model_labels": labels})
        a, b = means
        assert a.keys() == b.keys()
        assert models[0] == models[1]
        names = sorted(a)
        differences = [a[k] - b[k] for k in names]
        x = np.array(differences, dtype=float)
        rng = np.random.default_rng(SEED)  # reset per protocol
        bootstrap = np.concatenate([
            x[rng.integers(0, n, size=(10_000, n))].mean(axis=1)
            for _ in range(RESAMPLES // 10_000)
        ]) * 100
        ci = np.quantile(bootstrap, [.025, .975], method="linear")
        result = {"protocol": protocol, "tasks": n,
                  "wins": sum(d > 0 for d in differences),
                  "ties": sum(d == 0 for d in differences),
                  "losses": sum(d < 0 for d in differences),
                  "segos_reward_pct": float(sum(a.values()) / n * 100),
                  "skilldag_reward_pct": float(sum(b.values()) / n * 100),
                  "difference_pp": float(sum(differences) / n * 100),
                  "bootstrap_ci_pp": ci.tolist(),
                  "signed_rank": exact_signed_rank(differences)}
        results.append(result)
        with (OUT / f"{protocol}_task_pairs.csv").open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["task_name", "segos_mean_reward", "skilldag_mean_reward", "difference_pp"])
            writer.writerows((k, a[k], b[k], (a[k] - b[k]) * 100) for k in names)
    payload = {"seed_per_protocol": SEED, "resamples": RESAMPLES,
               "numpy_version": np.__version__, "rng": "PCG64",
               "quantile_method": "linear", "inputs": inputs, "results": results}
    (OUT / "paired_statistics.json").write_text(json.dumps(payload, indent=2) + "\n")
    header = ["| Protocol | Tasks | W / T / L | SE-GoS R (%) | SkillDAG R (%) | Difference (%) | 95% paired bootstrap CI (%) | Two-sided Wilcoxon p |",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
    tex = []
    for r in results:
        label = "Full-benchmark" if r["protocol"] == "full87" else "Held-out"
        lo, hi = r["bootstrap_ci_pp"]
        p = r["signed_rank"]["p_two_sided"]
        header.append(f'| {label} | {r["tasks"]} | {r["wins"]} / {r["ties"]} / {r["losses"]} | {format_reward(r["segos_reward_pct"])} | {format_reward(r["skilldag_reward_pct"])} | {r["difference_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.4f} |')
        tex.append(f'{label} & {r["tasks"]} & {r["wins"]}/{r["ties"]}/{r["losses"]} & {r["difference_pp"]:+.2f} & [{lo:+.2f}, {hi:+.2f}] & {p:.4f} \\\\')
    (OUT / "paired_table.md").write_text("\n".join(header) + "\n")
    latex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Protocol & Tasks & W/T/L & Difference (\%) & 95\% paired CI (\%) & Wilcoxon $p$ \\",
             r"\midrule", *tex, r"\bottomrule", r"\end{tabular}"]
    (OUT / "paired_table.tex").write_text("% Generated by scripts/paired_statistics.py; do not edit.\n" + "\n".join(latex) + "\n")
    print("\n".join(header))


if __name__ == "__main__":
    main()
