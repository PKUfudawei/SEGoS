"""Within-task avoid/conflict mining over existing attempts (free).

Pools attempts per task across the two k=1 jobs that share the substrate the
paper reports: the static train run and the fused L1L2L3 run (~4 attempts/task).
The k=8 lineage also had a three-channel L1L2L3 job to pool; it has no k=1
counterpart, because the interface contrast it belonged to is not part of this
substrate's results.

For each skill pair co-present in some attempt of a task, compares reward of
attempts that read both A and B vs attempts that read only one -- within the
same task, so task difficulty is cancelled. A pair is a candidate conflict if it
is harmful in >=1 task and never beneficial in any other task.

Then estimates the AVOID_BUNDLE_EXCLUSION trigger rate: for each candidate pair,
replay the train queries of the tasks where the pair is involved and count how
often both skills land in the retrieval top-5 simultaneously (the only case the
consumer could change the bundle).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.skillsbench.evo import analyze as A
from benchmarks.skillsbench.evo import signals as S
from benchmarks.skillsbench.evo import levers

BASE = Path("generated/shared/graphskills_bundle_semonly_token_k1_1000.json")
SIGNALS = Path("generated/full87_evo_k1/1000/signals.json")
JOBS = [
    "graphskills-full87-k1-1000-train",
    "graphskills-full87-k1-1000-eval-L1L2L3",
]
MARGIN = 0.20  # reward gap for "harmful" / "beneficial"


def attempts_per_task() -> dict[str, list[dict]]:
    bundle = S.load_bundle(BASE)
    d2n = S.dir_to_name_map(bundle)
    out: dict[str, list[dict]] = defaultdict(list)
    for job in JOBS:
        root = Path("jobs") / job
        if not root.is_dir():
            continue
        for tdir in sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name):
            t = S.extract_trial(tdir, d2n)
            if t is None or not t["used_skills"]:
                continue
            out[t["task_name"] or tdir.name.split("__")[0]].append(t)
    return out


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    tasks = attempts_per_task()
    print(f"tasks with >=1 usable attempt: {len(tasks)}")
    multi = {t: a for t, a in tasks.items() if len(a) >= 3}
    print(f"tasks with >=3 attempts (within-task power): {len(multi)}")

    # pair -> {task: (kind, n_both, n_solo, r_both, r_solo_base)}
    evidence: dict[tuple, dict] = defaultdict(dict)
    from itertools import combinations
    for task, attempts in multi.items():
        rows = [(set(t["used_skills"]), t["reward"]) for t in attempts]
        seen = set()
        for u, _r in rows:
            for p in combinations(sorted(u), 2):
                seen.add(p)
        for (x, y) in seen:
            both = [r for u, r in rows if x in u and y in u]
            only_x = [r for u, r in rows if x in u and y not in u]
            only_y = [r for u, r in rows if y in u and x not in u]
            if not both:
                continue
            solo_arms = [a for a in (only_x, only_y) if a]
            if not solo_arms:
                continue
            rb = mean(both)
            base = min(mean(a) for a in solo_arms)
            if rb <= base - MARGIN:
                evidence[(x, y)][task] = ("harm", len(both), max(len(a) for a in solo_arms), rb, base)
            elif rb >= base + MARGIN:
                evidence[(x, y)][task] = ("benefit", len(both), max(len(a) for a in solo_arms), rb, base)

    candidates = []
    for (x, y), per_task in evidence.items():
        harm_tasks = [t for t, v in per_task.items() if v[0] == "harm"]
        ben_tasks = [t for t, v in per_task.items() if v[0] == "benefit"]
        if harm_tasks and not ben_tasks:
            candidates.append((x, y, harm_tasks, per_task))
    print(f"\ncandidate conflict pairs (harm>=1 task, no contradicting benefit): {len(candidates)}")

    # trigger-rate estimate via train queries on involved tasks
    sig = json.loads(SIGNALS.read_text())
    triggers = 0
    trig_queries = 0
    names_top5 = set()
    for x, y, harm_tasks, per in candidates:
        for task in harm_tasks:
            trial = sig["trials"].get(task)
            if not trial or not trial.get("queries"):
                continue
            for q in trial["queries"]:
                res = levers.retrieve_bundle(S.load_bundle(BASE), q)
                top = {s["name"] for s in (res.get("skills") or [])}
                trig_queries += 1
                if x in top and y in top:
                    triggers += 1
    print(f"trigger-rate check: over {trig_queries} train queries on involved tasks, "
          f"both skills of a candidate pair co-appear in top-5 {triggers} times")
    for x, y, harm_tasks, per in sorted(candidates, key=lambda c: -len(c[2]))[:25]:
        detail = "; ".join(f"{t}: both={v[3]:.2f} solo={v[4]:.2f} (nboth={v[1]},nsolo={v[2]})" for t, v in list(per.items())[:3])
        print(f"  {x} <-> {y}: harm_tasks={len(harm_tasks)} | {detail}")


if __name__ == "__main__":
    main()
