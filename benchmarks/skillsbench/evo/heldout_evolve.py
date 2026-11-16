"""SE-GoS held-out (train/test) experiment on the 50/37 SkillsBench split.

The main and multi-round tables follow SkillDAG's deployment-utility convention:
all 87 tasks generate the traces and all 87 are re-measured, so the evolved graph
is scored on the very tasks it learned from. This is the control for that.

Split semantics (``evo_data/split.json``, stratified over the eight skill
domains):

  - **train (50)** — the only tasks whose traces feed the evolution. Signals are
    harvested from a static (round-0) run and filtered down to these 50, then the
    L1/L2/L3 levers run once on the cold-start substrate.
  - **eval (37)** — disjoint from train. The evolved graph is measured here, and
    the static baseline is measured on the same 37 for the paired comparison.

Round 1 of the multi-round table and the graph produced here are *different*
artifacts: 1,102 edges (all 87 traces) vs 990 edges (50 train traces). They are
never interchangeable.

Usage (from benchmarks/skillsbench):
    PYTHONPATH=$PWD python -m evo.heldout_evolve \
      --gos-workspace-dir ../../data/gos_workspace \
      --env-file ../../.env --n-attempts 2 --concurrency 20

    # only rebuild the graph from traces that are already on disk
    PYTHONPATH=$PWD python -m evo.heldout_evolve --evolve-only

By default the traces come from the existing full-87 static run
(``jobs/graphskills-full87-k1-1000-train``) filtered to the 50 training tasks —
that is how the published graph was produced, and it costs no extra agent runs.
Pass ``--train-job`` to point elsewhere, or ``--run-train`` to run a dedicated
50-task static job when that job is not available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.skillsbench.evo import signals
from benchmarks.skillsbench.evo.analyze import is_infra_exception, mean
from benchmarks.skillsbench.evo.run_full87_experiments import (
    SKILLSBENCH_ROOT,
    build_token_sem_base,
    evolve_bundle_serial,
    generate_tree,
    job_complete,
    make_config,
    run_harbor,
    write_yaml,
)

DEFAULT_COND = "L1L2L3"


def _job_dir(job_name: str) -> Path:
    return SKILLSBENCH_ROOT / "jobs" / job_name


def _task_of(data: dict, trial_dir_name: str, tasks: set[str]) -> str | None:
    """Resolve a trial to its task name.

    ``result.json`` carries the full task name; fall back to matching the trial
    directory, which Harbor truncates to 32 characters.
    """
    name = data.get("task_name")
    if name in tasks:
        return name
    prefix = trial_dir_name.split("__")[0]
    return next((t for t in tasks if t.startswith(prefix)), None)


def stats_over_tasks(job_dir: Path, tasks: set[str]) -> dict:
    """Pooled R / T over the trials of ``tasks`` only.

    Same accounting as ``pooled_reward_stats``: attempts lost to harness
    failures are excluded from both numerator and denominator rather than scored
    as zero.
    """
    scored: list[float] = []
    tokens: list[float] = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir() and "__" in p.name):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if is_infra_exception(data):
            continue
        if _task_of(data, trial_dir.name, tasks) is None:
            continue
        reward = ((data.get("verifier_result") or {}).get("rewards", {}) or {}).get("reward")
        scored.append(float(reward) if reward is not None else 0.0)
        n_in = (data.get("agent_result") or {}).get("n_input_tokens")
        if n_in is not None:
            tokens.append(float(n_in))
    return {
        "R": mean(scored),
        "T_M": (mean(tokens) / 1e6) if tokens else None,
        "n_scored": len(scored),
    }


def run_static_tree(
    job_name: str,
    task_names: list[str],
    static_bundle: Path,
    tree_root: Path,
    skillset_name: str,
    gos_workspace: Path,
    size: int,
    n_attempts: int,
    concurrency: int,
    env_file: Path,
) -> None:
    """Run the static (round-0) graph on a task set, mounting the base bundle."""
    generate_tree(
        "graphskills", "static", static_bundle, static_bundle, task_names,
        skillset_name, gos_workspace, tree_root, size,
    )
    cfg = make_config(job_name, task_names, tree_root, "graphskills",
                      n_attempts, concurrency)
    yaml_p = tree_root / "eval.yaml"
    write_yaml(cfg, yaml_p)
    run_harbor(yaml_p, env_file, job_name)


def main() -> None:
    ap = argparse.ArgumentParser(description="SE-GoS held-out (train/test) experiment.")
    ap.add_argument("--split", type=Path, default=SKILLSBENCH_ROOT / "evo_data" / "split.json")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--semantic-k", type=int, default=1,
                    help="cold-start kNN density; 1 is the substrate every table uses")
    ap.add_argument("--cond", type=str, default=DEFAULT_COND)
    ap.add_argument("--gos-workspace-dir", type=Path, required=False,
                    default=SKILLSBENCH_ROOT.parents[1] / "data" / "gos_workspace")
    ap.add_argument("--env-file", type=Path, default=Path("../../.env"))
    ap.add_argument("--out-root", type=Path,
                    default=SKILLSBENCH_ROOT / "generated" / "heldout_k1_split")
    ap.add_argument("--n-attempts", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--train-job", action="append", default=[],
                    help="job dir (under jobs/) whose traces feed the evolution; "
                         "repeatable. Default: the full-87 static run.")
    ap.add_argument("--static-job", default=None,
                    help="job dir for the static baseline row on the eval tasks. "
                         "Default: the full-87 static run.")
    ap.add_argument("--run-train", action="store_true",
                    help="run a dedicated 50-task static job instead of reusing "
                         "an existing job's traces")
    ap.add_argument("--evolve-only", action="store_true",
                    help="produce the evolved graph and stop (no eval job)")
    ap.add_argument("--l3-from", type=Path, default=None,
                    help="recover the L3 delta by diffing descriptions against "
                         "this reference bundle instead of calling the LLM. "
                         "L1/L2 never touch descriptions, so this reproduces a "
                         "published graph exactly and costs no API calls.")
    args = ap.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_tasks: list[str] = split["train"]
    eval_tasks: list[str] = split["eval"]
    overlap = set(train_tasks) & set(eval_tasks)
    if overlap:
        raise SystemExit(f"split is not disjoint: {sorted(overlap)}")
    print(f"[heldout] {len(train_tasks)} train / {len(eval_tasks)} eval tasks", flush=True)

    size = args.size
    skillset = f"skills_{size}"
    gos_ws = args.gos_workspace_dir / f"skills_{size}_v1"
    # Matches the published job names (and what download_data.sh --traces
    # restores), so an existing job directory is resumed rather than re-run.
    job_suffix = f"graphskills-full87-k{args.semantic_k}-{size}-heldout"

    args.out_root.mkdir(parents=True, exist_ok=True)

    # --- substrate ---------------------------------------------------------
    base = build_token_sem_base(
        size, args.gos_workspace_dir, SKILLSBENCH_ROOT / "generated" / "shared",
        semantic_k=args.semantic_k,
    )

    # --- training traces ---------------------------------------------------
    train_jobs = [Path(j) if Path(j).is_absolute() else _job_dir(j)
                  for j in (args.train_job or [f"graphskills-full87-k1-{size}-train"])]
    existing = [j for j in train_jobs if j.is_dir()]
    if args.run_train or not existing:
        train_job_name = f"{job_suffix}-train"
        if not job_complete(_job_dir(train_job_name), len(train_tasks), args.n_attempts):
            run_static_tree(
                train_job_name, train_tasks, base, args.out_root / "train",
                skillset, gos_ws, size, args.n_attempts, args.concurrency, args.env_file,
            )
        train_jobs = [_job_dir(train_job_name)]
    else:
        if len(existing) != len(train_jobs):
            missing = [j.name for j in train_jobs if j not in existing]
            print(f"[heldout] skipping missing job(s): {missing}", flush=True)
        train_jobs = existing

    sig = signals.parse_jobs(train_jobs, base, args.out_root / "signals.json")
    sig["trials"] = {t: v for t, v in sig["trials"].items() if t in set(train_tasks)}
    print(f"[heldout] {sig['trial_count']} trials from "
          f"{[j.name for j in train_jobs]} -> {len(sig['trials'])} on train tasks",
          flush=True)

    # --- evolve on the training tasks only ---------------------------------
    bundle = json.loads(base.read_text(encoding="utf-8"))

    if args.l3_from is not None and "L3" in args.cond:
        # L1/L2 never touch descriptions, so diffing a reference bundle against
        # the base recovers the L3 delta exactly. evolve_bundle_serial reuses an
        # existing delta_<cond>_L3.json, so seeding it here skips the LLM call.
        reference = json.loads(args.l3_from.read_text(encoding="utf-8"))
        base_desc = {s["name"]: s.get("description") for s in bundle["skills"]}
        l3 = {s["name"]: s["description"] for s in reference["skills"]
              if s["description"] != base_desc.get(s["name"])}
        cache = args.out_root / f"delta_{args.cond}_L3.json"
        cache.write_text(json.dumps(l3, indent=2), encoding="utf-8")
        print(f"[heldout] recovered {len(l3)} L3 description edits from "
              f"{args.l3_from.name} (no LLM call)", flush=True)

    evolved = evolve_bundle_serial(
        bundle, sig, set(train_tasks), args.cond, out_dir=args.out_root
    )
    bundle_path = args.out_root / f"bundle_{args.cond}_train{len(train_tasks)}.json"
    bundle_path.write_text(json.dumps(evolved, indent=1), encoding="utf-8")
    print(f"[heldout] evolved graph: {len(evolved['edges'])} edges -> {bundle_path}",
          flush=True)
    if args.evolve_only:
        return

    # --- measure the evolved graph on the held-out tasks -------------------
    eval_job = f"{job_suffix}-{args.cond}"
    tree_root = args.out_root / "eval"
    generate_tree(
        "graphskills", args.cond, bundle_path, base, eval_tasks,
        skillset, gos_ws, tree_root, size,
    )
    cfg = make_config(eval_job, eval_tasks, tree_root, "graphskills",
                      args.n_attempts, args.concurrency)
    yaml_p = args.out_root / f"heldout_{args.cond}.yaml"
    write_yaml(cfg, yaml_p)
    if job_complete(_job_dir(eval_job), len(eval_tasks), args.n_attempts):
        print(f"[heldout] {eval_job} already complete, skipping", flush=True)
    else:
        run_harbor(yaml_p, args.env_file, eval_job)

    # --- report ------------------------------------------------------------
    static_job = (Path(args.static_job) if args.static_job
                  else _job_dir(f"graphskills-full87-k1-{size}-train"))
    static_stats = (stats_over_tasks(static_job, set(eval_tasks))
                    if static_job.is_dir() else {"R": None, "T_M": None, "n_scored": 0})
    evolved_stats = stats_over_tasks(_job_dir(eval_job), set(eval_tasks))

    results = {
        "train_tasks": len(train_tasks),
        "eval_tasks": len(eval_tasks),
        "cond": args.cond,
        f"static_eval{len(eval_tasks)}_R": static_stats["R"],
        "static_T_M": static_stats["T_M"],
        "static_n": static_stats["n_scored"],
        f"{args.cond}_eval{len(eval_tasks)}_R": evolved_stats["R"],
        f"{args.cond}_T_M": evolved_stats["T_M"],
        f"{args.cond}_n": evolved_stats["n_scored"],
        "edges": len(evolved["edges"]),
        "evolved_bundle": str(bundle_path),
    }
    results_path = args.out_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"[heldout] DONE -> {results_path}")


if __name__ == "__main__":
    main()
