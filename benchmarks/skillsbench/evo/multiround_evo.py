"""SE-GoS multi-round evolution (evo2, evo3, ...) on the full-87 protocol.

Round semantics (each round = one full L1L2L3 evolution pass):

  evo0 = static GoS:      base token-sem bundle, train job on all 87 tasks
                         (already produced by run_full87_experiments).
  evo1 = SE-GoS round 1:  L1L2L3 deltas applied once onto the base, eval on
                         all 87 tasks (already produced, job
                         graphskills-full87-{size}-eval-L1L2L3).
  evoN (N >= 2):         extract signals from the *previous* round's eval job,
                         recompute L1/L2/L3 deltas on the previous round's
                         bundle, apply them to get bundle_N, then eval on all
                         87 tasks. Avoid evidence accumulates actual attempts
                         from rounds 0 through N-1, preserving success vetoes.

Each new round costs one job (87 tasks x n_attempts).  Jobs are resume-safe:
a round whose job already has every expected result.json is skipped.

Usage (from benchmarks/skillsbench):
    PYTHONPATH=$PWD python -m evo.multiround_evo \
      --sizes 1000 \
      --gos-workspace-dir ../../data/gos_workspace \
      --env-file ../../.env \
      --n-attempts 2 --concurrency 20 \
      --rounds 3                 # evolve up to evo3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmarks.skillsbench.evo import signals
from benchmarks.skillsbench.evo.analyze import attempt_token_stats, pooled_reward_stats
from benchmarks.skillsbench.evo.run_full87_experiments import (
    SKILLSBENCH_ROOT,
    build_token_sem_base,
    generate_tree,
    job_complete,
    make_config,
    run_harbor,
    write_yaml,
)

# Where run_full87_experiments.py already wrote evo0/evo1 artifacts.
SEED_TREE = "generated/full87_evo"          # relative to skillsbench root
SEED_ROUND0_JOB = "graphskills-full87-{size}-train"
SEED_ROUND1_JOB = "graphskills-full87-{size}-eval-L1L2L3"


def _job_dir(job_name: str) -> Path:
    return SKILLSBENCH_ROOT / "jobs" / job_name


def main() -> None:
    ap = argparse.ArgumentParser(description="SE-GoS multi-round evolution (full-87 protocol).")
    ap.add_argument("--sizes", type=str, default="1000", help="comma-separated sizes")
    ap.add_argument("--gos-workspace-dir", type=Path, required=True)
    ap.add_argument("--env-file", type=Path, default=Path("../../.env"))
    ap.add_argument("--out-root", type=Path, default=Path("generated/multiround_evo"))
    ap.add_argument("--n-attempts", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3,
                    help="highest evo round to reach; evo0/evo1 must already exist "
                         "from run_full87_experiments. Rounds >= 2 are run here.")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    tasks_root = SKILLSBENCH_ROOT / "tasks"
    task_names = sorted(
        d.name for d in tasks_root.iterdir() if d.is_dir() and d.name != "__pycache__"
    ) if tasks_root.is_dir() else []
    if not task_names:
        raise SystemExit("no tasks found under benchmarks/skillsbench/tasks/")

    for size in sizes:
        print(f"[multiround] size {size}, {len(task_names)} tasks, target <= evo{args.rounds}", flush=True)
        skillset = f"skills_{size}"
        gos_ws = args.gos_workspace_dir / f"skills_{size}_v1"

        # --- base bundle (evo0 substrate) ----------------------------------
        base = build_token_sem_base(
            size, args.gos_workspace_dir, SKILLSBENCH_ROOT / "generated" / "shared"
        )

        # --- evo1 bundle must exist (produced by run_full87 L1L2L3) --------
        seed_dir = SKILLSBENCH_ROOT / SEED_TREE / str(size)
        evo1_bundle = seed_dir / "bundle_L1L2L3.json"
        if not evo1_bundle.exists():
            raise SystemExit(
                f"evo1 bundle not found: {evo1_bundle}. Run run_full87_experiments "
                "first (it produces the L1L2L3 bundle and eval job)."
            )

        # --- verify evo0 / evo1 jobs are complete --------------------------
        r0_job = SEED_ROUND0_JOB.format(size=size)
        r1_job = SEED_ROUND1_JOB.format(size=size)
        for name in (r0_job, r1_job):
            if not job_complete(_job_dir(name), len(task_names), args.n_attempts):
                print(f"[multiround] WARNING: {name} not complete "
                      f"({len(list(_job_dir(name).glob('*__*'))) if _job_dir(name).exists() else 0} dirs)")

        # Track per-round bundle + eval job.  evo0 = base (train), evo1 = L1L2L3.
        summary = {
            "size": size,
            "round0_job": r0_job,
            "round1_job": r1_job,
            "rounds": [],
        }
        prev_bundle: Path = evo1_bundle
        prev_job: str = r1_job
        initial_signals = signals.parse_jobs([_job_dir(r0_job)], base)
        avoid_history = [dict(t, round=0) for t in initial_signals["attempts"]]

        for k in range(2, args.rounds + 1):
            rdir = args.out_root / str(size) / f"round{k}"
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"[multiround] ===== EVO {k} =====", flush=True)

            # 1) signals from the previous round's eval job (evo{k-1})
            sig = signals.parse_jobs(
                [_job_dir(prev_job)], prev_bundle
            )
            sig["trials"] = {t: v for t, v in sig["trials"].items() if t in set(task_names)}
            avoid_history.extend(dict(t, round=k-1) for t in sig["attempts"])
            sig["avoid_trials"] = [t for t in avoid_history if t["task_name"] in set(task_names)]
            (rdir / "signals.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
            print(f"[multiround] evo{k}: {sig['trial_count']} trials from {prev_job}", flush=True)

            # 2) evolve serially (L1 -> L2 -> L3, each on the settled graph of
            #    the earlier levers) -> bundle_k
            cur_bundle = json.loads(prev_bundle.read_text(encoding="utf-8"))
            from benchmarks.skillsbench.evo.run_full87_experiments import evolve_bundle_serial  # noqa: PLC0415
            new_bundle = evolve_bundle_serial(
                cur_bundle, sig, set(task_names), "L1L2L3", out_dir=rdir
            )
            bundle_k = rdir / "bundle.json"
            bundle_k.write_text(json.dumps(new_bundle, indent=1), encoding="utf-8")
            print(f"[multiround] evo{k}: edges={len(new_bundle['edges'])} "
                  f"-> {bundle_k}", flush=True)

            # 3) eval bundle_k on all 87 tasks
            job_name = f"graphskills-full87-{size}-evo{k}"
            tree_root = rdir / "eval"
            generate_tree(
                "graphskills", f"evo{k}", bundle_k, base, task_names,
                skillset, gos_ws, tree_root, size,
            )
            cfg = make_config(job_name, task_names, tree_root, "graphskills",
                              args.n_attempts, args.concurrency)
            yaml_p = rdir / "eval.yaml"
            write_yaml(cfg, yaml_p)
            if job_complete(_job_dir(job_name), len(task_names), args.n_attempts):
                print(f"[multiround] evo{k} job already complete, skipping", flush=True)
            else:
                run_harbor(yaml_p, args.env_file, job_name)

            tok = attempt_token_stats(_job_dir(job_name))
            st = pooled_reward_stats(_job_dir(job_name))
            l3_file = rdir / "delta_L1L2L3_L3.json"
            n_l3 = len(json.loads(l3_file.read_text(encoding="utf-8"))) if l3_file.exists() else 0
            info = {
                "round": k,
                "R": st["R"],
                "T_M": tok["T_M"],
                "n": st["n_scored"],
                "n_attempts_with_tokens": tok["n_tokens"],
                "edges": len(new_bundle["edges"]),
                "L3_descriptions": n_l3,
            }
            summary["rounds"].append(info)
            print(f"[multiround] evo{k}: {info}", flush=True)

            prev_bundle = bundle_k
            prev_job = job_name

        (args.out_root / str(size) / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"[multiround] DONE -> {args.out_root / str(size) / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
