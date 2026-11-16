"""Run the SE-GoS evolution experiment on the full 87-task SkillsBench set.

Per library size (200/500/1000/2000), all on the full 87 tasks x n_attempts:

  1. static (Initial Graph) on all 87 -- the round-0 cell (the semantic-only initial condition) and the evolution trace source
  2. offline deltas: L1 (workflow + dep + avoid + evidence-backed avoid retraction),
     L2 (semantic attenuation + Hebbian reinforcement),
     L3 (ProTeGi rank>3) -> condition-specific evolved bundles
  3. evolved cells: L1L2L3 first (the SE-GoS headline), then the rest of the
     2^3 factorial at the main size (1000); other sizes run L1L2L3 only
  4. analyze: per-size R/T/S summary written to <out-root>/<size>/summary.json

Static on all 87 is both the round-0 cell and the trace source. Vanilla and
vector retrieval are not reproduced here; the README says where they live.
The held-out protocol lives in evo/heldout_evolve.py.

Usage (from benchmarks/skillsbench):
    PYTHONPATH=$PWD python -m evo.run_full87_experiments \
      --sizes 200,500,1000,2000 \
      --gos-workspace-dir ../../data/gos_workspace \
      --env-file ../../.env \
      --n-attempts 2 --concurrency 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.skillsbench.evo import configs, levers, signals
from benchmarks.skillsbench.evo.analyze import attempt_token_stats, pooled_reward_stats
from benchmarks.skillsbench.evo.llm import resolve_rewriter_model
from segos.evolution.pipeline import evolve_bundle_serial

SKILLSBENCH_ROOT = Path(__file__).resolve().parents[1]

# SE-GoS headline cell first; the rest are the 2^3 factorial ablation cells.
FACTORIAL_CELLS = ["L1L2L3", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3"]
SCALE_CELLS = ["L1L2L3"]
ALL_TASKS = None  # resolved from the benchmark tasks later if needed

# Evolution order within a condition: each lever is computed on the graph that
# results from applying the earlier levers (serial / dependency-ordered), so
# e.g. L2 sees the workflow edges L1 added, and L3's miss detection runs on the
# fully evolved graph.  This is the semantics method.tex describes
# ("refinement of the settled edge set") and differs from computing every delta
# on the base bundle and merging them at once.
LEVER_ORDER = ("L1", "L2", "L3")
DELTA_CACHE_VERSION = "avoid-success-veto-all-attempts-v5"

# 200's igraph is the original official build (the active file was overwritten
# by the ALFWorld harness), so the rebuild uses the .official snapshot.
_ORIGINAL_IGRAPH = {
    200: "graph_igraph_data.pklz.official",
}


def write_yaml(cfg: dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")




def job_complete(job_dir: Path, n_tasks: int, n_attempts: int) -> bool:
    """True if every expected trial already has a usable result (resume skip).

    A dir counts only when its result.json is not a harbor-internal crash
    (attempts whose process was killed before the verifier ran), so leftover
    crash artifacts never mark a job complete and get re-scheduled on relaunch.
    """
    from benchmarks.skillsbench.evo import analyze as evo_analyze

    expected = n_tasks * n_attempts
    if not job_dir.is_dir():
        return False
    done = 0
    for d in os.listdir(job_dir):
        sub = job_dir / d
        rp = sub / "result.json"
        if not (sub.is_dir() and "__" in d and rp.exists()):
            continue
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if evo_analyze.is_harbor_crash(data):
            continue
        done += 1
    return done >= expected


def run_harbor(yaml_path: Path, env_file: Path, job_name: str) -> Path:
    import shutil

    job_dir = SKILLSBENCH_ROOT / "jobs" / job_name
    recorded = job_dir / "config.json"
    if recorded.exists():
        write_yaml(json.loads(recorded.read_text(encoding="utf-8")), yaml_path)
        print(f"[full87] reusing recorded config for {job_name}", flush=True)
    if (job_dir / "lock.json").exists():
        (job_dir / "lock.json").unlink()
        print(f"[full87] removed stale lock for {job_name}", flush=True)
    cmd = ["harbor", "run", "-c", str(yaml_path), "--env-file", str(env_file), "-y"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SKILLSBENCH_ROOT)
    r = subprocess.run(cmd, cwd=SKILLSBENCH_ROOT, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-4000:], file=sys.stdout)
        print(r.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"harbor run failed: {yaml_path.name}")
    return job_dir


def build_token_sem_base(size: int, gos_workspace_dir: Path, out_dir: Path,
                         semantic_k: int = 1) -> Path:
    """Build (or load) the token-overlap sem-only base for a library size.

    The output filename carries the kNN density ``k`` so that different
    cold-start densities never share or overwrite each other's bundle.
    k=8 keeps the legacy name (no suffix) so existing k=8 trees/jobs that
    reference the shared base keep resolving to the same file.
    """
    name = (f"graphskills_bundle_semonly_token_{size}.json"
            if semantic_k == 8
            else f"graphskills_bundle_semonly_token_k{semantic_k}_{size}.json")
    out = out_dir / name
    if out.exists():
        print(f"[full87] {size}: base exists -> {out}", flush=True)
        return out
    ws = gos_workspace_dir / f"skills_{size}_v1"
    igraph = ws / _ORIGINAL_IGRAPH.get(size, "graph_igraph_data.pklz")
    cmd = [
        sys.executable, "-m", "evo.rebuild_graph",
        "--igraph", str(igraph),
        "--edge-types", "sem",
        "--sem-metric", "token",
        "--semantic-k", str(semantic_k),
        "--out", str(out),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SKILLSBENCH_ROOT)
    r = subprocess.run(cmd, cwd=SKILLSBENCH_ROOT, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-3000:], file=sys.stdout)
        print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"rebuild_graph failed for size {size}")
    print(f"[full87] {size}: built base -> {out}", flush=True)
    return out


def generate_tree(
    label: str,
    bundle: Path,
    static_bundle: Path,
    task_names: list[str],
    skillset_name: str,
    gos_workspace: Path,
    output_root: Path,
    size: int,
) -> Path:
    """Generate the per-condition task tree via benchmark.py."""
    benchmark = SKILLSBENCH_ROOT / "graphskills_benchmark.py"
    cmd = [
        sys.executable, str(benchmark),
        "--skillset-name", skillset_name,
        "--gos-workspace", str(gos_workspace.resolve()),
        "--output-root", str(output_root.resolve()),
    ]
    for t in task_names:
        cmd += ["--task", t]
    cmd += ["--skip-allskills", "--skip-vectorskills",
            "--bundle-path", str(static_bundle.resolve()),
            "--bundle-mount", str(bundle.resolve())]

    print(f"[full87] {size}: generating graphskills tree ({len(task_names)} tasks)", flush=True)
    r = subprocess.run(cmd, cwd=SKILLSBENCH_ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], file=sys.stdout)
        print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"tree generation failed for {label}")
    # benchmark.py writes the tree under output_root/tasks_graph_skills;
    # make_config joins that subdir.
    return output_root


def make_config(job_name: str, task_names: list[str], tree_root: Path,
                n_attempts: int, concurrency: int) -> dict:
    cfg = configs.build_config(job_name, task_names, tree_root.parent)
    cfg["datasets"] = [
        {
            "task_names": task_names,
            "exclude_task_names": None,
            "path": str((tree_root / "tasks_graph_skills").resolve()),
        }
    ]
    cfg["job_name"] = job_name
    cfg["n_attempts"] = n_attempts
    cfg["orchestrator"]["n_concurrent_trials"] = concurrency
    cfg["environment"]["force_build"] = True
    return cfg


def run_size(
    size: int,
    gos_workspace_dir: Path,
    env_file: Path,
    out_root: Path,
    n_attempts: int,
    concurrency: int,
    skillset_name: str,
    task_names: list[str],
    factorial: bool,
    semantic_k: int = 1,
) -> dict:
    rdir = out_root / str(size)
    rdir.mkdir(parents=True, exist_ok=True)

    # Job-name stem: k=8 keeps the original names (so a k=8 run can resume
    # against pre-existing jobs); any other k gets a distinct stem so that a
    # different-density cold start never collides with the k=8 artifacts.
    stem = (
        f"graphskills-full87-{size}"
        if semantic_k == 8
        else f"graphskills-full87-k{semantic_k}-{size}"
    )

    # 1) token-overlap sem base (canonical location under generated/shared)
    base = build_token_sem_base(size, gos_workspace_dir,
                                SKILLSBENCH_ROOT / "generated" / "shared",
                                semantic_k)
    static_bundle = base
    gos_workspace = gos_workspace_dir / f"skills_{size}_v1"

    # 2) static (Initial Graph) on all 87 -> round-0 cell + trace source
    print(f"[full87] {size}: ==== STATIC (trace source) ====", flush=True)
    train_tree = rdir / "train"
    train_tree = generate_tree(
        "train", base, static_bundle, task_names,
        skillset_name, gos_workspace, train_tree, size,
    )
    train_job = f"{stem}-train"
    train_cfg = make_config(train_job, task_names, train_tree,
                            n_attempts, concurrency)
    train_yaml = rdir / "train.yaml"
    write_yaml(train_cfg, train_yaml)
    if job_complete(Path("jobs") / train_job, len(task_names), n_attempts):
        print(f"[full87] {size}: static already complete, skipping", flush=True)
    else:
        run_harbor(train_yaml, env_file, train_job)

    # 3) signals + evolved bundles (serial: each lever on the settled graph of
    #    the previous levers, so L2 sees L1's workflow edges and L3's miss
    #    detection runs on the fully evolved graph).
    print(f"[full87] {size}: ==== DELTAS ====", flush=True)
    sig = signals.parse_jobs([Path("jobs") / train_job], base, rdir / "signals.json")
    sig["trials"] = {t: v for t, v in sig["trials"].items() if t in set(task_names)}
    base_bundle = json.loads(base.read_text())

    cells = FACTORIAL_CELLS if factorial else SCALE_CELLS
    cond_bundles: dict[str, Path] = {"static": base}
    for cond in cells:
        new_bundle = evolve_bundle_serial(
            base_bundle, sig, set(task_names), cond, out_dir=rdir
        )
        p = rdir / f"bundle_{cond}.json"
        p.write_text(json.dumps(new_bundle, indent=1), encoding="utf-8")
        cond_bundles[cond] = p
        print(f"[full87] {size}: {cond}: edges={len(new_bundle['edges'])}", flush=True)

    results: dict[str, dict] = {}
    # eval cells (evolved), L1L2L3 first
    print(f"[full87] {size}: ==== EVOLVED EVALS ====", flush=True)
    for cond in ["static"] + cells:
        job_name = f"{stem}-eval-{cond}"
        if cond == "static":
            # static R reported as pooled mean over the train job's attempts
            # (SkillDAG-aligned accounting), not the collapsed per-task signals.
            train_job_name = f"{stem}-train"
            st = pooled_reward_stats(Path("jobs") / train_job_name)
            tok = attempt_token_stats(Path("jobs") / train_job_name)
            results[cond] = {
                "R": st["R"],
                "T_M": tok["T_M"],
                "n": st["n_scored"],
                "n_attempts_with_tokens": tok["n_tokens"],
                "n_infra": st["n_infra"],
            }
            print(f"[full87] {size}: static (from train, pooled): {results[cond]}", flush=True)
            continue
        eval_tree = rdir / f"eval_{cond}"
        eval_tree = generate_tree(
            f"eval_{cond}", cond_bundles[cond], static_bundle,
            task_names, skillset_name, gos_workspace, eval_tree, size,
        )
        cfg = make_config(job_name, task_names, eval_tree,
                          n_attempts, concurrency)
        yaml_p = rdir / f"eval_{cond}.yaml"
        write_yaml(cfg, yaml_p)
        if job_complete(Path("jobs") / job_name, len(task_names), n_attempts):
            print(f"[full87] {size}: eval {cond} already complete, skipping", flush=True)
        else:
            run_harbor(yaml_p, env_file, job_name)
        tok = attempt_token_stats(Path("jobs") / job_name)
        st = pooled_reward_stats(Path("jobs") / job_name)
        results[cond] = {
            "R": st["R"],
            "T_M": tok["T_M"],
            "n": st["n_scored"],
            "n_attempts_with_tokens": tok["n_tokens"],
            "n_infra": st["n_infra"],
        }
        print(f"[full87] {size}: {cond}: {results[cond]}", flush=True)

    (rdir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[full87] {size}: DONE -> {rdir / 'summary.json'}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Full SkillsBench experiment suite on token-sem bases.")
    ap.add_argument("--sizes", type=str, default="1000",
                    help="comma-separated library sizes (200,500,1000,2000)")
    ap.add_argument("--gos-workspace-dir", type=Path, required=True)
    ap.add_argument("--env-file", type=Path, default=Path("../../.env"))
    ap.add_argument("--out-root", type=Path, default=Path("generated/full87_evo"))
    ap.add_argument("--n-attempts", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--factorial", action="store_true",
                    help="run the full 2^3 factorial at the first (main) size; "
                         "other sizes run L1L2L3 only (scale 4-way)")
    ap.add_argument("--semantic-k", type=int, default=1,
                    help="kNN density k of the token-overlap semantic cold start "
                         "(default 1; other k values get k-prefixed job names so "
                         "a k!=8 run never collides with k=8 artifacts)")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    # full 87-task SkillsBench set (same at every library size, as in GoS)
    tasks_root = SKILLSBENCH_ROOT / "tasks"
    task_names = sorted(
        d.name for d in tasks_root.iterdir() if d.is_dir() and d.name != "__pycache__"
    ) if tasks_root.is_dir() else []
    if not task_names:
        raise SystemExit("no tasks found under benchmarks/skillsbench/tasks/")
    print(f"[full87] {len(task_names)} tasks", flush=True)

    for size in sizes:
        run_size(
            size,
            args.gos_workspace_dir,
            args.env_file,
            args.out_root,
            args.n_attempts,
            args.concurrency,
            f"skills_{size}",
            task_names,
            factorial=(args.factorial and size == sizes[0]),
            semantic_k=args.semantic_k,
        )


if __name__ == "__main__":
    main()
