"""Commands for inspecting, evolving, preparing and evaluating skill graphs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def prepare(args):
    from benchmarks.skillsbench import graphskills_benchmark as generator
    from benchmarks.skillsbench.evo.configs import build_config
    import yaml

    tasks_root = args.tasks_root.resolve()
    graph = args.graph.resolve()
    skills = args.skills_root.resolve()
    output = args.output.resolve()
    tasks = args.task or sorted(p.name for p in tasks_root.iterdir() if p.is_dir() and (p / "instruction.md").is_file())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Select a nonempty set of distinct tasks")
    if not graph.is_file() or not skills.is_dir():
        raise ValueError("Provide an existing graph and unpacked skill library")
    if output.exists():
        raise ValueError("Output already exists; use a new run directory")
    command = [sys.executable, "-m", "benchmarks.skillsbench.graphskills_benchmark",
               "--tasks-root", str(tasks_root), "--skills-root", str(skills),
               "--output-root", str(output), "--bundle-path", str(graph),
               "--bundle-mount", str(graph), "--skip-allskills", "--skip-vectorskills"]
    for task in tasks:
        if not (tasks_root / task / "instruction.md").is_file():
            raise ValueError(f"Unknown task: {task}")
        command.extend(["--task", task])
    subprocess.run(command, check=True)
    config = build_config(output.name, tasks, output.parent)
    config["job_name"] = output.name
    config["jobs_dir"] = str(output / "jobs")
    config["datasets"] = [{"path": str(output / "tasks_graph_skills"), "task_names": tasks}]
    config["agents"][0]["model_name"] = args.model
    config["n_attempts"] = args.attempts
    config["orchestrator"]["n_concurrent_trials"] = args.concurrency
    (output / "run.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    from segos.retrieval import query
    write_json(output / "run_manifest.json", {
        "schema_version": 1, "benchmark": "skillsbench", "model": args.model,
        "tasks": tasks, "attempts_per_task": args.attempts, "graph_sha256": digest(graph),
        "retriever_sha256": digest(query.__file__), "generator_sha256": digest(generator.__file__),
        "task_inputs": {task: {str(p.relative_to(tasks_root / task)): digest(p)
                               for p in sorted((tasks_root / task).rglob("*")) if p.is_file()}
                        for task in tasks},
    })
    print(output / "run.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate and summarize a graph")
    inspect.add_argument("graph", type=Path)
    query = sub.add_parser("query", help="Retrieve a skill bundle")
    query.add_argument("graph", type=Path)
    query.add_argument("text")
    query.add_argument("--benchmark", choices=["skillsbench", "alfworld"], default="skillsbench")
    evo = sub.add_parser("evolve", help="Apply condition-specific offline graph updates")
    evo.add_argument("--graph", type=Path, required=True)
    evo.add_argument("--signals", type=Path, required=True)
    evo.add_argument("--split", type=Path, required=True)
    evo.add_argument("--condition", choices=["static", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3", "L1L2L3"], default="L1L2L3")
    evo.add_argument("--benchmark", choices=["skillsbench", "alfworld"], default="skillsbench")
    evo.add_argument("--model")
    evo.add_argument("--output", type=Path, required=True)
    signals = sub.add_parser("signals", help="Extract SkillsBench evidence from existing jobs")
    signals.add_argument("--jobs", type=Path, nargs="+", required=True)
    signals.add_argument("--graph", type=Path, required=True)
    signals.add_argument("--selection", type=Path)
    signals.add_argument("--attempt-policy", choices=["all_attempts", "selected_attempt"],
                         default="all_attempts",
                         help="all_attempts for new runs; selected_attempt for historical reproduction")
    signals.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="Prepare a fixed-graph SkillsBench evaluation")
    prep.add_argument("--tasks-root", type=Path, required=True)
    prep.add_argument("--skills-root", type=Path, required=True)
    prep.add_argument("--graph", type=Path, required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--task", action="append")
    prep.add_argument("--attempts", type=int, default=2)
    prep.add_argument("--concurrency", type=int, default=4)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Execute an explicitly prepared Harbor configuration")
    run.add_argument("config", type=Path)
    data = sub.add_parser("unpack", help="Verify and unpack a local archive")
    data.add_argument("archive", type=Path)
    data.add_argument("--sha256", required=True)
    data.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from segos.retrieval.settings import ALFWORLD, SKILLSBENCH
    if args.command in {"inspect", "query"}:
        bundle = json.loads(args.graph.read_text())
        names = [s["name"] for s in bundle["skills"]]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate skill names")
        if any(e["source"] not in names or e["target"] not in names for e in bundle["edges"]):
            raise ValueError("Edge endpoint missing from node set")
        if args.command == "inspect":
            print(json.dumps({"nodes": len(names), "edges": len(bundle["edges"]),
                              "relations": Counter(e["type"] for e in bundle["edges"]),
                              "sha256": digest(args.graph)}, indent=2))
        else:
            from segos.retrieval.query import retrieve
            options = ALFWORLD if args.benchmark == "alfworld" else SKILLSBENCH
            print(json.dumps(retrieve(bundle, args.text, **options, seed_mode="lexical",
                                     propagation_mode="ppr", vector_store_path=None), indent=2))
    elif args.command == "evolve":
        from segos.evolution.pipeline import evolve_bundle_serial
        split = json.loads(args.split.read_text())
        train = set(split["train"])
        if train & set(split.get("eval", [])):
            raise ValueError("Training and evaluation tasks overlap")
        if args.output.exists():
            raise ValueError("Output graph already exists; select a new graph version")
        result = evolve_bundle_serial(json.loads(args.graph.read_text()), json.loads(args.signals.read_text()),
                                      train, args.condition, args.output.parent / "deltas",
                                      benchmark_model=args.model,
                                      retrieval=ALFWORLD if args.benchmark == "alfworld" else SKILLSBENCH)
        write_json(args.output, result)
    elif args.command == "signals":
        from benchmarks.skillsbench.evo.signals import parse_jobs
        selection = json.loads(args.selection.read_text()) if args.selection else None
        parse_jobs(args.jobs, args.graph, args.output, merge_attempts=False, selection=selection,
                   attempt_policy=args.attempt_policy)
    elif args.command == "prepare":
        if args.attempts < 1 or args.concurrency < 1:
            raise ValueError("Attempts and concurrency must be positive")
        prepare(args)
    elif args.command == "run":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY before running an evaluation")
        subprocess.run(["harbor", "run", "-c", str(args.config.resolve())], check=True)
    elif args.command == "unpack":
        if digest(args.archive) != args.sha256:
            raise ValueError("Archive checksum does not match")
        if args.output.exists():
            raise ValueError("Extraction directory already exists")
        args.output.mkdir(parents=True)
        with tarfile.open(args.archive) as archive:
            archive.extractall(args.output, filter="data")


if __name__ == "__main__":
    main()
