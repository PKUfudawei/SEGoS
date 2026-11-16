"""Run separate ALFWorld training and two-attempt frozen-graph evaluations."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_resume(manifest, identity):
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Run inputs changed; use a new output directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--skills-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--factorial", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    from segos.retrieval.settings import ALFWORLD
    from segos.evolution import pipeline, updates
    from segos.retrieval import query
    identity = {
        "model": args.model, "graph_sha256": sha(args.graph), "retrieval": ALFWORLD,
        "skills": {str(p.relative_to(args.skills_root)): sha(p)
                   for p in sorted(args.skills_root.rglob("*")) if p.is_file()},
        "implementation": {Path(p).name: sha(p) for p in
                           [__file__, pipeline.__file__, updates.__file__, query.__file__]},
    }
    if not identity["skills"]:
        raise ValueError("Provide an unpacked skill library")
    manifest = root / "pipeline_manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest.exists():
        raise ValueError("Existing run has no input manifest; use a new output directory")
    check_resume(manifest, identity)
    root.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity, indent=2) + "\n")

    def evaluate(split, condition, graph, attempt, mode="graphskills"):
        command = [sys.executable, "-m", "benchmarks.alfworld.alfworld_run",
                   "--model", args.model, "--split", split, "--mode", mode,
                   "--use_skill", "--skills_dir", str(args.skills_root.resolve()),
                   "--bundle", str(graph.resolve()),
                   "--output-dir", str(root / "jobs"), "--exp_name", condition,
                   "--attempt-id", str(attempt), "--max_workers", str(args.workers),
                   "--max_games", "420" if split == "train" else "140"]
        if split == "train":
            command.extend(["--shuffle_seed", "2026"])
        subprocess.run(command, check=True)

    evaluate("train", "static", args.graph, 1)
    train_dir = root / "jobs" / args.model / "train_static_mode_graphskills" / "attempt_1"
    if len(list(train_dir.glob("idx_*.json"))) != 420:
        raise RuntimeError("Expected 420 completed training episodes")
    signals = root / "signals.json"
    subprocess.run([sys.executable, "-m", "benchmarks.alfworld.alfworld_signals",
                    "--results-dir", str(train_dir), "--out", str(signals)], check=True)
    trials = json.loads(signals.read_text())["trials"]
    if len(trials) != 420:
        raise RuntimeError("Training episode identifiers are not unique")
    split = root / "train_split.json"
    split.write_text(json.dumps({"train": sorted(trials), "eval": []}, indent=2))
    conditions = ["L1L2L3", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3"] if args.factorial else ["L1L2L3"]
    graphs = {"static": args.graph}
    for condition in conditions:
        graph = root / condition / "graph.json"
        graph_manifest = graph.with_suffix(".inputs.json")
        inputs = {"signals_sha256": sha(signals), "split_sha256": sha(split),
                  "condition": condition}
        if graph.exists():
            if not graph_manifest.exists():
                raise ValueError(f"No evolution manifest for {graph}; use a new output directory")
            check_resume(graph_manifest, {**inputs, "graph_sha256": sha(graph)})
        if not graph.exists():
            subprocess.run([sys.executable, "-m", "segos", "evolve", "--benchmark", "alfworld",
                            "--graph", str(args.graph.resolve()), "--signals", str(signals),
                            "--split", str(split), "--model", args.model,
                            "--condition", condition, "--output", str(graph)], check=True)
            graph_manifest.write_text(json.dumps({**inputs, "graph_sha256": sha(graph)}, indent=2) + "\n")
        graphs[condition] = graph
    for condition, graph in graphs.items():
        for attempt in (1, 2):
            evaluate("dev", condition, graph, attempt)
            results = root / "jobs" / args.model / f"dev_{condition}_mode_graphskills" / f"attempt_{attempt}"
            if len(list(results.glob("idx_*.json"))) != 140:
                raise RuntimeError(f"Incomplete evaluation: {condition}, attempt {attempt}")


if __name__ == "__main__":
    main()
