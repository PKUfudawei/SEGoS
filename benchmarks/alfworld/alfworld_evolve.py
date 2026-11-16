"""Apply SE-GoS evolution deltas to the ALFWorld skill graph bundle.

Pipeline (identical to the SkillsBench SE-GoS line):
    cold-start bundle -> L1 (workflow, dependency, avoid, avoid retraction)
    -> L2 (semantic attenuation and reinforcement) -> L3 -> evolved bundle

The evolved bundle is what the ALFWorld run consumes via `--bundle`, exactly
like SkillsBench mounts a per-condition bundle into the task container. The old
GoS-engine write-back into `graph_igraph_data.pklz` is gone: ALFWorld retrieves
through `graphskills_assets/query.py`, which reads bundle JSON only.

Usage (from the repository root):
    # full SE-GoS (L1L2L3) into an evolved bundle:
    python benchmarks/alfworld/alfworld_evolve.py \
      --signals benchmarks/alfworld/generated/alfworld_signals.json \
      --bundle benchmarks/alfworld/generated/shared/alfworld_coldstart_skills200_k1_token.json \
      --out benchmarks/alfworld/generated/bundle_L1L2L3.json

    # single ablation condition:
    python benchmarks/alfworld/alfworld_evolve.py \
      --signals benchmarks/alfworld/generated/alfworld_signals.json \
      --bundle benchmarks/alfworld/generated/shared/alfworld_coldstart_skills200_k1_token.json \
      --levers L1,L2 --out benchmarks/alfworld/generated/bundle_L1L2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The L1/L2/L3 machinery is shared with the SkillsBench SE-GoS line.
SKILLSBENCH = Path(__file__).resolve().parents[1] / "skillsbench"
sys.path.insert(0, str(SKILLSBENCH))
from segos.evolution.pipeline import evolve_bundle_serial
from segos.retrieval.settings import ALFWORLD  # noqa: E402


PIPELINE_VERSION = "public-alfworld-gos-v4-budget"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="cold-start (static) bundle JSON to evolve from")
    ap.add_argument("--out", type=Path, required=True,
                    help="where to write the evolved bundle JSON")
    ap.add_argument("--levers", type=str, default="L1,L2,L3",
                    help="Comma-separated levers to apply (default: all three). "
                         "e.g. --levers L1 for topology-only, L1,L2 for the pair cell.")
    ap.add_argument("--model", default=None,
                    help="Benchmark model; required for legacy signals without model metadata.")
    args = ap.parse_args()

    requested = [s.strip() for s in args.levers.split(",") if s.strip()]
    for lev in requested:
        if lev not in {"L1", "L2", "L3"}:
            raise ValueError(f"Unknown lever: {lev!r} (expected L1|L2|L3)")
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Select each lever at most once and at least one lever")
    selected = [lev for lev in ("L1", "L2", "L3") if lev in requested]

    base_bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    print(f"[alf-evolve] base: {len(base_bundle['skills'])} skills, "
          f"{len(base_bundle['edges'])} edges", flush=True)

    signals = json.loads(args.signals.read_text())
    train_tasks = set(signals["trials"].keys())

    # The SkillsBench helper caches each delta per condition and computes it on
    # the bundle produced by preceding levers. Its L1 includes dependency and
    # avoid induction; L3 is applied as a description override, not merged into
    # an edge-weight delta. The old alfworld_delta_*.json files are incompatible
    # with this serial pipeline and are deliberately not read.
    new_bundle = evolve_bundle_serial(
        base_bundle, signals, train_tasks, "".join(selected), out_dir=args.out.parent,
        benchmark_model=args.model, retrieval=ALFWORLD,
    )
    new_bundle["segos_evolution_pipeline"] = PIPELINE_VERSION

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(new_bundle, indent=2), encoding="utf-8")
    print(f"[alf-evolve] wrote evolved bundle: {len(new_bundle['skills'])} skills, "
          f"{len(new_bundle['edges'])} edges -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
