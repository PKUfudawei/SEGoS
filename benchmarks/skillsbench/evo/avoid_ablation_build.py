"""Small-batch avoid-consumer ablation: 0 / A / B.

Isolates the AVOID_BUNDLE_EXCLUSION consumer on a fixed evolved graph
(k1 + L1L2L3) over tasks implicated by within-task mining:
  0: L1L2L3 edges with every avoid edge removed (no avoid relation)
  A: L1L2L3 + injected mined avoid edges (all weight 0 = PPR-inert), consumer OFF
  B: identical edge set to A, consumer ON  (only variable = bundle exclusion)
A-vs-0 checks injected inert edges change nothing; B-vs-A isolates the consumer.
"""
from __future__ import annotations

import json
import shutil
import sys
import yaml
from pathlib import Path

SKILLSBENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILLSBENCH_ROOT))

# The k=1 graph that carries the avoid edges, i.e. the released round-1 graph
# (1,118 edges: 863 semantic, 228 workflow, 27 avoid). The local
# generated/full87_evo_k1/1000/bundle_L1L2L3.json is the pre-regeneration graph
# and holds no avoid edge at all, so variant "0" would have nothing to strip.
BUNDLE = SKILLSBENCH_ROOT / "generated/evolved/segos_evolved_skills1000_round1.json"
# Graph-skills task tree for the k=1 L1L2L3 cell, produced by
# graphskills_benchmark.py (see benchmarks/skillsbench/README.md, "Generating
# Task Variants"). It is not in the dataset; the generator writes it.
SRC_TREE = SKILLSBENCH_ROOT / "generated/full87_evo_k1/1000/eval_L1L2L3/tasks_graph_skills"
QUERY_PY = SKILLSBENCH_ROOT / "graphskills_assets/query.py"
OUT = SKILLSBENCH_ROOT / "generated"
TASKS = [
    "edit-pdf", "latex-formula-extraction", "invoice-fraud-detection",
    "paper-anonymizer", "jpg-ocr-stat", "hvac-control",
    "civ6-adjacency-optimizer", "lake-warming-attribution",
    "dapt-intrusion-detection", "pdf-excel-diff",
]

# candidate pairs mined by mine_within_task_avoid.py (harm in >=1 task, no
# contradicting benefit); each becomes a weight-0 avoid edge.
#
# Refreshed against the k=1 substrate: `PYTHONPATH=$PWD python -m
# evo.mine_within_task_avoid`, which pools the k=1 static train run and the k=1
# fused L1L2L3 run. The previous list had been mined over the k=8 lineage and
# only two of its twenty pairs reproduce here, so it injected k=8 evidence into
# a k=1 graph.
MINED_PAIRS = [
    ("TTS Audio Mastering", "report-generator"),
    ("TTS Audio Mastering", "video-processor"),
    ("ac-branch-pi-model", "casadi-ipopt-nlp"),
    ("audio-extractor", "report-generator"),
    ("audio-extractor", "video-processor"),
    ("casadi-ipopt-nlp", "power-flow-data"),
    ("civ6lib", "map-optimization-strategy"),
    ("excitation-signal-design", "integral-action-design"),
    ("excitation-signal-design", "mpc-horizon-tuning"),
    ("ffmpeg-keyframe-extraction", "gemini-count-in-video"),
    ("ffmpeg-keyframe-extraction", "openai-vision"),
    ("fuzzy-match", "pdf"),
    ("gemini-count-in-video", "sampling-and-indexing"),
    ("gemini-count-in-video", "video-frame-extraction"),
    ("hex-grid-spatial", "map-optimization-strategy"),
    ("image-ocr", "xlsx"),
    ("openai-vision", "video-frame-extraction"),
    ("report-generator", "video-processor"),
    ("senior-security", "trivy-offline-vulnerability-scanning"),
    ("senior-security", "vulnerability-csv-reporting"),
]


def build_bundles() -> dict[str, Path]:
    bundle = json.loads(BUNDLE.read_text())
    names = {s["name"] for s in bundle["skills"]}
    avoid_edges = []
    for u, v in MINED_PAIRS:
        if u not in names or v not in names:
            print(f"  skip missing skill pair: {u} <-> {v}")
            continue
        avoid_edges.append({
            "source": u, "target": v, "type": "avoid", "weight": 0.0,
            "confidence": 1.0,
            "description": f"Co-loading {u} and {v} is avoided (within-task failure evidence).",
        })
    base_edges = [e for e in bundle["edges"] if e.get("type") != "avoid"]
    print(f"base non-avoid edges: {len(base_edges)}; injected avoid edges: {len(avoid_edges)}")
    bundles = {
        "0": dict(bundle, edges=list(base_edges)),
        "A": dict(bundle, edges=list(base_edges) + [dict(e) for e in avoid_edges]),
        "B": dict(bundle, edges=list(base_edges) + [dict(e) for e in avoid_edges]),
    }
    paths = {}
    for tag, b in bundles.items():
        d = OUT / f"avoid{tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "bundle_L1L2L3.json"
        p.write_text(json.dumps(b, ensure_ascii=False))
        paths[tag] = p
        print(f"bundle {tag}: {len(b['edges'])} edges -> {p}")
    return paths


def build_tree(tag: str, consumer_on: bool) -> None:
    dst = OUT / f"avoid{tag}" / "tasks_graph_skills"
    if dst.exists():
        shutil.rmtree(dst)
    for task in TASKS:
        src = SRC_TREE / task
        dd = dst / task
        shutil.copytree(src, dd)
        env = dd / "environment"
        shutil.copyfile(QUERY_PY, env / "graphskills" / "query.py")
        if consumer_on:
            compose = env / "docker-compose.yaml"
            text = compose.read_text()
            if "AVOID_BUNDLE_EXCLUSION" not in text:
                text = text.replace("    environment:",
                                    "    environment:\n      - AVOID_BUNDLE_EXCLUSION=1", 1)
                compose.write_text(text, encoding="utf-8")
    print(f"tree {tag} ready ({len(TASKS)} tasks, consumer={'on' if consumer_on else 'off'})")


def make_yaml(tag: str) -> Path:
    from benchmarks.skillsbench.evo import run_full87_experiments as R
    tree = OUT / f"avoid{tag}"
    cfg = R.make_config(f"graphskills-avoid-ablation-{tag}", TASKS, tree, "graphskills", 2, 8)
    p = OUT / f"avoid_{tag}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"yaml {tag} -> {p}")
    return p


if __name__ == "__main__":
    build_bundles()
    build_tree("0", consumer_on=False)
    build_tree("A", consumer_on=False)
    build_tree("B", consumer_on=True)
    for tag in ("0", "A", "B"):
        make_yaml(tag)
    print("done")
