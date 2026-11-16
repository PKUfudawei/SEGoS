"""Minimal probe for #2 (SkillDAG-style three-channel search) on our harness.

Treatment B: the agent queries with GOS_CHANNEL_SEARCH=1 (matches / typed
neighbors / conflicts) against the L1L2L3 graph.  Control A is the *existing*
eval-L1L2L3 fused-bundle run on the same tasks and the same graph, so the only
difference is the query interface + instructions, not the graph.

Offline prefilter: only tasks whose seed matches have >=1 workflow/dependency
neighbor are selected -- a task with no typed neighbors cannot exercise the
neighbor channel, so including it would only dilute the probe.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SKILLSBENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILLSBENCH_ROOT))
sys.path.insert(0, str(SKILLSBENCH_ROOT / "graphskills_assets"))

from benchmarks.skillsbench.evo import analyze as A  # noqa: E402
from benchmarks.skillsbench.evo import signals as S  # noqa: E402
import query as q  # noqa: E402

JOB_A = SKILLSBENCH_ROOT / "jobs" / "graphskills-full87-1000-eval-L1L2L3"
SRC_TREE = SKILLSBENCH_ROOT / "generated/full87_evo/1000/eval_L1L2L3/tasks_graph_skills"
BUNDLE_L1L2L3 = SKILLSBENCH_ROOT / "generated/full87_evo/1000/bundle_L1L2L3.json"
OUT_ROOT = SKILLSBENCH_ROOT / "generated/channels_probe"
QUERY_PY = SKILLSBENCH_ROOT / "graphskills_assets/query.py"

CHANNEL_AGENTS_ADD = """

## Retrieval channels (this environment)

Query with the three-channel interface (do NOT use plain `graphskills-query`
without the prefix):

    GOS_CHANNEL_SEARCH=1 graphskills-query "<targeted query>"

The output is split into three evidence channels:
- **Matches**: skills whose text directly matches your query -- your primary leads.
- **Neighbors**: skills reachable through typed prerequisite/workflow edges from
  a match. A neighbor is not a text match; load it only if its stated
  relationship to a match suggests the task needs it (e.g. a prerequisite,
  parsing/format utility, or a later pipeline step).
- **Conflicts**: pairs that must NOT be co-loaded (none present for most queries).

Read full skill bodies only after deciding to use a skill, via its exact
`Source:` path. Re-run the query with refined wording when matches miss the mark.
"""


def task_typed_neighbor_stats(bundle: dict, task_name: str) -> tuple[int, int]:
    """(seed-matches, typed-neighbors-reachable) for the task's train queries."""
    sig_path = SKILLSBENCH_ROOT / "generated/full87_evo/1000/signals.json"
    sig = json.loads(sig_path.read_text())
    trial = sig["trials"].get(task_name)
    if not trial or not trial.get("queries"):
        return 0, 0
    n_match_hits = 0
    n_with_neighbors = 0
    for query in trial["queries"]:
        res = q.retrieve(
            bundle, query,
            top_n=5, seed_top_k=4, max_skill_chars=1800, max_context_chars=9000,
            seed_mode="lexical", propagation_mode="ppr",
        )
        seed_names = {s["name"] for s in res["seeds"]}
        shown_matches = [s for s in res["skills"] if s["name"] in seed_names]
        if not shown_matches:
            continue
        n_match_hits += 1
        nbs = q._reachable_neighbor_names(bundle["edges"], {s["name"] for s in shown_matches}, depth=2, max_neighbors=8)
        if nbs:
            n_with_neighbors += 1
    return n_match_hits, n_with_neighbors


def pick_subset(bundle: dict, n_target: int = 15) -> list[str]:
    # tasks with exactly two valid attempts in the A control job
    valid: dict[str, int] = {}
    for tdir in JOB_A.iterdir():
        if not tdir.is_dir() or "__" not in tdir.name:
            continue
        rp = tdir / "result.json"
        if not rp.exists():
            continue
        d = json.loads(rp.read_text())
        rew = (d.get("verifier_result") or {}).get("rewards", {}).get("reward")
        if rew is None:
            continue
        task = d.get("task_name") or tdir.name.split("__")[0]
        valid[task] = valid.get(task, 0) + 1
    candidates = [t for t, n in valid.items() if n >= 2]
    scored: list[tuple[int, int, str]] = []
    for task in sorted(candidates):
        n_hits, n_nb = task_typed_neighbor_stats(bundle, task)
        if n_hits and n_nb >= max(1, n_hits // 2):
            scored.append((n_nb, -n_hits, task))
    scored.sort(reverse=True)
    return [task for _, _, task in scored[:n_target]]


def build_b_tree(tasks: list[str]) -> Path:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for task in tasks:
        src = SRC_TREE / task
        dst = OUT_ROOT / "tasks_graph_skills" / task
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copytree(src, dst)
        env = dst / "environment"
        # 1) fresh query.py (with channel support) baked into the image
        shutil.copyfile(QUERY_PY, env / "graphskills" / "query.py")
        # 2) AGENTS.md / CLAUDE.md / GEMINI.md get the channel instructions
        for md in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
            p = env / md
            if p.exists():
                p.write_text(p.read_text() + CHANNEL_AGENTS_ADD, encoding="utf-8")
        # 3) docker-compose sets GOS_CHANNEL_SEARCH=1 inside the container
        compose = env / "docker-compose.yaml"
        text = compose.read_text()
        if "GOS_CHANNEL_SEARCH" not in text:
            text = text.replace("    environment:", "    environment:\n      - GOS_CHANNEL_SEARCH=1", 1)
            compose.write_text(text, encoding="utf-8")
    return OUT_ROOT


def main() -> None:
    bundle = S.load_bundle(BUNDLE_L1L2L3)
    tasks = pick_subset(bundle)
    print(f"[probe] selected {len(tasks)} tasks with typed-neighbor coverage:")
    for t in tasks:
        nh, nn = task_typed_neighbor_stats(bundle, t)
        print(f"   {t}: hit_queries={nh} queries_with_typed_neighbors={nn}")
    build_b_tree(tasks)
    print(f"[probe] B tree written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
