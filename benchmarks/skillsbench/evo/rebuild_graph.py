"""Rebuild the GoS static bundle from the official HF prebuilt workspace.

Uses the official normalized skill nodes (with I/O schema) so that:
  - dependency edges can be induced deterministically from I/O overlap (>= 0.6),
  - semantic edges are sparse top-k nearest neighbours from the official
    3072-dim HNSW embedding index (index id == igraph vertex order),
  - "is" equivalence edges from the official igraph are preserved.

The released data contains no wf/alt edges: those require an LLM validation
pass that is not available in this environment (see paper / memory).

Output bundle JSON has the same shape as the existing
generated/shared/graphskills_bundle.json (skills + edges).

Usage (from benchmarks/skillsbench):
    PYTHONPATH=$PWD python -m evo.rebuild_graph \
        --igraph data/.../graph_igraph_data.pklz \
        --hnsw data/.../entities_hnsw_index_3072.bin \
        --semantic-k 8 \
        --out generated/shared/graphskills_bundle_rebuilt.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import hnswlib

from segos.retrieval.query import signature_tokens  # noqa: E402
from segos.graph import schema_overlap_score

DEPENDENCY_THRESHOLD = 0.6
SEMANTIC_WEIGHT = 0.6
MIN_TOKEN_OVERLAP = 2  # require >= this many shared signature tokens for a sem edge


def _split_multi(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split("\n") if part.strip()]


def _edge_description(edge_type: str, source: str, target: str,
                      evidence: list[str] | None = None) -> str:
    if edge_type == "dependency":
        return (f"{source} produces data that {target} consumes: "
                f"{', '.join(evidence) or 'compatible I/O'}.")
    if edge_type == "semantic":
        return (f"{source} and {target} are top-k nearest neighbours in the "
                f"official embedding space.")
    return f"{source} is {target}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--igraph", type=Path, required=True)
    ap.add_argument("--hnsw", type=Path, required=False, default=None,
                    help="official 3072-dim HNSW index; required only for --sem-metric embedding")
    ap.add_argument("--semantic-k", type=int, default=8)
    ap.add_argument("--sem-metric", type=str, default="embedding",
                    choices=["embedding", "token"],
                    help="semantic-neighbour metric: official-embedding kNN, or "
                         "token-overlap (signature-token Jaccard) kNN -- no model, "
                         "recomputable after description edits")
    ap.add_argument("--edge-types", type=str, default="dep,sem,is",
                    help="comma-separated subset of {dep,sem,is} to build; "
                         "e.g. 'sem' produces a semantic-only cold-start base")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    requested = {part.strip() for part in args.edge_types.split(",") if part.strip()}
    do_dep = "dep" in requested
    do_sem = "sem" in requested
    do_is = "is" in requested

    with gzip.open(args.igraph, "rb") as f:
        graph = pickle.load(f)

    # Provenance comes from the workspace layout (<gos_workspace>/skills_<N>_v1/…)
    # rather than a hardcoded 1,000-skill string, so a rebuild for another library
    # size records where its nodes actually came from.
    workspace_dir = f"{args.igraph.parent.parent.name}_{args.igraph.parent.name}"
    skillset_name = re.sub(r"_v\d+$", "", args.igraph.parent.name)

    n = graph.vcount()
    names = list(graph.vs["name"])
    descriptions = list(graph.vs["description"])
    one_line = list(graph.vs["one_line_capability"])
    inputs = [_split_multi(v) for v in graph.vs["inputs"]]
    outputs = [_split_multi(v) for v in graph.vs["outputs"]]
    rendered = list(graph.vs["rendered_snippet"])
    domain = [_split_multi(v) for v in graph.vs["domain_tags"]]
    tooling = [_split_multi(v) for v in graph.vs["tooling"]]
    example_tasks = [_split_multi(v) for v in graph.vs["example_tasks"]]
    entrypoints = [_split_multi(v) for v in graph.vs["script_entrypoints"]]
    raw = list(graph.vs["raw_content"])
    skill_ids = list(graph.vs["skill_id"])

    print(f"[rebuild] {n} official nodes loaded", flush=True)

    skills: list[dict] = []
    for i in range(n):
        # source_path must match write_vector_store's key format:
        #   {GRAPH_LIBRARY_PATH}/{relative_to(skills_root)}/SKILL.md
        # The directory slug must come from the official skill_id (normalized
        # lowercase dir name), NOT from the display `name` (61 names contain
        # spaces, e.g. "HeyGen Automation" vs dir "heygen-automation").
        skill_id_dir = str(skill_ids[i]).rsplit("/", 2)[-2] if skill_ids[i] else names[i]
        skills.append({
            "name": names[i],
            "description": descriptions[i],
            "one_line_capability": one_line[i],
            "inputs": inputs[i],
            "outputs": outputs[i],
            "compatibility": "",
            "allowed_tools": [],
            "domain_tags": domain[i],
            "tooling": tooling[i],
            "example_tasks": example_tasks[i],
            "script_entrypoints": entrypoints[i],
            "source_path": f"/opt/graphskills/skills/{skill_id_dir}/SKILL.md",
            # skill_dir must be the bare dir slug: signals.py matches
            # /opt/graphskills/skills/<dir>/ in trajectories and looks up
            # dir_to_name[skill_dir] with exactly this value.
            "skill_dir": skill_id_dir,
            "rendered_snippet": rendered[i],
            "raw_content": raw[i],
            "skill_id": skill_ids[i],
        })

    # 1) dependency edges: deterministic I/O overlap >= threshold
    edges: list[dict] = []
    dep_keys: set[tuple[str, str, str]] = set()
    n_dep = 0
    if do_dep:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                fwd, fwd_ev = schema_overlap_score(outputs[i], inputs[j])
                if fwd >= DEPENDENCY_THRESHOLD:
                    key = (names[i], names[j], "dependency")
                    if key not in dep_keys:
                        dep_keys.add(key)
                        edges.append({
                            "source": names[i], "target": names[j], "type": "dependency",
                            "weight": round(float(fwd), 4),
                            "confidence": round(float(fwd), 4),
                            "description": _edge_description("dependency", names[i], names[j], fwd_ev),
                        })
                        n_dep += 1
                rev, rev_ev = schema_overlap_score(outputs[j], inputs[i])
                if rev >= DEPENDENCY_THRESHOLD:
                    key = (names[j], names[i], "dependency")
                    if key not in dep_keys:
                        dep_keys.add(key)
                        edges.append({
                            "source": names[j], "target": names[i], "type": "dependency",
                            "weight": round(float(rev), 4),
                            "confidence": round(float(rev), 4),
                            "description": _edge_description("dependency", names[j], names[i], rev_ev),
                        })
                        n_dep += 1
    print(f"[rebuild] dependency edges: {n_dep}", flush=True)

    # 2) semantic edges: top-k nearest neighbours, embedding or token-overlap
    sem_keys: set[tuple[str, str, str]] = set()
    n_sem = 0
    if do_sem and args.sem_metric == "embedding":
        if args.hnsw is None:
            raise SystemExit("--sem-metric embedding requires --hnsw")
        idx = hnswlib.Index(space="cosine", dim=3072)
        idx.load_index(str(args.hnsw))
        # batch knn over all items
        vecs = idx.get_items(list(range(n)))
        labels, dists = idx.knn_query(vecs, k=args.semantic_k + 1)
        for i in range(n):
            for rank, j in enumerate(labels[i][1:], start=1):  # skip self
                if int(j) >= n:
                    continue
                key = (names[i], names[int(j)], "semantic")
                if key not in sem_keys and (names[int(j)], names[i], "semantic") not in sem_keys:
                    sem_keys.add(key)
                    edges.append({
                        "source": names[i], "target": names[int(j)], "type": "semantic",
                        "weight": SEMANTIC_WEIGHT, "confidence": SEMANTIC_WEIGHT,
                        "description": _edge_description("semantic", names[i], names[int(j)]),
                    })
                    n_sem += 1
        print(f"[rebuild] semantic edges: {n_sem} (embedding kNN k={args.semantic_k}, dedup)", flush=True)
    elif do_sem:  # token-overlap sem: no model, recomputable after description edits
        node_tokens = [
            signature_tokens(
                [
                    names[i],
                    descriptions[i],
                    "\n".join(inputs[i]),
                    "\n".join(outputs[i]),
                    rendered[i],
                ]
            )
            for i in range(n)
        ]
        for i in range(n):
            scored = []
            for j in range(n):
                if i == j:
                    continue
                overlap = node_tokens[i] & node_tokens[j]
                if len(overlap) < MIN_TOKEN_OVERLAP:
                    continue
                union = node_tokens[i] | node_tokens[j]
                scored.append((len(overlap) / max(len(union), 1), j))
            scored.sort(key=lambda item: item[0], reverse=True)
            for score, j in scored[: args.semantic_k]:
                key = (names[i], names[j], "semantic")
                if key not in sem_keys and (names[j], names[i], "semantic") not in sem_keys:
                    sem_keys.add(key)
                    edges.append({
                        "source": names[i], "target": names[j], "type": "semantic",
                        "weight": SEMANTIC_WEIGHT, "confidence": SEMANTIC_WEIGHT,
                        "description": (
                            f"{names[i]} and {names[j]} share "
                            f"{len(node_tokens[i] & node_tokens[j])} signature tokens "
                            f"(Jaccard {score:.3f})."
                        ),
                    })
                    n_sem += 1
        print(f"[rebuild] semantic edges: {n_sem} (token-overlap kNN k={args.semantic_k}, dedup)", flush=True)

    # 3) "is" equivalence edges from official igraph
    is_keys: set[tuple[str, str, str]] = set()
    n_is = 0
    if do_is:
        for e in graph.es:
            if (e["description"] or "").strip().lower() == "is":
                s = names[e.source]
                t = names[e.target]
                key = (s, t, "is")
                if key not in is_keys:
                    is_keys.add(key)
                    edges.append({
                        "source": s, "target": t, "type": "is",
                        "weight": 1.0, "confidence": 1.0,
                        "description": _edge_description("is", s, t),
                    })
                    n_is += 1
    print(f"[rebuild] is edges: {n_is}", flush=True)

    bundle = {
        "metadata": {
            "skillset_name": skillset_name,
            "source": f"official HF prebuilt workspace ({workspace_dir})",
            "dep_edges": n_dep,
            "semantic_edges": n_sem,
            "semantic_k": args.semantic_k,
            "sem_metric": args.sem_metric,
            "is_edges": n_is,
            "total_edges": len(edges),
        },
        "skills": skills,
        "edges": edges,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    print(f"[rebuild] wrote {args.out} ({len(edges)} edges)", flush=True)


if __name__ == "__main__":
    main()
