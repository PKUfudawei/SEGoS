"""Condition-specific L1 → L2 → L3 graph evolution."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from segos.evolution import updates as levers
from segos.clients.chat import resolve_rewriter_model
from segos.retrieval.settings import use_settings, options as retrieval_options
LEVER_ORDER = ("L1", "L2", "L3")
DELTA_CACHE_VERSION = "public-serial-v1"
def _evolve_bundle_serial(
    base_bundle: dict,
    sig: dict,
    task_set: set[str],
    cond: str,
    out_dir: Path | None = None,
    *,
    benchmark_model: str | None = None,
) -> dict:
    """Evolve a condition bundle serially in lever order.

    Each lever present in ``cond`` is computed on the graph that results from
    the levers earlier in LEVER_ORDER, then applied, so L2 sees L1's new
    workflow edges and L3's miss detection runs on the fully evolved graph.
    Returns the final bundle dict; per-lever delta JSON files are written under
    ``out_dir`` when given.
    """
    from segos.graph import apply_bundle_overrides

    if cond not in {"static", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3", "L1L2L3"}:
        raise ValueError(f"Unknown evolution condition: {cond}")
    bundle = json.loads(json.dumps(base_bundle))
    rewriter_model = resolve_rewriter_model(sig, benchmark_model) if "L3" in cond else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for lever in LEVER_ORDER:
        if lever not in cond:
            continue
        # Reuse a previously written per-lever delta when present.  This is what
        # makes resuming a partially-run eval job sound: L3 deltas come from LLM
        # calls (temperature > 0), so recomputing them would silently swap in a
        # different graph for the trials that already ran.  Stale caches must be
        # cleared whenever the lever computation itself changes.
        cache_name = f"delta_{DELTA_CACHE_VERSION}_{cond}_{lever}"
        if lever != "L3":
            inputs = {"bundle": bundle, "signals": sig, "tasks": sorted(task_set),
                      "avoid_policy": levers.AVOID_POLICY_VERSION,
                      "avoid_fail_min": levers.AVOID_FAIL_MIN,
                      "avoid_success_max": levers.AVOID_SUCC_MAX,
                      "avoid_retract_min": levers.AVOID_RETRACT_SUCC_MIN,
                      "success_reward": levers.SUCCESS_REWARD,
                      "positive_relations": sorted(levers.POSITIVE_RELATIONS),
                      "new_edge_weight": levers.NEW_EDGE_WEIGHT,
                      "dependency_threshold": levers.DEPENDENCY_THRESHOLD,
                      "eta": levers.ETA,
                      "prune_min": levers.PRUNE_RETRIEVED_MIN,
                      "prune_scale": levers.PRUNE_SCALE,
                      "retrieval": levers.RETRIEVE_KWARGS | retrieval_options()}
            fingerprint = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            cache_name += f"_{fingerprint}"
        if lever == "L3":
            # Legacy description caches do not identify their model. Key new
            # caches by model AND actual inputs so another backbone's rewrites
            # cannot be silently used when resuming a run.
            inputs = {
                "model": rewriter_model, "bundle": bundle,
                "trials": sig.get("trials", {}), "tasks": sorted(task_set),
                "miss_source": levers.L3_MISS_SOURCE,
                "rank_threshold": levers.L3_RANK_THRESHOLD,
                "no_eviction_guard": levers.L3_NO_EVICTION_GUARD,
                "guard_top_k": levers.L3_GUARD_TOP_K,
                "success_reward": levers.SUCCESS_REWARD,
                "max_added_tokens": levers.L3_MAX_ADDED_TOKENS,
                "retrieval": levers.RETRIEVE_KWARGS | retrieval_options(),
            }
            fingerprint = hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            cache_name += f"_{fingerprint}"
        cache_p = None if out_dir is None else out_dir / f"{cache_name}.json"
        if cache_p is not None and cache_p.exists():
            cached = json.loads(cache_p.read_text(encoding="utf-8"))
            cache_compatible = not (
                (lever == "L1" and "remove" not in cached)
                or (lever == "L2" and "scale" not in cached)
            )
            if not cache_compatible:
                print(f"[full87] {cond}: ignoring pre-topology/L2 cache "
                      f"({cache_p.name})", flush=True)
                cached = None
            if cached is not None and lever == "L1":
                bundle = apply_bundle_overrides(
                    bundle,
                    {"remove": cached.get("remove", []), "add": cached.get("add", [])},
                    None,
                )
            elif cached is not None and lever == "L2":
                bundle = apply_bundle_overrides(
                    bundle, {"scale": cached.get("scale", {}),
                             "delta": cached.get("delta", {})}, None
                )
            elif cached is not None and lever == "L3":
                bundle = apply_bundle_overrides(bundle, None, cached or None)
            if cached is not None:
                print(f"[full87] {cond}: reused cached {lever} delta "
                      f"({cache_p.name})", flush=True)
                continue
        if lever == "L1":
            l1 = levers.compute_l1_delta(bundle, sig, task_set)
            constructive = l1.get("add", []) + levers.compute_dep_edges(bundle, sig, task_set)
            bundle = apply_bundle_overrides(
                bundle, {"remove": l1.get("remove", []), "add": constructive}, None
            )
            # Induce avoid *after* this lever's workflow and dependency edges
            # are in the graph, or the non-contradiction invariant cannot see
            # them and an avoid edge lands on a pair this same lever just
            # marked constructive.
            avoid = levers.compute_avoid_edges(sig, task_set, bundle)
            l1["add"] = constructive + avoid
            if avoid:
                bundle = apply_bundle_overrides(bundle, {"add": avoid}, None)
            if out_dir is not None:
                (out_dir / f"delta_{cond}_{lever}.json").write_text(
                    json.dumps(l1, indent=2), encoding="utf-8")
                assert cache_p is not None
                cache_p.write_text(json.dumps(l1, indent=2), encoding="utf-8")
        elif lever == "L2":
            l2 = levers.compute_l2_delta(bundle, sig, task_set, eta=levers.ETA)
            bundle = apply_bundle_overrides(
                bundle, {"scale": l2.get("scale", {}),
                         "delta": l2.get("delta", {})}, None
            )
            if out_dir is not None:
                (out_dir / f"delta_{cond}_{lever}.json").write_text(
                    json.dumps(l2, indent=2), encoding="utf-8")
                assert cache_p is not None
                cache_p.write_text(json.dumps(l2, indent=2), encoding="utf-8")
        elif lever == "L3":
            # L3 returns a plain {skill: new_description} dict: pass it as the
            # description_override argument, never through merge_deltas.
            l3 = levers.compute_l3_delta(
                bundle, sig, task_set, benchmark_model=rewriter_model
            )
            bundle = apply_bundle_overrides(bundle, None, l3 or None)
            if out_dir is not None:
                # Keep the readable export; only the fingerprinted file is
                # eligible for automatic reuse.
                (out_dir / f"delta_{cond}_{lever}.json").write_text(
                    json.dumps(l3, indent=2), encoding="utf-8")
                assert cache_p is not None
                cache_p.write_text(json.dumps(l3, indent=2), encoding="utf-8")
                (out_dir / f"{cache_name}.metadata.json").write_text(
                    json.dumps({"benchmark_model": rewriter_model,
                                "rewriter_model": rewriter_model,
                                "input_fingerprint": fingerprint}, indent=2),
                    encoding="utf-8",
                )
        print(f"[full87] {cond}: applied {lever} "
              f"(edges={len(bundle['edges'])})", flush=True)
    if rewriter_model is not None:
        bundle["segos_rewriter_model"] = rewriter_model
    return bundle

def evolve_bundle_serial(base_bundle, sig, task_set, cond, out_dir=None, *, benchmark_model=None, retrieval=None):
    with use_settings(retrieval):
        return _evolve_bundle_serial(base_bundle, sig, task_set, cond, out_dir,
                                     benchmark_model=benchmark_model)
