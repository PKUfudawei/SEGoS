"""Phase C: three-lever evolution from train-subset execution feedback.

L1 (topology):  learned workflow edges seed-skill -> used-skill from successful
                trials; execution-certified dependency edges from ordered usage +
                I/O schema overlap; avoid edges from failure co-occurrence; and
                retract an avoid edge after any successful co-use.
L2 (weights):   Hebbian boost  weight(u->v) += eta * reward  for every edge whose
                target was actually used in a rewarded trial (eta=0.1), and
                halve semantic incoming weights for repeatedly surfaced,
                never-used skills.
L3 (descriptions): ProTeGi-style single-round textual gradient for skills that
                were USED but ranked below L3_RANK_THRESHOLD by retrieval:
                LLM-nabla explains the miss -> LLM-delta edits the description
                (append <=50 tokens) -> 2 paraphrases -> offline re-retrieval
                ranks candidates -> a variant kept iff it lifts the skill into
                the top-L3_RANK_THRESHOLD on at least one miss query AND (with
                L3_NO_EVICTION_GUARD) evicts no genuinely-used skill from the
                top-K of any recorded train query.  Miss queries come from
                failed trials when L3_MISS_SOURCE="failed" (the default), or
                from successful trials when L3_MISS_SOURCE="success".

All deltas are computed from the TRAIN subset only and written as three
condition-specific delta files. Each enabled lever is computed on the graph
produced by earlier enabled levers, in topology, weight, description order.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from contextvars import copy_context
from segos.retrieval.settings import options as retrieval_options
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any


from segos.retrieval import query as gq  # noqa: E402

from segos.clients.chat import LLMClient, resolve_rewriter_model  # noqa: E402

# Runtime retrieval parameters used by the benchmark agent (query.py defaults).
RETRIEVE_KWARGS = {
    "top_n": 5,
    "seed_top_k": 4,
    "max_skill_chars": 1800,
    "max_context_chars": 9000,
    "seed_mode": "lexical",
    "propagation_mode": "ppr",
    "vector_store_path": None,
}

ETA = 0.1  # learning rate for L2 (user-confirmed 2026-08-19)
SUCCESS_REWARD = 0.9  # reward >= this counts as a "successful" trial
NEW_EDGE_WEIGHT = 0.6
PRUNE_RETRIEVED_MIN = 2  # distinct tasks surfaced at least once without use
PRUNE_SCALE = 0.5  # incoming semantic edge weights get halved
AVOID_RETRACT_SUCC_MIN = 1  # symmetric with zero successful tasks at induction
AVOID_POLICY_VERSION = "successful-couse-veto-v1"
L3_MAX_ADDED_TOKENS = 50
L3_RANK_THRESHOLD = 3  # L3 targets used skills ranked worse than this (1-indexed rank)
# L3 rewrite policy (2026-09-05).  Historical behavior = ("success", guard off).
# "failed" targets real headroom: miss queries come only from trials that did not
# reach SUCCESS_REWARD.  The no-eviction guard rejects any rewrite that would push
# a genuinely-used skill out of the top-K of any recorded train query, making L3
# a safe no-op when no harmless rewrite exists instead of a source of regressions.
L3_MISS_SOURCE = "failed"  # "success" | "failed"
L3_NO_EVICTION_GUARD = True  # only accept rewrites that evict no used skill
L3_GUARD_TOP_K = RETRIEVE_KWARGS["top_n"]  # retrieval window the guard protects
# Require failed co-use on this many distinct training tasks, with no successful
# co-use in any inspected attempt. A single successful observation vetoes both
# induction and retention. Avoid edges only affect bundle composition, not PPR.
AVOID_FAIL_MIN = int(os.environ.get("AVOID_FAIL_MIN", "1"))
AVOID_SUCC_MAX = 0  # successful co-use always vetoes an avoid relation
# Relations that express "these two belong together". An avoid edge is withheld
# from any pair that already carries one: the pair cannot be both harmful and
# constructive, and the constructive witness comes from successful trials.
POSITIVE_RELATIONS = frozenset({"semantic", "workflow", "dependency", "alternative"})
DEPENDENCY_THRESHOLD = 0.6  # I/O schema-overlap threshold for execution-certified dep edges


ATTEMPT_POLICY_VERSION = "all-attempts-v1"


def training_attempts(
    signals: dict[str, Any], train_tasks: set[str],
) -> list[tuple[str, dict[str, Any], float]]:
    """Return (task, raw trial, task-normalization weight) in stable order.

    Missing policy means historical selected-attempt behavior. All-attempt
    evidence is never merged and never drawn from held-out tasks or avoid history.
    """
    policy = signals.get("evolution_attempt_policy", "selected_attempt")
    if policy == "selected_attempt":
        return [(task, signals["trials"][task], 1.0)
                for task in sorted(train_tasks) if task in signals["trials"]]
    if policy != "all_attempts":
        raise ValueError(f"Unknown evolution attempt policy: {policy}")
    if "attempts" not in signals or signals.get("merge_attempts"):
        raise ValueError("all_attempts requires unmerged raw attempts; re-extract signals")
    groups = {}
    seen = set()
    for trial in signals["attempts"]:
        task = trial["task_name"]
        if task not in train_tasks:
            continue
        if trial.get("attempt_dirs"):
            raise ValueError("all_attempts requires unmerged raw attempts")
        identity = (task, trial.get("source_job", ""), trial["trial_dir"])
        if identity in seen:
            raise ValueError(f"Duplicate training attempt: {identity}")
        seen.add(identity)
        groups.setdefault(task, []).append(trial)
    return [(task, trial, 1.0 / len(groups[task]))
            for task in sorted(groups)
            for trial in sorted(groups[task], key=lambda t: (t.get("source_job", ""), t["trial_dir"]))]


def retrieve_bundle(bundle: dict[str, Any], query: str) -> dict[str, Any]:
    if not query.strip():
        return {"skills": [], "seeds": []}
    return gq.retrieve(bundle, query, **(RETRIEVE_KWARGS | retrieval_options()))


def edge_key(edge: dict[str, Any]) -> str:
    return f"{edge['source']}|{edge['target']}|{edge.get('type', '')}"


def avoid_co_use_evidence(
    signals: dict[str, Any], train_tasks: set[str],
) -> dict[frozenset[str], dict[str, Any]]:
    """Inspect actual attempts; never manufacture co-use by merging skill sets.

    Multi-round callers supply cumulative ``avoid_trials`` from prior rounds.
    Otherwise use all raw ``attempts``, with legacy per-task signals as fallback.
    Evidence is restricted to training tasks even when history contains others.
    """
    records = signals.get("avoid_trials", signals.get("attempts"))
    allowed_tasks = train_tasks | {
        signals["trials"][key].get("task_name", key)
        for key in train_tasks if key in signals.get("trials", {})
    }
    if records is None:
        if signals.get("merge_attempts") or any(
            trial.get("attempt_dirs") for key, trial in signals["trials"].items()
            if key in train_tasks
        ):
            raise ValueError("Avoid evidence requires raw attempts; re-extract merged legacy signals.")
        records = [dict(trial, task_name=trial.get("task_name", key))
                   for key, trial in signals["trials"].items() if key in train_tasks]
    evidence: dict[frozenset[str], dict[str, Any]] = {}
    for trial in records:
        task = trial["task_name"]
        if task not in allowed_tasks:
            continue
        success = (trial.get("reward") or 0.0) >= SUCCESS_REWARD
        kind = "successful" if success else "failed"
        witness = {key: trial[key] for key in
                   ("task_name", "trial_dir", "reward", "round", "source_job")
                   if key in trial}
        for pair in combinations(sorted(set(trial.get("used_skills") or [])), 2):
            entry = evidence.setdefault(frozenset(pair), {
                "failed_tasks": set(), "successful_tasks": set(),
                "failed_attempts": [], "successful_attempts": [],
            })
            entry[f"{kind}_tasks"].add(task)
            entry[f"{kind}_attempts"].append(witness)
    return evidence


def with_description(bundle: dict[str, Any], name: str, description: str) -> dict[str, Any]:
    import copy

    patched = copy.deepcopy(bundle)
    for skill in patched["skills"]:
        if skill["name"] == name:
            skill["description"] = description
            break
    return patched


# --------------------------------------------------------------------------
# L2: Hebbian weight boost
# --------------------------------------------------------------------------
def compute_l2_delta(
    bundle: dict[str, Any],
    signals: dict[str, Any],
    train_tasks: set[str],
    *,
    eta: float = ETA,
) -> dict[str, Any]:
    edges_by_target: dict[str, list[dict[str, Any]]] = {}
    for edge in bundle["edges"]:
        # Avoid edges carry weight 0 on purpose so the diffusion operator never
        # traverses them. Reinforcing one here would give it positive weight and
        # make PPR propagate toward exactly the skill the relation says not to
        # co-load with its source, i.e. strengthen the pair the evidence called
        # harmful. A pair whose target is used successfully is already excluded
        # from the avoid set by AVOID_SUCC_MAX at induction time.
        if edge.get("type") == "avoid":
            continue
        edges_by_target.setdefault(edge["target"], []).append(edge)

    delta: dict[str, float] = {}
    records = training_attempts(signals, train_tasks)
    for task_name, trial, attempt_weight in records:
        reward = trial.get("reward") or 0.0
        if reward <= 0.0:
            continue
        for used_skill in trial.get("used_skills") or []:
            for edge in edges_by_target.get(used_skill, []):
                key = edge_key(edge)
                delta[key] = delta.get(key, 0.0) + eta * reward * attempt_weight

    # L2 soft-prunes semantic edges using task-level evidence. Multiple queries
    # from one task count once; a task surfaces a skill if any of its queries
    # retrieves it. No edge is deleted and there is no weight floor.
    retrieved_tasks: dict[str, set[str]] = {}
    used_anywhere: set[str] = set()
    for task_name, trial, attempt_weight in records:
        used_anywhere.update(trial.get("used_skills") or [])
        surfaced: set[str] = set()
        for query in trial.get("queries") or []:
            surfaced.update(
                item["name"] for item in retrieve_bundle(bundle, query).get("skills") or []
            )
        for name in surfaced:
            retrieved_tasks.setdefault(name, set()).add(task_name)
    scale = {
        edge_key(edge): PRUNE_SCALE
        for edge in bundle["edges"]
        if edge.get("type") == "semantic"
        and len(retrieved_tasks.get(edge["target"], set())) >= PRUNE_RETRIEVED_MIN
        and edge["target"] not in used_anywhere
    }
    return {"delta": delta, "scale": scale}


# --------------------------------------------------------------------------
# L1: learned topology
# --------------------------------------------------------------------------
def compute_l1_delta(
    bundle: dict[str, Any],
    signals: dict[str, Any],
    train_tasks: set[str],
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    records = training_attempts(signals, train_tasks)

    # 1) Workflow witnesses stay within each successful attempt.
    # Normalize query support by attempts per task in all_attempts mode.
    added: dict[str, dict[str, Any]] = {}
    for task_name, trial, attempt_weight in records:
        reward = trial.get("reward") or 0.0
        if reward < SUCCESS_REWARD:
            continue
        used_skills = set(trial.get("used_skills") or [])
        if not used_skills:
            continue
        for query in trial.get("queries") or []:
            result = retrieve_bundle(bundle, query)
            seed_names = [seed_skill["name"] for seed_skill in result.get("seeds") or []]
            for seed_name in seed_names:
                for used_skill in used_skills:
                    if seed_name == used_skill:
                        continue
                    key = f"{seed_name}|{used_skill}|workflow"
                    entry = added.setdefault(
                        key,
                        {
                            "source": seed_name,
                            "target": used_skill,
                            "type": "workflow",
                            "weight": NEW_EDGE_WEIGHT,
                            "confidence": NEW_EDGE_WEIGHT,
                            "description": (
                                f"Observed together in a successful trace: query "
                                f"seed {seed_name}; used skill {used_skill}."
                            ),
                            "_count": 0,
                        },
                    )
                    entry["_count"] += attempt_weight
                    entry["weight"] = min(0.9, NEW_EDGE_WEIGHT + 0.05 * max(0, entry["_count"] - 1))
                    entry["confidence"] = entry["weight"]

    # The same successful co-use that vetoes induction retracts an existing
    # avoid edge. Non-use alone is never counterevidence.
    evidence = avoid_co_use_evidence(signals, train_tasks)
    remove = []
    retraction_evidence = []
    for edge in bundle.get("edges") or []:
        if edge.get("type") != "avoid":
            continue
        pair = frozenset((edge["source"], edge["target"]))
        pair_evidence = evidence.get(pair, {})
        evidence_count = len(pair_evidence.get("successful_tasks", set()))
        if evidence_count >= AVOID_RETRACT_SUCC_MIN:
            remove.append(edge_key(edge))
            retraction_evidence.append(
                {"edge": edge_key(edge), "successful_task_count": evidence_count,
                 "successful_attempts": pair_evidence["successful_attempts"],
                 "reason": "successful_co_use"}
            )

    return {
        "add": [entry for _, entry in sorted(added.items())],
        "remove": sorted(remove),
        "retractions": sorted(retraction_evidence, key=lambda item: item["edge"]),
        "avoid_policy_version": AVOID_POLICY_VERSION,
    }


# --------------------------------------------------------------------------
# avoid: co-use harm induction (execution-negative structural signal)
# --------------------------------------------------------------------------
def compute_avoid_edges(
    signals: dict[str, Any],
    train_tasks: set[str],
    bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Induce avoid edges from failed-trial co-use in training traces.

    A pair (u, v) co-used on failed trials -- while never co-used in a successful
    trial -- is evidence for an avoid relation. This signal is narrower than
    mere co-loading into a retrieved bundle. The induced edge has weight 0 so PPR
    transition operator (which only traverses edges with weight > 0) ignores it;
    it is consumed only by bundle composition, where selecting one member drops
    the other.

    ``bundle``, when given, enforces a non-contradiction invariant: a pair that
    already carries a constructive edge (semantic, workflow, dependency,
    alternative) never also receives an avoid edge. Navigation and avoid
    signals on the same pair would cancel out, and the constructive edge is the
    better-supported of the two -- it is witnessed on successful trials.

    Counts are distinct tasks, inspected attempt by attempt. Any successful
    co-use vetoes induction, matching retraction. ``failed_attempt_count``
    additionally records the number of failed observations.
    """
    positive_pairs: set[frozenset[str]] = set()
    if bundle is not None:
        for edge in bundle.get("edges") or []:
            if edge.get("type") in POSITIVE_RELATIONS:
                positive_pairs.add(frozenset((edge["source"], edge["target"])))
    avoid: list[dict[str, Any]] = []
    for pair, evidence in avoid_co_use_evidence(signals, train_tasks).items():
        fail_count = len(evidence["failed_tasks"])
        if fail_count < AVOID_FAIL_MIN:
            continue
        succ_count = len(evidence["successful_tasks"])
        if succ_count > AVOID_SUCC_MAX:
            continue
        if pair in positive_pairs:
            continue
        u, v = sorted(pair)
        strength = min(0.9, 0.5 + 0.1 * (fail_count - 1))
        succ_phrase = ("never succeeded together" if succ_count == 0
                       else f"succeeded together {succ_count} times")
        avoid.append(
            {
                "source": u,
                "target": v,
                "type": "avoid",
                "weight": 0.0,  # PPR-inert: transition skips weight <= 0
                "confidence": strength,
                "fail_count": fail_count,
                "succ_count": succ_count,
                "failed_attempt_count": len(evidence["failed_attempts"]),
                "description": (
                    f"Failed together on {fail_count} distinct tasks and {succ_phrase}; "
                    f"co-loading {u} and {v} is avoided."
                ),
            }
        )
    avoid.sort(key=lambda e: (-e["fail_count"], e["source"], e["target"]))
    return avoid


# --------------------------------------------------------------------------
# dep: execution-certified prerequisite induction (ordered usage + I/O schema)
# --------------------------------------------------------------------------
def compute_dep_edges(
    bundle: dict[str, Any],
    signals: dict[str, Any],
    train_tasks: set[str],
) -> list[dict[str, Any]]:
    """Induce dependency edges from ordered usage plus I/O schema overlap.

    On a successful trial, if skill ``u`` is used before skill ``v`` and ``u``'s
    outputs overlap ``v``'s inputs by at least ``DEPENDENCY_THRESHOLD`` (GoS's
    deterministic rule), the trajectory certifies the schema-predicted
    prerequisite: a dependency edge ``u -> v``. Uses ``used_skills_ordered``
    (first-use order recorded by signals.py); falls back to the unordered set
    when order is unavailable.
    """
    from segos.graph import schema_overlap_score

    skills_by_name = {skill["name"]: skill for skill in bundle["skills"]}
    records = training_attempts(signals, train_tasks)
    added: dict[str, dict[str, Any]] = {}
    support_tasks: dict[str, set[str]] = {}
    for task_name, trial, attempt_weight in records:
        if (trial.get("reward") or 0.0) < SUCCESS_REWARD:
            continue
        ordered = trial.get("used_skills_ordered") or []
        if signals.get("evolution_attempt_policy", "selected_attempt") == "selected_attempt":
            ordered = ordered or trial.get("used_skills") or []
        if len(ordered) < 2:
            continue
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                u, v = ordered[i], ordered[j]
                if u == v:
                    continue
                su = skills_by_name.get(u)
                sv = skills_by_name.get(v)
                if su is None or sv is None:
                    continue
                fwd, evidence = schema_overlap_score(
                    su.get("outputs") or [],
                    sv.get("inputs") or [],
                )
                if fwd < DEPENDENCY_THRESHOLD:
                    continue
                key = f"{u}|{v}|dependency"
                entry = added.setdefault(
                    key,
                    {
                        "source": u,
                        "target": v,
                        "type": "dependency",
                        "weight": NEW_EDGE_WEIGHT,
                        "confidence": NEW_EDGE_WEIGHT,
                        "description": (
                            f"Execution-certified prerequisite: {u} produces data that "
                            f"{v} consumes ({', '.join(evidence) or 'compatible I/O'})."
                        ),
                        "_count": 0,
                    },
                )
                support_tasks.setdefault(key, set()).add(task_name)
                entry["_count"] = len(support_tasks[key])
                entry["weight"] = min(0.9, NEW_EDGE_WEIGHT + 0.05 * (entry["_count"] - 1))
                entry["confidence"] = entry["weight"]
    return [entry for _, entry in sorted(added.items())]


# --------------------------------------------------------------------------
# L3: ProTeGi description evolution
# --------------------------------------------------------------------------
def _gradient_prompt(skill: dict[str, Any], queries: list[str]) -> str:
    return (
        "You are optimizing the search-index description of a skill in a lexical skill-retrieval system.\n"
        "Retrieval computes token overlap between the task query and the skill's "
        "[name, description, inputs, outputs, rendered snippet].\n\n"
        f"Skill name: {skill['name']}\n"
        f"Current description: {skill['description']}\n"
        f"Inputs: {json.dumps(skill.get('inputs', []), ensure_ascii=False)}\n"
        f"Outputs: {json.dumps(skill.get('outputs', []), ensure_ascii=False)}\n\n"
        "This skill was actually USED by an agent to solve the following task queries, "
        f"but retrieval never surfaced it in the top-{L3_RANK_THRESHOLD} of the bundle "
        "(it ranked too low for the agent to rely on it).\n"
        + "\n".join(f"- {q}" for q in queries)
        + "\n\nExplain in 2-3 concise sentences WHY the description fails to match these "
        "queries lexically (missing terms, wrong vocabulary, wrong emphasis). "
        "Give concrete missing keywords you would add. This is a textual gradient."
    )


def _edit_prompt(skill: dict[str, Any], gradient: str) -> str:
    return (
        "Apply the following textual gradient to improve the skill's search description.\n"
        f"Skill name: {skill['name']}\n"
        f"Current description: {skill['description']}\n\n"
        f"Gradient (why it misses):\n{gradient}\n\n"
        "Rewrite the description. Requirements:\n"
        "1. Keep the original meaning and the opening clause of the current description.\n"
        "2. Append at most 50 additional tokens with concrete retrieval keywords "
        "(verbs, domain terms, artifact/format names) derived from the gradient.\n"
        "3. Output ONLY the new full description text, no preamble, no quotes."
    )


def _paraphrase_prompt(skill: dict[str, Any], edited: str) -> str:
    return (
        "Paraphrase the following skill description in TWO distinct variants.\n"
        "Keep exactly the same meaning and preserve all important technical keywords, "
        "but vary the wording and phrasing so the variants are lexically different.\n"
        f"Skill name: {skill['name']}\nDescription: {edited}\n\n"
        "Output two lines only, one variant per line, no numbering, no quotes."
    )


def compute_l3_delta(
    bundle: dict[str, Any],
    signals: dict[str, Any],
    train_tasks: set[str],
    llm: LLMClient | None = None,
    *,
    benchmark_model: str | None = None,
    max_workers: int = 8,
    seed: int = 2026,
) -> dict[str, Any]:
    import random

    rng = random.Random(seed)
    llm = llm or LLMClient(model=resolve_rewriter_model(signals, benchmark_model))

    records = training_attempts(signals, train_tasks)

    # Guard corpus + protected set shared by every candidate: replay every
    # recorded train query (any outcome); a rewrite may never push a skill that
    # some trial actually USED out of the top-L3_GUARD_TOP_K window.
    corpus_queries: list[str] = []
    seen_query: set[str] = set()
    protected: set[str] = set()
    for task_name, trial, attempt_weight in records:
        for query in trial.get("queries") or []:
            if query not in seen_query:
                seen_query.add(query)
                corpus_queries.append(query)
        protected.update(trial.get("used_skills") or [])
    skills_by_name = {skill["name"]: skill for skill in bundle["skills"]}
    protected &= set(skills_by_name)

    current_topk: dict[str, set[str]] = {}
    if L3_NO_EVICTION_GUARD:
        for q in corpus_queries:
            try:
                current_topk[q] = {
                    s["name"] for s in retrieve_bundle(bundle, q).get("skills", [])
                }
            except Exception:  # noqa: BLE001
                current_topk[q] = set()

    # Collect misses from every trial admitted by the evolution policy.
    misses: dict[str, list[str]] = {}
    for task_name, trial, attempt_weight in records:
        reward = trial.get("reward") or 0.0
        if L3_MISS_SOURCE == "success":
            if reward < SUCCESS_REWARD:
                continue
        else:  # "failed": only trials that did NOT reach the success threshold
            if reward >= SUCCESS_REWARD:
                continue
        used_skills = set(trial.get("used_skills") or [])
        if not used_skills:
            continue
        for query in trial.get("queries") or []:
            result = retrieve_bundle(bundle, query)
            retrieved = [skill["name"] for skill in result.get("skills") or []]
            rank_by_name = {name: rank for rank, name in enumerate(retrieved, start=1)}
            for used_skill in used_skills:
                rank = rank_by_name.get(used_skill, (RETRIEVE_KWARGS | retrieval_options())["top_n"] + 1)
                if rank > L3_RANK_THRESHOLD:
                    if query not in misses.setdefault(used_skill, []):
                        misses[used_skill].append(query)

    from segos.evolution.validation import within_edit_budget

    description_override: dict[str, str] = {}
    lock = threading.Lock()

    def guard_ok(candidate_bundle: dict[str, Any]) -> bool:
        """True if no protected (used) skill is evicted from top-K on any query."""
        for q in corpus_queries:
            cur = current_topk.get(q)
            if not cur:
                continue
            need = {s for s in cur if s in protected}
            if not need:
                continue
            try:
                cand = {s["name"] for s in retrieve_bundle(candidate_bundle, q).get("skills", [])}
            except Exception:  # noqa: BLE001
                return False
            if not need.issubset(cand):
                return False
        return True

    def process_skill(name: str, queries: list[str]) -> tuple[str, str | None]:
        skill = skills_by_name.get(name)
        if skill is None:
            return name, None
        try:
            gradient = llm.chat(
                [{"role": "user", "content": _gradient_prompt(skill, queries)}],
                max_tokens=1024,
                temperature=0.4,
            )
            edited = llm.chat(
                [{"role": "user", "content": _edit_prompt(skill, gradient)}],
                max_tokens=1024,
                temperature=0.5,
            )
            paraphrased = llm.chat(
                [{"role": "user", "content": _paraphrase_prompt(skill, edited)}],
                max_tokens=1024,
                temperature=0.6,
            )
            variants = [edited] + [
                line.strip().strip('"')
                for line in paraphrased.splitlines()
                if line.strip()
            ][:2]
            variants = [v for v in variants if v and v != skill["description"]][:3]
            if not variants:
                return name, None

            # offline candidate selection: keep the variant that lifts the skill
            # into the top-L3_RANK_THRESHOLD on at least one miss query, AND
            # (with the guard enabled) evicts no used skill on any train query.
            best_variant: str | None = None
            best_score: tuple[int, int] = (-1, 0)  # (hits, sum of ranks; lower is better)
            for variant in variants:
                if not within_edit_budget(skill["description"], variant, L3_MAX_ADDED_TOKENS):
                    continue
                candidate_bundle = with_description(bundle, name, variant)
                hits = 0
                rank_sum = 0
                min_rank = (RETRIEVE_KWARGS | retrieval_options())["top_n"] + 1  # reset for each candidate
                for query in queries:
                    result = retrieve_bundle(candidate_bundle, query)
                    for rank, skill_item in enumerate(result.get("skills") or [], start=1):
                        if skill_item["name"] == name:
                            hits += 1
                            rank_sum += rank
                            min_rank = min(min_rank, rank)
                            break
                if hits == 0 or min_rank > L3_RANK_THRESHOLD:
                    continue
                if L3_NO_EVICTION_GUARD and not guard_ok(candidate_bundle):
                    continue
                score = (hits, -rank_sum)
                if score > best_score:
                    best_score = score
                    best_variant = variant

            return name, best_variant
        except Exception as exc:  # noqa: BLE001
            print(f"[L3] error for skill {name}: {exc}", file=sys.stderr)
            return name, None

    miss_items = sorted(misses.items(), key=lambda item: rng.random())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(copy_context().run, process_skill, name, queries): name
            for name, queries in miss_items
        }
        for future in as_completed(futures):
            name, variant = future.result()
            if variant is not None:
                with lock:
                    description_override[name] = variant

    combined = bundle
    accepted = {}
    for name in sorted(description_override):
        candidate = with_description(combined, name, description_override[name])
        if L3_NO_EVICTION_GUARD and not guard_ok(candidate):
            continue
        # Combining descriptions may also move the edited node out of top three.
        if not any(name in [s["name"] for s in retrieve_bundle(candidate, q).get("skills", [])][:L3_RANK_THRESHOLD]
                   for q in misses[name]):
            continue
        combined = candidate
        accepted[name] = description_override[name]
    return accepted


def load_signals(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_delta(delta: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(delta, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute L1/L2/L3 evolution deltas from train signals.")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="delta output directory")
    parser.add_argument("--eta", type=float, default=ETA)
    parser.add_argument("--skip-l3", action="store_true", help="skip the ProTeGi description lever")
    parser.add_argument("--l3-max-workers", type=int, default=8)
    parser.add_argument("--model", default=None,
                        help="Benchmark model for legacy signals without model metadata; "
                             "must agree with any recorded model.")
    args = parser.parse_args()

    signals = load_signals(args.signals)
    split = load_split(args.split)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    train_tasks = set(split["train"])

    l1 = compute_l1_delta(bundle, signals, train_tasks)
    dep = compute_dep_edges(bundle, signals, train_tasks)
    from segos.graph import apply_bundle_overrides
    constructive = l1.get("add", []) + dep
    l1_bundle = apply_bundle_overrides(
        bundle, {"remove": l1.get("remove", []), "add": constructive}, None
    )
    avoid = compute_avoid_edges(signals, train_tasks, l1_bundle)
    if avoid:
        l1_bundle = apply_bundle_overrides(l1_bundle, {"add": avoid}, None)
    if dep:
        l1["add"] = constructive
    if avoid:
        l1.setdefault("add", []).extend(avoid)
    write_delta(l1, args.out / "L1.json")
    write_delta({"add": dep}, args.out / "DEP.json")
    write_delta({"add": avoid}, args.out / "AVOID.json")
    print(f"L1: {len(l1.get('add', []))} added, {len(l1.get('remove', []))} retracted "
          f"({len(dep)} dependency, {len(avoid)} avoid)")

    l2 = compute_l2_delta(l1_bundle, signals, train_tasks, eta=args.eta)
    write_delta(l2, args.out / "L2.json")
    print(f"L2: {len(l2['delta'])} weight deltas")

    if args.skip_l3:
        write_delta({}, args.out / "L3.json")
        print("L3: skipped")
    else:
        l3 = compute_l3_delta(
            apply_bundle_overrides(
                l1_bundle,
                {"scale": l2.get("scale", {}), "delta": l2.get("delta", {})},
                None,
            ),
            signals,
            train_tasks,
            benchmark_model=args.model,
            max_workers=args.l3_max_workers,
        )
        write_delta(l3, args.out / "L3.json")
        print(f"L3: {len(l3)} description edits")


if __name__ == "__main__":
    main()
