#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any



# Per-skill experience-weighting baselines (node-level flat re-ranking).
# EXPERIENCE_WEIGHTING in {none, success_rate, mean_reward}; the bundle skills
# carry an `experience_weight` field that is added to the PPR score during
# ranking. Default `none` keeps the main experiment path identical; these
# baselines are not part of the SE-GoS pipeline and no bundle in this repository
# sets the field.
EXPERIENCE_WEIGHTING = os.environ.get("EXPERIENCE_WEIGHTING", "none")
EXPERIENCE_LAMBDA = float(os.environ.get("EXPERIENCE_LAMBDA", "1.0"))

# Avoid-edge bundle-composition exclusion (execution-negative structural signal).
# A candidate skill whose avoid-edge partner is already selected is dropped from
# the bundle, so co-loading pairs that failed together are never both surfaced.
# This is the only way an avoid edge is consumed: avoid edges carry weight 0, so
# the PPR transition operator ignores them regardless of this flag, and they are
# deliberately excluded from traversal because walking a conflict edge would
# surface exactly the skills the evidence says not to co-load.
#
# On by default; it is a no-op until a bundle actually contains avoid edges.
AVOID_BUNDLE_EXCLUSION = os.environ.get("AVOID_BUNDLE_EXCLUSION", "1") == "1"

# SkillDAG-style three-channel output (matches / typed neighbors / conflicts).
# Opt-in at runtime (the agent prefixes its query call with the env var), so the
# default fused-bundle output -- and every existing condition -- is unchanged.
CHANNEL_SEARCH = os.environ.get("GOS_CHANNEL_SEARCH", "0") == "1"


TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "arg",
    "args",
    "array",
    "bool",
    "boolean",
    "data",
    "dict",
    "file",
    "float",
    "for",
    "from",
    "in",
    "input",
    "int",
    "json",
    "list",
    "object",
    "of",
    "on",
    "or",
    "output",
    "path",
    "record",
    "result",
    "set",
    "str",
    "string",
    "text",
    "the",
    "to",
    "value",
}

DEFAULT_REVERSE_WEIGHTS = {
    "dependency": 1.0,
    "workflow": 0.5,
    "semantic": 0.2,
    "alternative": 0.1,
}

MODEL_ALIASES = {
    "openai/text-embedding-3-large": "openai/text-embedding-3-large",
    # Local ollama deployment serves the bare model name; strip the provider prefix.
    "openai/bge-m3": "bge-m3",
}

EMBEDDING_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def signature_tokens(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        lowered = value.lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        if normalized:
            tokens.add(normalized)
        for token in re.findall(r"[a-z0-9]+", lowered):
            token = token.rstrip("s")
            if len(token) < 3 or token in TOKEN_STOPWORDS:
                continue
            tokens.add(token)
    return tokens


def _token_overlap_score(query_tokens: set[str], values: list[str]) -> float:
    candidate_tokens = signature_tokens(values)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(len(query_tokens), 1)


def _field_bonus(query_tokens: set[str], values: list[str], weight: float) -> float:
    return weight * _token_overlap_score(query_tokens, values)


def _shared_field_score(left_values: list[str], right_values: list[str]) -> float:
    left_tokens = signature_tokens(left_values)
    right_tokens = signature_tokens(right_values)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if not overlap:
        return 0.0
    return len(overlap) / max(min(len(left_tokens), len(right_tokens)), 1)


def _rerank_skill_score(
    query_tokens: set[str],
    query_text: str,
    skill: dict[str, Any],
) -> float:
    """GoS's field-weighted lexical rerank with eta=0 (no semantic term).

    Ported from GoS's gos/core/engine.py `_rerank_skill_score` with
    graph_score=0 and semantic_rank=None -- i.e. fully lexical seeding
    (z_i = eta*s_sem + (1-eta)*s_lex with eta=0). Schema-derived terms
    (rewritten-query task_name/artifacts/domain) are omitted because the
    harness seeds from the raw task query, not a rewritten query schema.
    """
    score = 0.0
    score += _field_bonus(query_tokens, [skill["name"]], 1.25)
    score += _field_bonus(
        query_tokens,
        [skill.get("one_line_capability", ""), skill.get("description", "")],
        0.9,
    )
    score += _field_bonus(query_tokens, skill.get("domain_tags", []), 1.15)
    score += _field_bonus(query_tokens, skill.get("tooling", []), 0.95)
    score += _field_bonus(
        query_tokens,
        skill.get("inputs", []) + skill.get("outputs", []),
        0.75,
    )
    score += _field_bonus(query_tokens, skill.get("example_tasks", []), 0.8)
    score += _field_bonus(query_tokens, skill.get("script_entrypoints", []), 0.6)

    normalized_node_name = re.sub(r"[^a-z0-9]+", " ", skill["name"].lower()).strip()
    if normalized_node_name and normalized_node_name in query_text.lower():
        score += 1.2

    if skill.get("script_entrypoints"):
        score += 0.08
    return score


def build_rank_distribution(count: int) -> list[float]:
    if count <= 0:
        return []

    weights = [1.0 / float(index + 1) for index in range(count)]
    total = sum(weights)
    return [weight / total for weight in weights]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[: max_chars - 3].rstrip()
    return f"{clipped}..."


def render_skill_payload(skill: dict[str, Any], max_chars: int) -> str:
    content = skill.get("raw_content") or skill.get("rendered_snippet") or f"{skill['name']}\n\n{skill['description']}"
    content = clip_text(content, max_chars)
    header = [
        f"## Skill: {skill['name']}",
        f"Source: {skill.get('source_path') or 'inline'}",
    ]
    return "\n".join([*header, content])


def lexical_seed_scores(
    query: str,
    skills: list[dict[str, Any]],
    seed_top_k: int,
) -> list[tuple[int, float, int]]:
    query_tokens = signature_tokens([query])
    if not query_tokens:
        return []

    scored: list[tuple[int, float]] = []
    for index, skill in enumerate(skills):
        node_tokens = signature_tokens(
            [
                skill["name"],
                skill.get("one_line_capability", ""),
                skill["description"],
                "\n".join(skill.get("inputs", [])),
                "\n".join(skill.get("outputs", [])),
                "\n".join(skill.get("domain_tags", [])),
                "\n".join(skill.get("tooling", [])),
                "\n".join(skill.get("example_tasks", [])),
                "\n".join(skill.get("script_entrypoints", [])),
                skill.get("rendered_snippet", ""),
            ]
        )
        overlap = query_tokens & node_tokens
        if overlap:
            # GoS's lexical seed (eta=0): token-overlap ratio plus the
            # field-weighted rerank term from engine.py _rerank_skill_score.
            score = len(overlap) / max(len(query_tokens), 1)
            score += _rerank_skill_score(query_tokens, query, skill)
            scored.append((index, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[:seed_top_k]
    if not selected:
        return []

    weights = build_rank_distribution(len(selected))
    return [(index, float(weights[rank]), rank + 1) for rank, (index, _) in enumerate(selected)]


def build_transition(
    skills: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    reverse_mode: str = "full",
) -> list[dict[int, float]]:
    if reverse_mode not in {"full", "none"}:
        raise ValueError(f"Unsupported reverse mode: {reverse_mode}")

    name_to_index = {skill["name"]: index for index, skill in enumerate(skills)}
    transition: list[dict[int, float]] = [{} for _ in skills]

    for edge in edges:
        source_index = name_to_index.get(edge["source"])
        target_index = name_to_index.get(edge["target"])
        if source_index is None or target_index is None:
            continue

        # A missing weight defaults to 1.0; an explicit 0.0 must stay 0. Using
        # `edge.get("weight", 1.0) or 1.0` collapses both to 1.0, which turns
        # zero-weight edges (the `avoid` relation) into the strongest edges in
        # the graph instead of skipping them.
        raw_weight = edge.get("weight")
        forward_weight = max(1.0 if raw_weight is None else float(raw_weight), 0.0)
        if forward_weight <= 0:
            continue

        transition[source_index][target_index] = transition[source_index].get(target_index, 0.0) + forward_weight

        reverse_weight = DEFAULT_REVERSE_WEIGHTS.get(edge.get("type", "semantic"), 0.0) if reverse_mode == "full" else 0.0
        if reverse_weight > 0:
            transition[target_index][source_index] = transition[target_index].get(source_index, 0.0) + (forward_weight * reverse_weight)

    for index, row in enumerate(transition):
        if not row:
            row[index] = 1.0
            continue
        total = sum(row.values())
        if total <= 0:
            row.clear()
            row[index] = 1.0
            continue
        for target_index, value in list(row.items()):
            row[target_index] = value / total

    return transition


def personalized_pagerank(
    transition: list[dict[int, float]],
    personalization: list[float],
    *,
    damping: float,
    max_iter: int,
    tol: float,
) -> list[float]:
    if not transition or not personalization:
        return []

    scores = personalization[:]
    for _ in range(max_iter):
        next_scores = [damping * value for value in personalization]
        factor = 1.0 - damping
        for source_index, row in enumerate(transition):
            source_score = scores[source_index]
            if source_score <= 0:
                continue
            for target_index, weight in row.items():
                next_scores[target_index] += factor * weight * source_score

        delta = sum(abs(new - old) for new, old in zip(next_scores, scores))
        scores = next_scores
        if delta <= tol:
            break

    total = sum(scores)
    if total > 0:
        scores = [score / total for score in scores]
    return scores


def fit_skills_to_budget(
    query: str,
    ranked_skills: list[dict[str, Any]],
    max_context_chars: int,
) -> list[dict[str, Any]]:
    if not ranked_skills or max_context_chars <= 0:
        return []

    total_chars = len(f"# Skill bundle for query: {query}")
    selected: list[dict[str, Any]] = []

    for skill in ranked_skills:
        remaining_context = max_context_chars - total_chars - 2
        if remaining_context <= 0:
            break

        payload = skill["payload"]
        if len(payload) > remaining_context:
            payload = clip_text(payload, remaining_context)

        if not payload:
            break

        updated_skill = dict(skill)
        updated_skill["payload"] = payload
        selected.append(updated_skill)
        total_chars += 2 + len(payload)

        if len(payload) < len(skill["payload"]):
            break

    return selected


def render_context(
    query: str,
    skills: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not skills:
        return clip_text(
            "\n\n".join(
                [
                    f"# Skill bundle for query: {query}",
                    "## Retrieval Status",
                    "Retrieval Status: NO_SKILL_HIT",
                    "No relevant skill bundle was found.",
                    "Do not claim that you used a retrieved skill.",
                    "Proceed on a no-skill path.",
                    "Before implementing, inspect the task tests/verifier and satisfy the minimum acceptance requirements.",
                ]
            ),
            max_chars,
        )

    sections = [
        f"# Skill bundle for query: {query}",
        "## Retrieval Status",
        "Retrieval Status: SKILL_HIT",
        "Use retrieved skills to narrow the solution space and take the shortest path to verifier pass.",
        "Before implementing, inspect the task tests/verifier and identify the minimum acceptance requirements.",
        "Satisfy only the minimum requirements first.",
        "Treat retrieved skills as constraints and reusable implementations, not permission to branch out.",
        "Use the exact Source paths already provided. Do not reconstruct paths from the skill name or scan the whole library if a Source path is already available.",
        "Prefer adapting retrieved scripts/interfaces over building a more general replacement.",
    ]
    for skill in skills:
        sections.append(skill["payload"])

    if relations:
        relation_lines = ["## Graph evidence"]
        for relation in relations:
            candidate_lines = [
                *relation_lines,
                (f"- {relation['source']} --({relation['type']})--> {relation['target']}: {relation['description']}"),
            ]
            candidate_context = "\n\n".join([*sections, "\n".join(candidate_lines)])
            if len(candidate_context) > max_chars:
                break
            relation_lines = candidate_lines

        if len(relation_lines) > 1:
            sections.append("\n".join(relation_lines))

    context = "\n\n".join(sections)
    return clip_text(context, max_chars)


def render_summary(
    query: str,
    skills: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
) -> str:
    if not skills:
        return "\n".join(
            [
                "### Retrieval Status",
                "- Retrieval Status: NO_SKILL_HIT",
                "- No relevant skill bundle was found for this query.",
                "- Do not claim that you used a retrieved skill.",
                "- Proceed on a no-skill path and inspect the task verifier/tests for the minimum requirements.",
                f"\nQuery: {query}",
            ]
        )

    lines = [
        "### Retrieval Status",
        "- Retrieval Status: SKILL_HIT",
        "- Use retrieved skills to narrow the solution space and take the shortest path to verifier pass.",
        "- Before coding, inspect the task verifier/tests and identify the minimum acceptance requirements.",
        "- Satisfy only those minimum requirements first.",
        "- Treat retrieved skills as constraints and reusable implementations, not permission to open extra branches.",
        "- Use the exact `Source:` path returned below. Do not reconstruct paths from the skill name or scan the whole library if a Source path is already available.",
        "- Prefer adapting the retrieved skill's scripts/interfaces over writing a broader replacement.",
        "\n### Retrieved Skills",
    ]
    for skill in skills:
        semantic_rank = f", seed rank {skill['semantic_rank']}" if skill.get("semantic_rank") is not None else ""
        lines.append(f"- {skill['name']}: {skill['description']} (graph score={skill['score']:.4f}{semantic_rank})")
        if skill.get("source_path"):
            lines.append(f"  Source: {skill['source_path']}")
        if skill.get("script_entrypoints"):
            preview = ", ".join(skill["script_entrypoints"][:3])
            lines.append(f"  Scripts: {preview}")

    if seeds:
        lines.append("\n### Semantic Seeds")
        for seed in seeds:
            lines.append(f"- {seed['name']} (seed weight={seed['seed_weight']:.4f}, rank={seed['semantic_rank']})")

    if relations:
        lines.append("\n### Graph Edges")
        for relation in relations:
            lines.append(
                f"- {relation['source']} --({relation['type']})--> {relation['target']}: "
                f"{relation['description']} (weight={relation['weight']:.3f})"
            )

    return "\n".join(lines)


def _reachable_neighbor_names(
    bundle_edges: list[dict[str, Any]],
    seed_names: set[str],
    *,
    depth: int = 2,
    max_neighbors: int = 8,
) -> list[tuple[str, str, str]]:
    """Skills reachable from seed matches within ``depth`` typed hops.

    Only non-semantic typed edges (workflow / dependency) are traversed: those
    carry real prerequisite/successor semantics learned from execution traces.
    The generic ``semantic`` similarity edges are deliberately not walked --
    over a dense kNN graph they surface only noise, not structure.
    Returns (neighbor_name, relation_label, via_match).
    """
    WALKABLE = {"workflow", "dependency"}
    adjacency: dict[str, list[tuple[str, str, str]]] = {}  # name -> (other, etype, dir)
    for edge in bundle_edges:
        etype = edge.get("type", "semantic")
        if etype not in WALKABLE:
            continue
        source = edge.get("source", "")
        target = edge.get("target", "")
        if not source or not target:
            continue
        adjacency.setdefault(source, []).append((target, etype, "out"))
        adjacency.setdefault(target, []).append((source, etype, "in"))

    frontier = sorted(seed_names)
    seen: set[str] = set(frontier)
    by_match: dict[str, str] = {}
    order: list[str] = []
    for _ in range(depth):
        next_frontier: list[str] = []
        for name in frontier:
            for other, etype, direction in adjacency.get(name, []):
                if other in seen:
                    continue
                seen.add(other)
                by_match[other] = by_match.get(other, name)
                order.append(other)
                next_frontier.append(other)
        frontier = next_frontier
        if not frontier:
            break

    out: list[tuple[str, str, str]] = []
    for name in order[:max_neighbors]:
        out.append((name, "", by_match.get(name, "")))
    return out


def render_channels(
    query: str,
    skills: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    bundle_edges: list[dict[str, Any]],
    all_skills: list[dict[str, Any]] | None = None,
) -> str:
    """SkillDAG-style separated-evidence rendering: matches / neighbors / conflicts.

    Unlike the fused bundle summary, this keeps the three channels apart so a
    weak backbone can drop a channel under a tight context budget, re-query on
    its own schedule, and load full skill bodies only once it decides to use a
    skill (via the Source: path).
    """
    if not skills:
        return "\n".join(
            [
                "### Retrieval Status",
                "- Retrieval Status: NO_SKILL_HIT",
                "- No relevant skill was found for this query.",
                "- Re-query with different wording if you have more context; otherwise proceed on a no-skill path and inspect the verifier for the minimum requirements.",
            ]
        )

    skill_by_name = {
        s["name"]: s for s in (all_skills if all_skills is not None else skills)
    }
    seed_names = {seed["name"] for seed in seeds}
    matches = [s for s in skills if s["name"] in seed_names]
    neighbor_ranked = [s for s in skills if s["name"] not in seed_names]
    neighbors = _reachable_neighbor_names(
        bundle_edges, seed_names, depth=2, max_neighbors=8
    )
    # conflict edges (co-selection harm) are exclusion-only: never navigate them.
    conflict_map: dict[str, list[str]] = {}
    for edge in bundle_edges:
        if edge.get("type") != "avoid":
            continue
        conflict_map.setdefault(edge["source"], []).append(edge["target"])
        conflict_map.setdefault(edge["target"], []).append(edge["source"])

    lines = [
        "### Retrieval Status",
        "- Retrieval Status: SKILL_HIT",
        "- Three evidence channels below. Treat **Matches** as your primary leads.",
        "- **Neighbors** are skills reachable through typed prerequisite/workflow edges from a match -- load one only if its relationship suggests it is needed.",
        "- **Conflicts** are pairs that should NOT be co-loaded.",
        "- Load a full skill body only after deciding to use it, via its exact `Source:` path. Do not reconstruct paths or scan the library.",
        "- Re-run this query with refined wording if the matches miss the mark.",
    ]

    lines.append("\n### Matches")
    if not matches:
        lines.append("- (no seed-text matches made the budget; see neighbors)")
    for skill in matches:
        lines.append(f"- {skill['name']}: {skill['description']}")
        if skill.get("source_path"):
            lines.append(f"  Source: {skill['source_path']}")
        if skill.get("script_entrypoints"):
            lines.append(f"  Scripts: {', '.join(skill['script_entrypoints'][:3])}")

    seen_nb: set[str] = set()
    lines.append("\n### Neighbors (typed workflow/dependency, depth<=2 from matches)")
    for nb_name, _label, via in neighbors:
        if nb_name in seen_nb:
            continue
        seen_nb.add(nb_name)
        skill = skill_by_name.get(nb_name)
        desc = skill.get("description", "") if skill else ""
        lines.append(f"- {nb_name}: {desc} (reached via match '{via}')")
        if skill and skill.get("source_path"):
            lines.append(f"  Source: {skill['source_path']}")
    for skill in neighbor_ranked:
        if skill["name"] in seen_nb:
            continue
        seen_nb.add(skill["name"])
        lines.append(f"- {skill['name']}: {skill['description']} (graph diffusion; not a seed match)")
        if skill.get("source_path"):
            lines.append(f"  Source: {skill['source_path']}")
    if not seen_nb:
        lines.append("- (no additional skills reachable through typed edges from these matches)")

    conflicts: list[str] = []
    for match in matches:
        for other in conflict_map.get(match["name"], []):
            conflicts.append(f"- do not co-load: {match['name']} <-> {other}")
    lines.append("\n### Conflicts")
    lines.extend(conflicts or ["- none identified"])

    return "\n".join(lines)


def normalize_embedding_model(model: str) -> str:
    normalized = (model or "").strip()
    return MODEL_ALIASES.get(normalized, normalized)


def fetch_embedding(query: str, model: str, api_key: str, base_url: str) -> list[float]:
    normalized_model = normalize_embedding_model(model)
    endpoint = f"{base_url.rstrip('/')}/embeddings"
    max_attempts = max(1, int(os.getenv("GOS_LIGHT_EMBED_RETRY_ATTEMPTS", "4")))
    last_error: str | None = None
    payload = json.dumps({"model": normalized_model, "input": [query]}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8", errors="replace")
                response_json = json.loads(body)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"Embedding request failed for model={normalized_model!r} via {endpoint}: status={exc.code}, body={body[:400]}"
            if exc.code not in EMBEDDING_RETRYABLE_STATUS_CODES:
                raise RuntimeError(last_error) from exc
        except Exception as exc:
            last_error = f"Embedding request failed for model={normalized_model!r} via {endpoint}: request_error={exc}"
        else:
            data = response_json.get("data") or []
            if not data:
                raise RuntimeError(f"Embedding response missing data: {response_json}")
            embedding = data[0].get("embedding") or []
            if not embedding:
                raise RuntimeError(f"Embedding response contained an empty vector: {response_json}")
            return [float(x) for x in embedding]

        if attempt < max_attempts:
            time.sleep(min(8.0, 0.75 * attempt))

    raise RuntimeError(last_error or f"Embedding request failed for model={normalized_model!r} via {endpoint}")


def load_vector_store(store_path: Path) -> dict[str, Any]:
    with store_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid vector store payload: {type(payload)}")
    ids = [int(item) for item in (payload.get("ids") or [])]
    dim = int(payload.get("dim") or 0)
    blob = payload.get("vectors_f32_le") or b""
    if dim <= 0:
        raise RuntimeError(f"Invalid vector store dim: {dim}")
    if len(blob) != len(ids) * dim * 4:
        raise RuntimeError(f"Vector blob size mismatch: ids={len(ids)}, dim={dim}, bytes={len(blob)}")
    return {"ids": ids, "dim": dim, "blob": blob}


def knn_query_vectors(
    query_vector: list[float],
    store: dict[str, Any],
    *,
    top_k: int,
) -> tuple[list[int], list[float]]:
    ids: list[int] = store["ids"]
    dim: int = store["dim"]
    blob: bytes = store["blob"]
    if not ids or top_k <= 0:
        return [], []
    if len(query_vector) != dim:
        raise RuntimeError(f"Query embedding dim mismatch: expected {dim}, got {len(query_vector)}")

    q_norm = math.sqrt(sum(value * value for value in query_vector))
    if q_norm == 0.0:
        raise RuntimeError("Query embedding had zero norm")
    normalized_query = [value / q_norm for value in query_vector]
    matrix = memoryview(blob).cast("f")

    scored: list[tuple[float, int]] = []
    for row_idx, skill_id in enumerate(ids):
        start = row_idx * dim
        row = matrix[start : start + dim]
        dot = 0.0
        norm_sq = 0.0
        for idx in range(dim):
            value = float(row[idx])
            dot += value * normalized_query[idx]
            norm_sq += value * value
        distance = 1.0 if norm_sq == 0.0 else 1.0 - (dot / math.sqrt(norm_sq))
        scored.append((distance, skill_id))

    scored.sort(key=lambda item: item[0])
    top = scored[: min(top_k, len(scored))]
    return [skill_id for distance, skill_id in top], [distance for distance, skill_id in top]


def embedding_seed_scores(
    query: str,
    skills: list[dict[str, Any]],
    seed_top_k: int,
    *,
    vector_store_path: Path,
) -> list[tuple[int, float, int]]:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY for embedding-seeded graphskills-query")
    model = os.getenv("GOS_EMBEDDING_MODEL", "openai/text-embedding-3-large")
    base_url = os.getenv("GOS_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    query_vector = fetch_embedding(query, model=model, api_key=api_key, base_url=base_url)
    store = load_vector_store(vector_store_path)
    labels, _distances = knn_query_vectors(query_vector, store, top_k=seed_top_k)

    ranked_indices: list[int] = []
    seen: set[int] = set()
    for raw_index in labels:
        index = int(raw_index)
        if index < 0 or index >= len(skills) or index in seen:
            continue
        seen.add(index)
        ranked_indices.append(index)

    if not ranked_indices:
        return []

    weights = build_rank_distribution(len(ranked_indices))
    return [(index, float(weights[rank]), rank + 1) for rank, index in enumerate(ranked_indices)]


def rank_from_seed_entries(
    skills: list[dict[str, Any]],
    seed_entries: list[tuple[int, float, int]],
    *,
    max_skill_chars: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, weight, rank in seed_entries:
        skill = skills[index]
        ranked.append(
            {
                "name": skill["name"],
                "description": skill["description"],
                "source_path": skill.get("source_path", ""),
                "score": float(weight),
                "semantic_rank": rank,
                "inputs": skill.get("inputs", []),
                "outputs": skill.get("outputs", []),
                "compatibility": skill.get("compatibility", []),
                "allowed_tools": skill.get("allowed_tools", []),
                "rendered_snippet": skill.get("rendered_snippet", ""),
                "payload": render_skill_payload(skill, max_skill_chars),
            }
        )
    return ranked


def rank_one_hop(
    skills: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    seed_entries: list[tuple[int, float, int]],
    *,
    max_skill_chars: int,
) -> list[dict[str, Any]]:
    """Rank seeds followed by their strongest incoming dependency prerequisites."""
    ranked = rank_from_seed_entries(
        skills,
        seed_entries,
        max_skill_chars=max_skill_chars,
    )
    name_to_index = {skill["name"]: index for index, skill in enumerate(skills)}
    selected_indices = {index for index, _, _ in seed_entries}
    seed_order = {index: rank for index, _, rank in seed_entries}
    candidates: list[tuple[int, float, int]] = []

    for edge in edges:
        if edge.get("type") != "dependency":
            continue
        source_index = name_to_index.get(edge.get("source", ""))
        target_index = name_to_index.get(edge.get("target", ""))
        if source_index is None or target_index is None:
            continue
        if target_index not in seed_order or source_index in selected_indices:
            continue
        candidates.append(
            (
                seed_order[target_index],
                -float(edge.get("weight", 1.0) or 1.0),
                source_index,
            )
        )

    for _seed_rank, negative_weight, index in sorted(candidates):
        if index in selected_indices:
            continue
        selected_indices.add(index)
        skill = skills[index]
        ranked.append(
            {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "source_path": skill.get("source_path", ""),
                "score": -negative_weight,
                "semantic_rank": None,
                "inputs": skill.get("inputs", []),
                "outputs": skill.get("outputs", []),
                "compatibility": skill.get("compatibility", []),
                "allowed_tools": skill.get("allowed_tools", []),
                "rendered_snippet": skill.get("rendered_snippet", ""),
                "payload": render_skill_payload(skill, max_skill_chars),
            }
        )

    return ranked


def build_seed_entries(
    query: str,
    skills: list[dict[str, Any]],
    seed_top_k: int,
    *,
    seed_mode: str,
    vector_store_path: Path | None,
) -> list[tuple[int, float, int]]:
    if seed_mode == "embedding":
        if vector_store_path is None:
            raise RuntimeError("Embedding seed mode requires --vector-store / vectors.pkl")
        return embedding_seed_scores(
            query,
            skills,
            seed_top_k,
            vector_store_path=vector_store_path,
        )
    if seed_mode != "lexical":
        raise RuntimeError(f"Unsupported seed mode: {seed_mode}")
    return lexical_seed_scores(query, skills, seed_top_k)


def retrieve(
    bundle: dict[str, Any],
    query: str,
    *,
    top_n: int,
    seed_top_k: int,
    max_skill_chars: int,
    max_context_chars: int,
    seed_mode: str = "lexical",
    propagation_mode: str = "ppr",
    reverse_mode: str = "full",
    vector_store_path: Path | None = None,
) -> dict[str, Any]:
    skills = bundle.get("skills", [])
    edges = bundle.get("edges", [])
    metadata = bundle.get("metadata", {})

    budget = {
        "seed_top_k": seed_top_k,
        "top_n": top_n,
        "max_skill_chars": max_skill_chars,
        "max_context_chars": max_context_chars,
        "ppr_damping": metadata.get("ppr_damping", 0.2) if propagation_mode == "ppr" else 0.0,
        "seed_mode": seed_mode,
        "propagation_mode": propagation_mode,
        "reverse_mode": reverse_mode,
    }

    if not query.strip():
        return {
            "query": query,
            "summary": "Empty query.",
            "rendered_context": "",
            "skills": [],
            "relations": [],
            "seeds": [],
            "budget": budget,
        }

    candidate_seed_entries = build_seed_entries(
        query,
        skills,
        max(seed_top_k, top_n),
        seed_mode=seed_mode,
        vector_store_path=vector_store_path,
    )
    seed_entries = candidate_seed_entries[:seed_top_k]
    if not seed_entries:
        return {
            "query": query,
            "summary": render_summary(query, [], [], []),
            "rendered_context": render_context(query, [], [], max_chars=max_context_chars),
            "skills": [],
            "relations": [],
            "seeds": [],
            "budget": budget,
        }

    rank_lookup = {index: rank for index, _, rank in seed_entries}
    if propagation_mode == "none":
        ranked = rank_from_seed_entries(
            skills,
            candidate_seed_entries[:top_n],
            max_skill_chars=max_skill_chars,
        )
    elif propagation_mode == "one-hop":
        ranked = rank_one_hop(
            skills,
            edges,
            seed_entries,
            max_skill_chars=max_skill_chars,
        )
    elif propagation_mode == "ppr":
        personalization = [0.0] * len(skills)
        for index, weight, _ in seed_entries:
            personalization[index] += weight
        total = sum(personalization)
        if total > 0:
            personalization = [weight / total for weight in personalization]

        transition = build_transition(skills, edges, reverse_mode=reverse_mode)
        scores = personalized_pagerank(
            transition,
            personalization,
            damping=float(metadata.get("ppr_damping", 0.2)),
            max_iter=int(metadata.get("ppr_max_iter", 50)),
            tol=float(metadata.get("ppr_tolerance", 1e-6)),
        )

        ranked: list[dict[str, Any]] = []

        def _sort_key(i: int) -> float:
            base = float(scores[i])
            if EXPERIENCE_WEIGHTING != "none":
                base += EXPERIENCE_LAMBDA * float(
                    skills[i].get("experience_weight", 0.0) or 0.0
                )
            return base

        for index in sorted(range(len(scores)), key=_sort_key, reverse=True):
            skill = skills[index]
            ranked.append(
                {
                    "name": skill["name"],
                    "description": skill["description"],
                    "source_path": skill.get("source_path", ""),
                    "score": float(scores[index]),
                    "semantic_rank": rank_lookup.get(index),
                    "inputs": skill.get("inputs", []),
                    "outputs": skill.get("outputs", []),
                    "compatibility": skill.get("compatibility", []),
                    "allowed_tools": skill.get("allowed_tools", []),
                    "rendered_snippet": skill.get("rendered_snippet", ""),
                    "payload": render_skill_payload(skill, max_skill_chars),
                }
            )
    else:
        raise RuntimeError(f"Unsupported propagation mode: {propagation_mode}")

    candidate_skills = ranked[:top_n]
    if AVOID_BUNDLE_EXCLUSION:
        avoid_map: dict[str, set[str]] = {}
        for edge in edges:
            if edge.get("type") == "avoid":
                avoid_map.setdefault(edge["source"], set()).add(edge["target"])
                avoid_map.setdefault(edge["target"], set()).add(edge["source"])
        if avoid_map:
            selected_names: set[str] = set()
            filtered: list[dict[str, Any]] = []
            for skill in candidate_skills:
                if avoid_map.get(skill["name"], set()) & selected_names:
                    continue  # co-loading with an already-selected partner is avoided
                filtered.append(skill)
                selected_names.add(skill["name"])
            candidate_skills = filtered

    budgeted_skills = fit_skills_to_budget(query, candidate_skills, max_context_chars)
    selected_names = {skill["name"] for skill in budgeted_skills}
    relations = [
        {
            "source": edge["source"],
            "target": edge["target"],
            "description": edge["description"],
            "type": edge["type"],
            "weight": float(edge.get("weight", 1.0)),
            "confidence": float(edge.get("confidence", 1.0)),
        }
        for edge in edges
        if edge["source"] in selected_names and edge["target"] in selected_names
    ]
    relations.sort(key=lambda edge: edge["weight"], reverse=True)

    seeds = [
        {
            "name": skills[index]["name"],
            "source_path": skills[index].get("source_path", ""),
            "seed_weight": float(weight),
            "semantic_rank": rank,
        }
        for index, weight, rank in seed_entries
    ]

    rendered_context = render_context(
        query,
        budgeted_skills,
        relations,
        max_chars=max_context_chars,
    )
    summary = render_summary(query, budgeted_skills, relations, seeds)

    return {
        "query": query,
        "summary": summary,
        "rendered_context": rendered_context,
        "skills": budgeted_skills,
        "relations": relations,
        "seeds": seeds,
        "budget": budget,
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Query the local Graph Skills benchmark bundle.")
    parser.add_argument("query", nargs="?", default="", help="Task or subproblem description.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=script_dir / "bundle.json",
        help="Path to the exported graph bundle JSON.",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=script_dir / "vectors.pkl",
        help="Path to the exported vector store used for embedding seed mode.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Maximum number of retrieved skills.")
    parser.add_argument("--seed-top-k", type=int, default=4, help="Number of seed skills.")
    parser.add_argument(
        "--seed-mode",
        choices=("lexical", "embedding"),
        default=os.getenv("GOS_LIGHT_SEED_MODE", "lexical"),
        help="Seed generation mode for lightweight graph retrieval.",
    )
    parser.add_argument(
        "--propagation-mode",
        choices=("ppr", "none", "one-hop"),
        default=os.getenv("GOS_LIGHT_PROPAGATION_MODE", "ppr"),
        help="Whether to apply personalized PageRank after seed selection.",
    )
    parser.add_argument(
        "--reverse-mode",
        choices=("full", "none"),
        default=os.getenv("GOS_LIGHT_REVERSE_MODE", "full"),
        help="Whether PPR includes relation-specific reverse transitions.",
    )
    parser.add_argument(
        "--max-skill-chars",
        type=int,
        default=1800,
        help="Maximum characters to render per skill payload.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=9000,
        help="Hard cap on the rendered skill bundle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full retrieval result as JSON.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the rendered skill bundle instead of the summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    result = retrieve(
        bundle,
        args.query,
        top_n=args.top_n,
        seed_top_k=args.seed_top_k,
        max_skill_chars=args.max_skill_chars,
        max_context_chars=args.max_context_chars,
        seed_mode=args.seed_mode,
        propagation_mode=args.propagation_mode,
        reverse_mode=args.reverse_mode,
        vector_store_path=args.vector_store,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    if CHANNEL_SEARCH:
        print(
            render_channels(
                args.query,
                result["skills"],
                result["seeds"],
                bundle.get("edges", []),
                bundle.get("skills", []),
            )
        )
        return

    if args.raw:
        print(result["rendered_context"])
        return

    print(result["summary"])


if __name__ == "__main__":
    main()
