from __future__ import annotations
import re
from typing import Any

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

def schema_overlap_score(
    producer_values: list[str],
    consumer_values: list[str],
) -> tuple[float, list[str]]:
    best_score = 0.0
    best_evidence: set[str] = set()

    for producer in producer_values:
        producer_norm = re.sub(r"[^a-z0-9]+", "_", producer.lower()).strip("_")
        producer_tokens = signature_tokens([producer])
        for consumer in consumer_values:
            consumer_norm = re.sub(r"[^a-z0-9]+", "_", consumer.lower()).strip("_")
            consumer_tokens = signature_tokens([consumer])

            if producer_norm and producer_norm == consumer_norm:
                return 1.0, [producer_norm]

            if producer_norm and consumer_norm and (producer_norm in consumer_norm or consumer_norm in producer_norm):
                candidate = {producer_norm, consumer_norm}
                return 0.85, sorted(candidate)

            overlap = producer_tokens & consumer_tokens
            if not overlap:
                continue

            union = producer_tokens | consumer_tokens
            score = max(0.5, len(overlap) / max(len(union), 1))
            if score > best_score:
                best_score = score
                best_evidence = overlap

    return best_score, sorted(best_evidence)

def _edge_key(edge: dict[str, Any]) -> str:
    return f"{edge['source']}|{edge['target']}|{edge.get('type', '')}"

def apply_bundle_overrides(
    bundle: dict[str, Any],
    weight_override: dict[str, Any] | None = None,
    description_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the evolution deltas to a bundle without touching it in place.

    ``weight_override`` may carry ``add`` (edge dicts), ``remove`` (edge keys),
    and ``scale`` / ``delta`` / ``set`` (edge key -> factor / increment /
    absolute value). Edge keys are ``source|target|type``, matching
    ``evo.levers.edge_key``.

    ``weight`` and ``confidence`` move together: the transition operator reads
    ``weight`` while the confidence-weighted interpolation reads ``confidence``,
    and evolution treats them as one quantity. Nothing is clamped -- an edge
    reinforced over several rounds legitimately exceeds 1.0, and the operator
    normalises. Private keys (``_count`` and friends) are dropped from added
    edges so the written bundle stays a clean graph artifact.
    """
    import copy

    patched = copy.deepcopy(bundle)
    edges = patched.setdefault("edges", [])
    by_key: dict[str, dict[str, Any]] = {_edge_key(e): e for e in edges}

    if weight_override:
        remove = set(weight_override.get("remove") or [])
        if remove:
            edges[:] = [e for e in edges if _edge_key(e) not in remove]
            by_key = {_edge_key(e): e for e in edges}

        for edge in weight_override.get("add") or []:
            key = _edge_key(edge)
            if key in by_key:
                continue
            clean = {k: v for k, v in edge.items() if not k.startswith("_")}
            edges.append(clean)
            by_key[key] = clean

        for key, factor in (weight_override.get("scale") or {}).items():
            edge = by_key.get(key)
            if edge is None:
                continue
            edge["weight"] = edge.get("weight", 0.0) * float(factor)
            if "confidence" in edge:
                edge["confidence"] = float(edge["confidence"]) * float(factor)

        for key, amount in (weight_override.get("delta") or {}).items():
            edge = by_key.get(key)
            if edge is None:
                continue
            edge["weight"] = edge.get("weight", 0.0) + float(amount)
            if "confidence" in edge:
                edge["confidence"] = float(edge["confidence"]) + float(amount)

        for key, value in (weight_override.get("set") or {}).items():
            edge = by_key.get(key)
            if edge is None:
                continue
            edge["weight"] = float(value)
            if "confidence" in edge:
                edge["confidence"] = float(value)

    if description_override:
        by_name = {skill["name"]: skill for skill in patched.get("skills", [])}
        for name, description in description_override.items():
            if name in by_name:
                by_name[name]["description"] = description

    metadata = patched.get("metadata")
    if isinstance(metadata, dict):
        from collections import Counter
        counts = Counter(edge.get("type") for edge in edges)
        metadata["edge_count"] = len(edges)
        metadata["total_edges"] = len(edges)
        for relation in ("semantic", "workflow", "dependency", "alternative", "avoid"):
            metadata[f"{relation}_edges"] = counts[relation]
        metadata["dep_edges"] = counts["dependency"]
        metadata["is_edges"] = counts["is"]
    return patched
