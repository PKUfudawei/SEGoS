#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from segos.graph import signature_tokens, schema_overlap_score, apply_bundle_overrides
from segos.retrieval import query as shared_query

SKILLSBENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SKILLSBENCH_ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.skillsbench.experiments.ablation_conditions import (  # noqa: E402
    ABLATION_CONDITIONS,
    get_condition,
    render_condition_environment,
)
from benchmarks.skillsbench.fractional_reward import install_fractional_reward  # noqa: E402

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

GRAPH_QUERY_PATH = "/opt/graphskills/query.py"
GRAPH_QUERY_COMMAND = "graphskills-query"
GRAPH_LIBRARY_PATH = "/opt/graphskills/skills"
GRAPH_MARKER_START = "# BEGIN GRAPH SKILLS BENCHMARK"
GRAPH_MARKER_END = "# END GRAPH SKILLS BENCHMARK"
SKILLS_HOST_ENV_VAR = "SKILLSBENCH_SKILLS_HOST_DIR"
GOS_WORKSPACE_ENV_VAR = "GOS_PREBUILT_HOST_WORKSPACE"

ASSETS_DIR = SKILLSBENCH_ROOT / "graphskills_assets"
BOOTSTRAP_SKILL_DIR = ASSETS_DIR / "bootstrap_skill"
VECTOR_BOOTSTRAP_SKILL_DIR = REPO_ROOT / "skills" / "vector-skills-retriever"
QUERY_TEMPLATE = Path(shared_query.__file__)
VECTOR_QUERY_TEMPLATE = ASSETS_DIR / "vector_query.py"
ALLSKILLS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_allskills_template"
GOS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_gos_template"
VECTORSKILLS_TEMPLATE_DIR = SKILLSBENCH_ROOT / "_vectorskills_template"
SKILLSETS_ROOT = REPO_ROOT / "data" / "skillsets"
DEFAULT_SKILLSET_NAME = "skills_200"

DEFAULT_TASKS_ROOT = SKILLSBENCH_ROOT / "tasks"
DEFAULT_SKILLS_ROOT = SKILLSETS_ROOT / DEFAULT_SKILLSET_NAME
DEFAULT_OUTPUT_ROOT = SKILLSBENCH_ROOT / "generated"






def _strip_prebuilt_workspace(rendered: str) -> str:
    """Drop the GoS prebuilt-workspace mount (volume + env) from a compose file.

    Bundle-driven conditions retrieve through ``graphskills/query.py``, which
    reads only ``bundle.json`` and never touches the GoS engine workspace.
    """
    kept: list[str] = []
    for line in rendered.splitlines():
        if "/opt/graphskills/prebuilt:ro" in line:
            while kept and kept[-1].lstrip().startswith("#"):
                kept.pop()
            continue
        if line.strip().startswith("- GOS_PREBUILT_WORKING_DIR="):
            continue
        kept.append(line)
    out = "\n".join(kept)
    return out + "\n" if rendered.endswith("\n") else out


def render_compose_template(
    template_path: Path,
    *,
    destination_env_dir: Path,
    skills_root: Path,
    gos_workspace: Path | None = None,
    bundle_mount: Path | None = None,
    retrieval_condition: str | None = None,
    mount_prebuilt_workspace: bool = True,
) -> str:
    rendered = template_path.read_text(encoding="utf-8")
    skills_relative = os.path.relpath(skills_root, destination_env_dir)
    rendered = rendered.replace("../../../../data/skillsets/skills_200", skills_relative)
    rendered = rendered.replace("../../../skillsets/skills_200", skills_relative)
    rendered = rendered.replace("../../../all_skills", skills_relative)

    if gos_workspace is not None:
        workspace_relative = os.path.relpath(gos_workspace, destination_env_dir)
        rendered = rendered.replace("../../../../data/gos_workspace/all_skills_v1", workspace_relative)
        rendered = rendered.replace("../../../gos_workspace/all_skills_v1", workspace_relative)
        rendered = re.sub(
            r"(\$\{GOS_PREBUILT_HOST_WORKSPACE:-)([^}]+)(\})",
            lambda m: f"{m.group(1)}{workspace_relative}{m.group(3)}",
            rendered,
        )

    if bundle_mount is not None:
        bundle_relative = os.path.relpath(bundle_mount, destination_env_dir)
        rendered = rendered.replace("__GOS_BUNDLE_MOUNT__", bundle_relative)
    else:
        rendered = "\n".join(
            line for line in rendered.splitlines() if "__GOS_BUNDLE_MOUNT__" not in line
        )

    if retrieval_condition is not None:
        rendered = render_condition_environment(
            rendered,
            get_condition(retrieval_condition),
        )

    if not mount_prebuilt_workspace:
        rendered = _strip_prebuilt_workspace(rendered)

    return rendered


def build_docker_block(variant: str) -> str:
    lines = [GRAPH_MARKER_START]

    instruction_block = [
        "COPY CLAUDE.md /root/CLAUDE.md",
        "COPY AGENTS.md /root/AGENTS.md",
        "COPY GEMINI.md /root/GEMINI.md",
        "RUN for d in /app /app/workspace /app/video /root /workspace /repo; do "
        'if [ -d "$d" ]; then '
        'cp /root/CLAUDE.md "$d/CLAUDE.md" 2>/dev/null || true; '
        'cp /root/AGENTS.md "$d/AGENTS.md" 2>/dev/null || true; '
        'cp /root/GEMINI.md "$d/GEMINI.md" 2>/dev/null || true; '
        "fi; "
        "done",
    ]

    if variant in {"graphskills", "vectorskills"}:
        command_name = "graphskills-query" if variant == "graphskills" else "vectorskills-query"
        lines.append(
            "RUN if command -v python3 >/dev/null 2>&1; then :; "
            "elif command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; "
            f'else echo "python3 is required for {command_name}" >&2; exit 1; fi'
        )

    if variant == "graphskills":
        lines.extend(
            [
                "COPY graphskills /opt/graphskills",
                "RUN chmod +x /opt/graphskills/query.py && ln -sf /opt/graphskills/query.py /usr/local/bin/graphskills-query",
                "ENV GRAPHSKILLS_ROOT=/opt/graphskills",
                *instruction_block,
            ]
        )
    elif variant == "vectorskills":
        lines.extend(
            [
                'RUN if command -v python3 >/dev/null 2>&1; then :; elif command -v apt-get >/dev/null 2>&1; then apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*; else echo "python3 is required for vectorskills-query" >&2; exit 1; fi',
                "COPY vectorskills /opt/graphskills/vectorskills",
                "RUN chmod +x /opt/graphskills/vectorskills/query.py && ln -sf /opt/graphskills/vectorskills/query.py /usr/local/bin/vectorskills-query",
                "ENV GOS_PREBUILT_WORKING_DIR=/opt/graphskills/prebuilt",
                "ENV GOS_WORKING_DIR=/opt/graphskills/runtime",
                *instruction_block,
            ]
        )
    else:
        lines.extend(instruction_block)

    lines.append(GRAPH_MARKER_END)
    return "\n".join(lines)


_PREINSTALL_BEGIN = "# BEGIN AGENT PREINSTALL"
_PREINSTALL_END = "# END AGENT PREINSTALL"
# Strips any previously-appended AGENT PREINSTALL layer (idempotent regen).
_PREINSTALL_RE = re.compile(
    rf"(?ms)^{re.escape(_PREINSTALL_BEGIN)}\n.*?^"
    rf"{re.escape(_PREINSTALL_END)}\n?"
)




def patch_dockerfile(dockerfile_path: Path, variant: str) -> None:
    original = dockerfile_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(GRAPH_MARKER_START)}.*?{re.escape(GRAPH_MARKER_END)}\n?",
        re.DOTALL,
    )
    updated = re.sub(pattern, "", original)
    updated = re.sub(r"(?m)^\s*COPY\s+skills(?:[^\n]*)\n?", "", updated)
    updated = re.sub(_PREINSTALL_RE, "", updated)
    updated = updated.rstrip()
    block = build_docker_block(variant)
    updated = f"{updated}\n\n{block}\n"
    dockerfile_path.write_text(updated, encoding="utf-8")


def copy_task_tree(source_task_dir: Path, destination_task_dir: Path, *, preserve_verifier: bool = False) -> None:
    if destination_task_dir.exists():
        shutil.rmtree(destination_task_dir)
    shutil.copytree(source_task_dir, destination_task_dir)
    normalize_task_layout(destination_task_dir)
    if not preserve_verifier and (destination_task_dir / "tests" / "test.sh").is_file():
        install_fractional_reward(destination_task_dir)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def hardlink_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        hardlink_or_copy_file(item, target)


def replace_task_skills(task_dir: Path, replacement_dir: Path) -> None:
    skills_dir = task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)
    link_or_copy_tree(replacement_dir, skills_dir)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _render_toml_section(name: str, values: dict[str, Any]) -> list[str]:
    rendered = [f"[{name}]"]
    for key, value in values.items():
        if value is None or isinstance(value, dict):
            continue
        rendered.append(f"{key} = {_toml_value(value)}")
    return rendered


def normalize_task_layout(task_dir: Path) -> None:
    """Convert current SkillsBench task.md layout for the pinned Harbor runtime."""
    if (task_dir / "task.toml").exists():
        return
    task_markdown = task_dir / "task.md"
    if not task_markdown.exists():
        return

    content = task_markdown.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", content, re.DOTALL)
    if match is None:
        raise ValueError(f"Invalid SkillsBench task.md front matter: {task_markdown}")
    config = yaml.safe_load(match.group(1)) or {}
    if not isinstance(config, dict):
        raise ValueError(f"SkillsBench task.md front matter must be a mapping: {task_markdown}")

    instruction = match.group(2).lstrip("\n")
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    metadata = dict(config.get("metadata") or {})
    verifier = dict(config.get("verifier") or {})
    agent = dict(config.get("agent") or {})
    environment = dict(config.get("environment") or {})
    solution = dict(config.get("solution") or {})
    network_mode = environment.pop("network_mode", None)
    if network_mode is not None:
        environment["allow_internet"] = network_mode != "none"
    environment.pop("workdir", None)
    verifier.pop("type", None)
    verifier.pop("service", None)
    verifier.pop("hardening", None)

    task_toml_lines = ['version = "1.0"', ""]
    for section_name, values in (
        ("metadata", metadata),
        ("verifier", verifier),
        ("agent", agent),
        ("environment", environment),
        ("solution", solution),
    ):
        task_toml_lines.extend(_render_toml_section(section_name, values))
        task_toml_lines.append("")
    (task_dir / "task.toml").write_text(
        "\n".join(task_toml_lines),
        encoding="utf-8",
    )

    for current_name, harbor_name in (("verifier", "tests"), ("oracle", "solution")):
        source = task_dir / current_name
        destination = task_dir / harbor_name
        if source.exists() and not destination.exists():
            shutil.copytree(source, destination)


def canonical_task_source(task_dir: Path) -> Path:
    """Prefer the stable canonical task under benchmarks/skillsbench/tasks.

    Older generated task directories may have stale or partially-corrupted top-level
    files. The canonical task source keeps task.toml/instruction/tests/solution
    authoritative while the generator still rewrites the environment for each method.
    """
    candidate = DEFAULT_TASKS_ROOT / task_dir.name
    if candidate.exists() and any((candidate / task_file).exists() for task_file in ("task.toml", "task.md")):
        return candidate
    return task_dir


def prepare_allskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    skills_root: Path,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir)

    skills_dir = destination_task_dir / "environment" / "skills"
    if skills_dir.exists():
        shutil.rmtree(skills_dir)

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(ALLSKILLS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            ALLSKILLS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
        ),
        encoding="utf-8",
    )

    patch_dockerfile(destination_task_dir / "environment" / "Dockerfile", "allskills")


def prepare_graphskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    bundle_path: Path,
    vector_store_path: Path | None,
    skills_root: Path,
    gos_workspace: Path,
    bundle_mount: Path | None = None,
    retrieval_condition: str | None = None,
    preserve_verifier: bool = False,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir, preserve_verifier=preserve_verifier)
    replace_task_skills(destination_task_dir, BOOTSTRAP_SKILL_DIR)

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(GOS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            GOS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
            gos_workspace=gos_workspace,
            bundle_mount=bundle_mount,
            retrieval_condition=retrieval_condition,
            # graphskills retrieval is bundle-only (query.py never opens the
            # GoS engine workspace), so the prebuilt mount is dead weight.
            mount_prebuilt_workspace=False,
        ),
        encoding="utf-8",
    )

    graphskills_dir = destination_task_dir / "environment" / "graphskills"
    if graphskills_dir.exists():
        shutil.rmtree(graphskills_dir)
    graphskills_dir.mkdir(parents=True, exist_ok=True)

    hardlink_or_copy_file(QUERY_TEMPLATE, graphskills_dir / "query.py")
    # Always bake a bundle.json so the image content is independent of the
    # runtime override: when bundle_mount is set, the per-condition bundle is
    # mounted read-only over this baked (static) copy at container start.
    hardlink_or_copy_file(bundle_path, graphskills_dir / "bundle.json")
    if vector_store_path is not None:
        hardlink_or_copy_file(vector_store_path, graphskills_dir / "vectors.pkl")

    patch_dockerfile(
        destination_task_dir / "environment" / "Dockerfile", "graphskills",
    )


def prepare_vectorskills_task(
    source_task_dir: Path,
    destination_task_dir: Path,
    metadata_path: Path,
    vector_store_path: Path,
    skills_root: Path,
    gos_workspace: Path,
) -> None:
    copy_task_tree(source_task_dir, destination_task_dir)
    copy_task_tree(VECTOR_BOOTSTRAP_SKILL_DIR, destination_task_dir / "environment" / "skills")

    env_dir = destination_task_dir / "environment"
    for name in ("AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        hardlink_or_copy_file(VECTORSKILLS_TEMPLATE_DIR / name, env_dir / name)
    (env_dir / "docker-compose.yaml").write_text(
        render_compose_template(
            VECTORSKILLS_TEMPLATE_DIR / "docker-compose.yaml",
            destination_env_dir=env_dir,
            skills_root=skills_root,
            gos_workspace=gos_workspace,
        ),
        encoding="utf-8",
    )

    vectorskills_dir = destination_task_dir / "environment" / "vectorskills"
    if vectorskills_dir.exists():
        shutil.rmtree(vectorskills_dir)
    vectorskills_dir.mkdir(parents=True, exist_ok=True)

    hardlink_or_copy_file(VECTOR_QUERY_TEMPLATE, vectorskills_dir / "query.py")
    hardlink_or_copy_file(metadata_path, vectorskills_dir / "metadata.json")
    hardlink_or_copy_file(vector_store_path, vectorskills_dir / "vectors.pkl")

    patch_dockerfile(destination_task_dir / "environment" / "Dockerfile", "vectorskills")


def build_task_list(tasks_root: Path, selected_tasks: list[str]) -> list[Path]:
    tasks = sorted(
        path for path in tasks_root.iterdir() if path.is_dir() and any((path / task_file).exists() for task_file in ("task.toml", "task.md"))
    )
    if not selected_tasks:
        return tasks

    selected = set(selected_tasks)
    filtered = [path for path in tasks if path.name in selected]
    missing = selected - {path.name for path in filtered}
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Unknown task ids: {missing_text}")
    return filtered


def write_bundle(bundle: dict[str, Any], output_root: Path) -> Path:
    bundle_dir = output_root / "shared"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "graphskills_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def write_vector_metadata(bundle: dict[str, Any], output_root: Path) -> Path:
    bundle_dir = output_root / "shared"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = bundle_dir / "vectorskills_metadata.json"
    metadata = {"skills": bundle["skills"]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def write_vector_store(bundle: dict[str, Any], skills_root: Path, gos_workspace: Path, output_root: Path) -> Path:
    if bundle.get("skills") and all(
        "graph_vertex_id" in skill for skill in bundle["skills"]
    ):
        from evaluation.analysis.workspace_bundle import write_workspace_vector_store

        return write_workspace_vector_store(
            bundle,
            gos_workspace,
            output_root / "shared" / "vectorskills_vectors.pkl",
        )

    try:
        import hnswlib  # type: ignore
    except Exception as exc:
        raise RuntimeError("hnswlib is required in the host environment to export vectorskills store") from exc

    index_path = None
    for candidate in sorted(gos_workspace.glob("entities_hnsw_index_*.bin")):
        index_path = candidate
        break
    if index_path is None:
        raise FileNotFoundError(f"No entities_hnsw_index_*.bin found under {gos_workspace}")

    dim_text = index_path.stem.removeprefix("entities_hnsw_index_")
    dim = int(dim_text)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))

    id_list = [int(item) for item in index.get_ids_list()]
    vectors = index.get_items(id_list)

    skill_by_path = {
        f"{GRAPH_LIBRARY_PATH}/{skill_file.parent.relative_to(skills_root).as_posix()}/SKILL.md": position
        for position, skill_file in enumerate(sorted(skills_root.rglob("SKILL.md")))
    }

    ordered_rows: list[int] = []
    ordered_ids: list[int] = []
    for skill in bundle["skills"]:
        source_path = skill.get("source_path", "")
        row_idx = skill_by_path.get(source_path)
        if row_idx is None:
            raise KeyError(f"Missing vector row for skill source path: {source_path}")
        ordered_rows.append(row_idx)
        ordered_ids.append(row_idx)

    # The current workspace indexing order matches the sorted SKILL.md order used above.
    # We still reorder explicitly so the exported rows align exactly with metadata.json indices.
    row_to_vector = {row_id: vectors[pos] for pos, row_id in enumerate(id_list)}
    ordered_vectors = [row_to_vector[row_id] for row_id in ordered_rows]

    store_path = output_root / "shared" / "vectorskills_vectors.pkl"
    payload = {
        "dim": dim,
        "ids": ordered_ids,
        "vectors_f32_le": b"".join(memoryview(vector).cast("B") for vector in ordered_vectors),
    }
    import pickle

    with store_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return store_path


def write_manifest(
    output_root: Path,
    *,
    tasks: list[Path],
    bundle: dict[str, Any],
    allskills_dir: Path,
    graphskills_dir: Path,
    vectorskills_dir: Path,
    skills_root: Path,
    skillset_name: str | None,
    gos_workspace: Path,
) -> None:
    manifest = {
        "source_tasks_root": str(DEFAULT_TASKS_ROOT),
        "skills_root": str(skills_root),
        "skillset_name": skillset_name,
        "gos_workspace": str(gos_workspace),
        "task_count": len(tasks),
        "task_ids": [task.name for task in tasks],
        "bundle_metadata": bundle["metadata"],
        "datasets": {
            "allskills": str(allskills_dir),
            "graphskills": str(graphskills_dir),
            "vectorskills": str(vectorskills_dir),
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )






def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SkillsBench datasets for all-skills, graph-skills, and vector-only evaluation.")
    parser.add_argument("--preserve-verifier", action="store_true", help="Keep graphskills task verifier scripts unchanged instead of installing the fractional wrapper.")
    parser.add_argument(
        "--skillset-name",
        default=DEFAULT_SKILLSET_NAME,
        help=("Named skillset under data/skillsets/. Ignored when --skills-root is explicitly provided."),
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=DEFAULT_TASKS_ROOT,
        help="Source tasks directory, typically tasks-no-skills.",
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        default=DEFAULT_SKILLS_ROOT,
        help="Root directory containing the external all_skills library.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where generated datasets will be written.",
    )
    parser.add_argument(
        "--gos-workspace",
        type=Path,
        default=None,
        help="Prebuilt GoS workspace to mount into generated graphskills tasks.",
    )
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help="Load a pre-built graph bundle JSON instead of rebuilding it (skips the O(n^2) edge construction).",
    )
    parser.add_argument(
        "--weight-override",
        type=Path,
        default=None,
        help="Path to a weight-override JSON applied on top of the base bundle (remove/scale/delta/set/add).",
    )
    parser.add_argument(
        "--description-override",
        type=Path,
        default=None,
        help="Path to a {skill_name: new_description} JSON applied on top of the base bundle.",
    )
    parser.add_argument(
        "--bundle-mount",
        type=Path,
        default=None,
        help="Host path to a bundle.json mounted read-only into the task container at /opt/graphskills/bundle.json. "
        "The bundle is also baked into the image; this mount overrides it at runtime.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Optional task id to generate. Repeat to include multiple tasks.",
    )
    parser.add_argument(
        "--retrieval-condition",
        choices=tuple(ABLATION_CONDITIONS),
        default="lexical-reverse-ppr",
        help="Pinned lightweight retrieval condition for generated graph-skills tasks.",
    )
    parser.add_argument(
        "--skip-allskills",
        action="store_true",
        help="Do not generate the all-skills full-context dataset.",
    )
    parser.add_argument(
        "--skip-graphskills",
        action="store_true",
        help="Do not generate the graph-skills retrieval dataset.",
    )
    parser.add_argument(
        "--skip-vectorskills",
        action="store_true",
        help="Do not generate the vector-only retrieval dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_root = args.skills_root
    skillset_name = None
    if args.skills_root == DEFAULT_SKILLS_ROOT and args.skillset_name:
        skills_root = SKILLSETS_ROOT / args.skillset_name
        skillset_name = args.skillset_name

    skills_root = skills_root.resolve()
    gos_workspace = (
        args.gos_workspace.resolve()
        if args.gos_workspace is not None
        else (REPO_ROOT / "data" / "gos_workspace" / f"{(skillset_name or skills_root.name)}_v1").resolve()
    )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_tasks_root = args.tasks_root.resolve()
    tasks = build_task_list(source_tasks_root, args.task)

    # Some older generated graphskills directories were used as source roots for
    # ablation variants. A subset of those directories ended up with zero-byte
    # top-level task files such as instruction.md/task.toml while their original
    # non-ablation counterparts still had valid task payloads. Repair that here by
    # sourcing each task from the canonical sibling directory when available.
    canonical_tasks_root = source_tasks_root
    source_name = source_tasks_root.name
    if source_name == "tasks_graph_skills":
        parent_name = source_tasks_root.parent.name
        for suffix in ("_noppr", "_embedseed"):
            if parent_name.endswith(suffix):
                candidate_root = source_tasks_root.parent.parent / parent_name[: -len(suffix)] / source_name
                if candidate_root.exists():
                    canonical_tasks_root = candidate_root.resolve()
                break

    repaired_tasks: list[Path] = []
    for task in tasks:
        candidate = canonical_tasks_root / task.name
        instruction_path = task / "instruction.md"
        task_toml_path = task / "task.toml"
        needs_repair = (
            canonical_tasks_root != source_tasks_root
            and candidate.exists()
            and (
                not instruction_path.exists()
                or instruction_path.stat().st_size == 0
                or not task_toml_path.exists()
                or task_toml_path.stat().st_size == 0
            )
        )
        repaired = candidate if needs_repair else task
        repaired_tasks.append(canonical_task_source(repaired.resolve()))

    tasks = repaired_tasks
    if args.bundle_path is None:
        raise SystemExit(
            "--bundle-path is required: SE-GoS always starts from a static "
            "(cold-start) bundle, e.g. generated/shared/<coldstart>.json"
        )
    bundle = json.loads(args.bundle_path.read_text(encoding="utf-8"))
    if args.weight_override is not None or args.description_override is not None:
        weight_override = (
            json.loads(args.weight_override.read_text(encoding="utf-8"))
            if args.weight_override is not None
            else None
        )
        description_override = (
            json.loads(args.description_override.read_text(encoding="utf-8"))
            if args.description_override is not None
            else None
        )
        bundle = apply_bundle_overrides(bundle, weight_override, description_override)
    bundle_path = write_bundle(bundle, output_root)
    vector_metadata_path = None
    vector_store_path = None
    retrieval_condition = get_condition(args.retrieval_condition)
    if not args.skip_vectorskills or (not args.skip_graphskills and retrieval_condition.requires_vector_store):
        vector_store_path = write_vector_store(bundle, skills_root, gos_workspace, output_root)
    if not args.skip_vectorskills:
        vector_metadata_path = write_vector_metadata(bundle, output_root)

    allskills_dir = output_root / "tasks_all_skills"
    graphskills_dir = output_root / "tasks_graph_skills"
    vectorskills_dir = output_root / "tasks_vector_skills"

    if not args.skip_allskills:
        ensure_clean_dir(allskills_dir)
        for task in tasks:
            prepare_allskills_task(
                task,
                allskills_dir / task.name,
                skills_root,
            )

    if not args.skip_graphskills:
        ensure_clean_dir(graphskills_dir)
        for task in tasks:
            prepare_graphskills_task(
                task,
                graphskills_dir / task.name,
                bundle_path,
                vector_store_path,
                skills_root,
                gos_workspace,
                bundle_mount=args.bundle_mount,
                retrieval_condition=args.retrieval_condition,
                preserve_verifier=args.preserve_verifier,
            )

    if not args.skip_vectorskills:
        assert vector_metadata_path is not None
        assert vector_store_path is not None
        ensure_clean_dir(vectorskills_dir)
        for task in tasks:
            prepare_vectorskills_task(
                task,
                vectorskills_dir / task.name,
                vector_metadata_path,
                vector_store_path,
                skills_root,
                gos_workspace,
            )

    write_manifest(
        output_root,
        tasks=tasks,
        bundle=bundle,
        allskills_dir=allskills_dir,
        graphskills_dir=graphskills_dir,
        vectorskills_dir=vectorskills_dir,
        skills_root=skills_root,
        skillset_name=skillset_name,
        gos_workspace=gos_workspace,
    )

    print(f"Tasks generated: {len(tasks)}")
    print(f"Graph bundle: {bundle_path}")
    if not args.skip_allskills:
        print(f"All-skills dataset: {allskills_dir}")
    if not args.skip_graphskills:
        print(f"Graph-skills dataset: {graphskills_dir}")
    if not args.skip_vectorskills:
        print(f"Vector-skills dataset: {vectorskills_dir}")


if __name__ == "__main__":
    main()
