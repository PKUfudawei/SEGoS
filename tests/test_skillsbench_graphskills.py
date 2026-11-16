from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR_PATH = REPO_ROOT / "benchmarks" / "skillsbench" / "graphskills_benchmark.py"
RUNTIME_PATH = (
    REPO_ROOT / "benchmarks" / "skillsbench" / "graphskills_assets" / "query.py"
)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_bundle() -> dict:
    """Minimal SE-GoS bundle: a dependency chain read_csv -> analyze_trend -> render_chart."""
    names = ["read_csv", "analyze_trend", "render_chart"]
    return {
        "metadata": {"skill_count": 3, "edge_count": 2},
        "skills": [
            {
                "name": name,
                "description": f"{name} step of the reporting pipeline",
                "inputs": [],
                "outputs": [],
                "one_line_capability": name,
                "rendered_snippet": f"# {name}\nUsage.",
                "raw_content": f"# {name}\nUsage.",
                "source_path": f"/opt/graphskills/skills/{name}/SKILL.md",
            }
            for name in names
        ],
        "edges": [
            {
                "source": "read_csv",
                "target": "analyze_trend",
                "type": "dependency",
                "weight": 0.9,
                "confidence": 0.9,
                "description": "produces the input consumed by the next step",
            },
            {
                "source": "analyze_trend",
                "target": "render_chart",
                "type": "dependency",
                "weight": 0.9,
                "confidence": 0.9,
                "description": "produces the input consumed by the next step",
            },
        ],
    }


def test_graphskills_runtime_retrieval_respects_context_budget():
    runtime = _load_module("skillsbench_graph_runtime", RUNTIME_PATH)

    bundle = _fixture_bundle()
    result = runtime.retrieve(
        bundle,
        "Analyze sales csv trends and make a chart",
        top_n=3,
        seed_top_k=3,
        max_skill_chars=160,
        max_context_chars=520,
        seed_mode="lexical",
        propagation_mode="ppr",
        vector_store_path=None,
    )

    retrieved_names = [skill["name"] for skill in result["skills"]]
    assert "analyze_trend" in retrieved_names
    assert "render_chart" in retrieved_names
    assert len(result["rendered_context"]) <= 520
    assert result["relations"]


def test_graphskills_forward_only_transition_omits_reverse_edges():
    runtime = _load_module("skillsbench_graph_runtime_forward_only", RUNTIME_PATH)
    skills = [{"name": "producer"}, {"name": "consumer"}]
    edges = [
        {
            "source": "producer",
            "target": "consumer",
            "type": "dependency",
            "weight": 1.0,
        }
    ]

    full = runtime.build_transition(skills, edges, reverse_mode="full")
    forward_only = runtime.build_transition(skills, edges, reverse_mode="none")

    assert full[1] == {0: 1.0}
    assert forward_only[1] == {1: 1.0}
    assert forward_only[0] == {1: 1.0}


def test_graphskills_one_hop_adds_incoming_dependency_prerequisite():
    runtime = _load_module("skillsbench_graph_runtime_one_hop", RUNTIME_PATH)
    skills = [
        {
            "name": "read_csv",
            "description": "Read CSV data.",
            "source_path": "/skills/read_csv/SKILL.md",
            "raw_content": "Read CSV data.",
        },
        {
            "name": "analyze_trend",
            "description": "Analyze a dataset.",
            "source_path": "/skills/analyze_trend/SKILL.md",
            "raw_content": "Analyze a dataset.",
        },
    ]
    edges = [
        {
            "source": "read_csv",
            "target": "analyze_trend",
            "type": "dependency",
            "weight": 1.0,
        }
    ]

    ranked = runtime.rank_one_hop(
        skills,
        edges,
        seed_entries=[(1, 1.0, 1)],
        max_skill_chars=200,
    )

    assert [skill["name"] for skill in ranked] == ["analyze_trend", "read_csv"]
    assert ranked[1]["semantic_rank"] is None


def test_no_graph_fills_matched_top_n_beyond_ppr_seed_count():
    runtime = _load_module("skillsbench_graph_runtime_matched_none", RUNTIME_PATH)
    skills = [
        {
            "name": f"skill-{index}",
            "description": f"shared query capability {index}",
            "source_path": f"/skills/skill-{index}/SKILL.md",
            "raw_content": f"shared query capability {index}",
        }
        for index in range(6)
    ]
    bundle = {"metadata": {}, "skills": skills, "edges": []}

    result = runtime.retrieve(
        bundle,
        "shared query capability",
        top_n=5,
        seed_top_k=4,
        max_skill_chars=100,
        max_context_chars=5000,
        seed_mode="lexical",
        propagation_mode="none",
        reverse_mode="full",
    )

    assert len(result["skills"]) == 5
    assert result["budget"]["seed_top_k"] == 4


def test_build_task_list_accepts_current_task_markdown_schema(tmp_path):
    generator = _load_module("skillsbench_graph_builder_task_md", GENERATOR_PATH)
    task = tmp_path / "dialogue-parser"
    task.mkdir()
    (task / "task.md").write_text("---\nschema_version: '1.3'\n---\nSolve it.\n")

    tasks = generator.build_task_list(tmp_path, ["dialogue-parser"])

    assert tasks == [task]


def test_normalize_task_layout_converts_current_skillsbench_schema(tmp_path):
    generator = _load_module("skillsbench_graph_builder_normalize", GENERATOR_PATH)
    task = tmp_path / "dialogue-parser"
    (task / "verifier").mkdir(parents=True)
    (task / "oracle").mkdir()
    (task / "task.md").write_text(
        """---
schema_version: '1.3'
metadata:
  author_name: reviewer
  difficulty: easy
  category: software-engineering
  tags: [parsing, json]
verifier:
  timeout_sec: 900.0
agent:
  timeout_sec: 800.0
environment:
  build_timeout_sec: 600.0
  cpus: 2
  memory_mb: 4096
  storage_mb: 10240
  gpus: 0
  network_mode: public
---

Implement the parser.
""",
        encoding="utf-8",
    )
    (task / "verifier" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (task / "oracle" / "solve.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    generator.normalize_task_layout(task)

    assert (task / "instruction.md").read_text(
        encoding="utf-8"
    ) == "Implement the parser.\n"
    task_toml = (task / "task.toml").read_text(encoding="utf-8")
    assert 'author_name = "reviewer"' in task_toml
    assert "timeout_sec = 900.0" in task_toml
    assert "memory_mb = 4096" in task_toml
    assert "allow_internet = true" in task_toml
    assert (task / "tests" / "test.sh").exists()
    assert (task / "solution" / "solve.sh").exists()


def test_graphskills_compose_without_legacy_harbor_variables(tmp_path):
    from harbor.environments.docker.compose_env import ComposeInfraEnvVars

    if not shutil.which("docker"):
        pytest.skip("Docker Compose is required; no daemon needed")
    version = subprocess.run(["docker", "compose", "version"], capture_output=True)
    if version.returncode:
        pytest.skip("Docker Compose is required; no daemon needed")

    generator = _load_module("skillsbench_graph_builder_compose", GENERATOR_PATH)
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    skills = tmp_path / "skills"
    skills.mkdir()
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    compose = env_dir / "docker-compose.yaml"
    compose.write_text(generator.render_compose_template(
        generator.GOS_TEMPLATE_DIR / "docker-compose.yaml",
        destination_env_dir=env_dir,
        skills_root=skills,
        bundle_mount=bundle,
        mount_prebuilt_workspace=False,
    ))

    # Reproduce Harbor's default path: infra variables without legacy log binds.
    env = {"PATH": os.environ["PATH"]}
    env.update(ComposeInfraEnvVars(
        main_image_name="segos-compose-test", context_dir=str(env_dir),
        cpus=1, memory="1G",
    ).to_env_dict())
    result = subprocess.run(
        ["docker", "compose", "--project-name", "segos-compose-test",
         "--env-file", os.devnull, "-f", str(compose), "config", "--format", "json"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert "variable is not set" not in result.stderr
    main = json.loads(result.stdout)["services"]["main"]
    assert main["image"] == "segos-compose-test"
    assert main["build"]["context"] == str(env_dir)
    mounts = {mount["target"]: mount for mount in main["volumes"]}
    assert set(mounts) == {"/opt/graphskills/skills", "/opt/graphskills/bundle.json"}
    for target, source in (("/opt/graphskills/skills", skills),
                           ("/opt/graphskills/bundle.json", bundle)):
        assert mounts[target]["type"] == "bind"
        assert mounts[target]["source"] == str(source)
        assert mounts[target]["read_only"] is True
    assert "TEST_DIR" not in main["environment"]
