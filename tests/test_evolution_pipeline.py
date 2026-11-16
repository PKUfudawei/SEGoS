from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLSBENCH = REPO_ROOT / "benchmarks" / "skillsbench"
if str(SKILLSBENCH) not in sys.path:
    sys.path.insert(0, str(SKILLSBENCH))

from evo import levers  # noqa: E402


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        return ("gradient", "first", "second\nthird")[self.calls - 1]


class EvolutionPipelineTests(unittest.TestCase):
    def test_node_candidate_must_reach_top_three_itself(self):
        bundle = {"skills": [{"name": "target", "description": "original"}], "edges": []}
        signals = {
            "trials": {
                "task": {"reward": 0.0, "queries": ["q1", "q2"], "used_skills": ["target"]}
            }
        }

        def retrieve(candidate_bundle, query):
            description = candidate_bundle["skills"][0]["description"]
            if description == "first" and query == "q1":
                return {"skills": [{"name": "target"}]}
            if description == "second":
                return {"skills": [{"name": f"other-{i}"} for i in range(3)]
                        + [{"name": "target"}]}
            return {"skills": []}

        with patch.object(levers, "retrieve_bundle", side_effect=retrieve):
            result = levers.compute_l3_delta(
                bundle, signals, {"task"}, llm=FakeLLM(), max_workers=1
            )

        # The original is absent from even an empty bundle, so it is a miss.
        # The second edit hits more queries but never reaches the top three.
        self.assertEqual(result, {"target": "first"})

    def test_alfworld_uses_skillsbench_serial_pipeline(self):
        calls = []
        fake_pipeline = types.ModuleType("segos.evolution.pipeline")

        def evolve(base, signals, tasks, condition, out_dir=None, *, benchmark_model=None, retrieval=None):
            calls.append((base, signals, tasks, condition, out_dir, benchmark_model))
            return {"skills": [{"name": "target", "description": "edited"}], "edges": []}

        fake_pipeline.evolve_bundle_serial = evolve
        with patch.dict(sys.modules, {"segos.evolution.pipeline": fake_pipeline}):
            script = load_script(
                "alfworld_evolve_under_test",
                REPO_ROOT / "benchmarks" / "alfworld" / "alfworld_evolve.py",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "base.json"
            signals_path = root / "signals.json"
            output = root / "bundle_L1L2L3.json"
            source.write_text(json.dumps({"skills": [], "edges": []}), encoding="utf-8")
            signals_path.write_text(json.dumps({"trials": {"task": {}}}), encoding="utf-8")
            with patch.object(sys, "argv", [
                "alfworld_evolve.py", "--signals", str(signals_path),
                "--bundle", str(source), "--levers", "L3,L1,L2", "--out", str(output),
                "--model", "gpt-5.2-codex",
            ]):
                script.main()
            generated = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(calls[0][2:4], ({"task"}, "L1L2L3"))
        self.assertEqual(calls[0][5], "gpt-5.2-codex")
        self.assertEqual(generated["skills"][0]["description"], "edited")
        self.assertEqual(generated["segos_evolution_pipeline"], script.PIPELINE_VERSION)

    def test_alfworld_signals_preserve_first_use_order(self):
        script = load_script(
            "alfworld_signals_under_test",
            REPO_ROOT / "benchmarks" / "alfworld" / "alfworld_signals.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = root / "jobs"
            jobs.mkdir()
            (jobs / "idx_0.json").write_text(json.dumps({
                "name": "task", "reward": 1.0,
                "runtime_skill_events": [
                    {"trigger": "agent_request:read_skill", "skill_names": ["b"]},
                    {"trigger": "agent_request:read_skill", "skill_names": ["a", "b"]},
                ],
            }), encoding="utf-8")
            output = root / "signals.json"
            with patch.object(sys, "argv", [
                "alfworld_signals.py", "--results-dir", str(jobs), "--out", str(output),
            ]):
                script.main()
            generated = json.loads(output.read_text(encoding="utf-8"))
            trial = generated["trials"]["task"]

        self.assertEqual(generated["signal_schema_version"], 2)
        self.assertEqual(trial["used_skills"], ["a", "b"])
        self.assertEqual(trial["used_skills_ordered"], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
