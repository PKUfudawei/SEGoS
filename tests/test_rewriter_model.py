from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

SKILLSBENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "skillsbench"
if str(SKILLSBENCH) not in sys.path:
    sys.path.insert(0, str(SKILLSBENCH))

from evo import levers, signals
from evo.llm import benchmark_model_from_jobs, resolve_rewriter_model
from evo.run_full87_experiments import evolve_bundle_serial


def test_rewriter_uses_benchmark_model_instead_of_independent_env():
    sig = {"benchmark_model": "gpt-5.2-codex", "trials": {}}
    with patch.dict(os.environ, {"EVO_LLM_MODEL": "unrelated-model"}):
        with patch.object(levers, "LLMClient") as client:
            assert levers.compute_l3_delta({"skills": [], "edges": []}, sig, set()) == {}
    client.assert_called_once_with(model="gpt-5.2-codex")


def test_source_job_model_is_preserved_in_signals(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "config.json").write_text(json.dumps({
        "agents": [{"model_name": "minimax-m2.7"}],
    }))
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"skills": [], "edges": []}))
    sig = signals.parse_jobs([job], bundle)
    assert sig["benchmark_model"] == "minimax-m2.7"
    assert resolve_rewriter_model(sig) == "minimax-m2.7"
    # Legacy signals can still resolve the model from the original job.
    assert resolve_rewriter_model({"source_jobs": [str(job)]}) == "minimax-m2.7"


def test_unknown_model_does_not_silently_use_environment_default():
    with patch.dict(os.environ, {"EVO_LLM_MODEL": "unrelated-model"}):
        with pytest.raises(ValueError, match="Cannot determine"):
            resolve_rewriter_model({"trials": {}})
    assert resolve_rewriter_model({}, "gpt-5.2-codex") == "gpt-5.2-codex"


def test_conflicting_or_mixed_model_records_fail(tmp_path):
    with pytest.raises(ValueError, match="disagrees"):
        resolve_rewriter_model({"benchmark_model": "minimax-m2.7"}, "gpt-5.2-codex")
    jobs = []
    for i, model in enumerate(("minimax-m2.7", "gpt-5.2-codex")):
        job = tmp_path / str(i)
        job.mkdir()
        (job / "config.json").write_text(json.dumps({"agents": [{"model_name": model}]}))
        jobs.append(job)
    with pytest.raises(ValueError, match="mix benchmark models"):
        benchmark_model_from_jobs(jobs)


def test_l3_cache_is_scoped_to_model_and_actual_inputs(tmp_path):
    # Graph merging is independent of model selection/cache routing.
    merge = types.ModuleType("graphskills_benchmark")
    merge.apply_bundle_overrides = lambda bundle, _edges, _descriptions: bundle
    base = {"skills": [], "edges": []}
    sig = {"benchmark_model": "minimax-m2.7", "trials": {}}
    with patch.dict(sys.modules, {"graphskills_benchmark": merge}):
        with patch.object(levers, "compute_l3_delta", return_value={}) as rewrite:
            first = evolve_bundle_serial(base, sig, set(), "L3", tmp_path)
            evolve_bundle_serial(base, sig, set(), "L3", tmp_path)
            assert rewrite.call_count == 1
            assert first["segos_rewriter_model"] == "minimax-m2.7"
            changed = {"benchmark_model": "gpt-5.2-codex", "trials": {}}
            second = evolve_bundle_serial(base, changed, set(), "L3", tmp_path)
            assert rewrite.call_count == 2
            assert second["segos_rewriter_model"] == "gpt-5.2-codex"
            changed["trials"] = {"new-task": {"reward": 1}}
            evolve_bundle_serial(base, changed, {"new-task"}, "L3", tmp_path)
            assert rewrite.call_count == 3
    metadata = [json.loads(p.read_text()) for p in tmp_path.glob("*.metadata.json")]
    assert {m["rewriter_model"] for m in metadata} == {"minimax-m2.7", "gpt-5.2-codex"}


def test_structural_only_evolution_needs_no_llm_model():
    merge = types.ModuleType("graphskills_benchmark")
    merge.apply_bundle_overrides = lambda bundle, _edges, _descriptions: bundle
    with patch.dict(sys.modules, {"graphskills_benchmark": merge}):
        result = evolve_bundle_serial({"skills": [], "edges": []}, {"trials": {}}, set(), "L2")
    assert "segos_rewriter_model" not in result
