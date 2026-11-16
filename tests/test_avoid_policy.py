"""Avoid induction and retraction share attempt-level successful counterevidence."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/skillsbench"))
from evo import levers, signals
from evo.run_full87_experiments import evolve_bundle_serial


def trial(task="train", reward=0.0, used=("a", "b"), attempt="one"):
    return dict(task_name=task, reward=reward, used_skills=list(used),
                queries=[], trial_dir=attempt)


def bundle():
    return {"skills": [], "edges": [
        dict(source="b", target="a", type="avoid", weight=0.0),
        dict(source="a", target="c", type="avoid", weight=0.0),
    ]}


def test_one_success_retracts_and_vetoes_reinduction_even_if_selected_attempt_fails():
    sig = {"trials": {"train": trial()},
           "attempts": [trial(), trial(reward=0.9, attempt="two")]}
    delta = levers.compute_l1_delta(bundle(), sig, {"train"})
    assert delta["remove"] == ["b|a|avoid"]
    assert delta["retractions"][0]["successful_task_count"] == 1
    assert delta["retractions"][0]["successful_attempts"][0]["trial_dir"] == "two"
    assert levers.compute_avoid_edges(sig, {"train"}) == []


def test_failures_and_nonuse_do_not_retract():
    sig = {"trials": {"train": trial()}}
    assert levers.compute_l1_delta(bundle(), sig, {"train"})["remove"] == []
    sig["trials"]["train"] = trial(reward=1.0, used=("a",))
    assert levers.compute_l1_delta(bundle(), sig, {"train"})["remove"] == []


def test_attempts_do_not_create_cross_attempt_couse():
    # The merged/selected view must not be used for avoid evidence.
    sig = {"trials": {"train": trial()},
           "attempts": [trial(used=("a",)), trial(used=("b",), attempt="two")]}
    assert levers.compute_avoid_edges(sig, {"train"}) == []


def test_merged_legacy_signals_require_reextraction():
    sig = {"trials": {"train": dict(trial(), attempt_dirs=["one", "two"])}}
    with pytest.raises(ValueError, match="requires raw attempts"):
        levers.compute_avoid_edges(sig, {"train"})


def test_failed_tasks_are_distinct_and_heldout_success_does_not_leak():
    sig = {"trials": {"train": trial()}, "attempts": [
        trial(), trial(attempt="two"), trial(task="heldout", reward=1.0)]}
    edges = levers.compute_avoid_edges(sig, {"train"})
    assert len(edges) == 1
    assert edges[0]["fail_count"] == 1
    assert edges[0]["failed_attempt_count"] == 2
    assert edges[0]["weight"] == 0
    assert levers.compute_l1_delta(bundle(), sig, {"train"})["remove"] == []
    with patch.object(levers, "AVOID_FAIL_MIN", 2):
        assert levers.compute_avoid_edges(sig, {"train"}) == []


def test_historical_success_prevents_later_readdition():
    sig = {"trials": {"train": trial()}, "attempts": [trial()],
           "avoid_trials": [dict(trial(reward=1.0), round=0), dict(trial(), round=1)]}
    assert levers.compute_avoid_edges(sig, {"train"}) == []
    assert levers.compute_l1_delta(bundle(), sig, {"train"})["remove"] == ["b|a|avoid"]


def test_navigation_relation_blocks_new_avoid_in_either_direction():
    sig = {"trials": {"train": trial()}}
    graph = {"edges": [dict(source="b", target="a", type="workflow", weight=0.6)]}
    assert levers.compute_avoid_edges(sig, {"train"}, graph) == []


def test_parser_keeps_raw_attempts_even_when_task_view_is_merged(tmp_path):
    job = tmp_path / "job"
    for name in ("one", "two"):
        (job / name).mkdir(parents=True)
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"skills": [], "edges": []}))
    records = [trial(used=("a",)), trial(used=("b",), attempt="two")]
    with patch.object(signals, "extract_trial", side_effect=records):
        sig = signals.parse_jobs([job], graph, merge_attempts=True)
    assert sig["trials"]["train"]["used_skills"] == ["a", "b"]
    assert [t["used_skills"] for t in sig["attempts"]] == [["a"], ["b"]]
    assert levers.compute_avoid_edges(sig, {"train"}) == []


def test_serial_pipeline_invalidates_cache_when_success_evidence_changes(tmp_path):
    sig = {"trials": {"train": trial()}, "attempts": [trial()]}
    with patch.object(levers, "compute_dep_edges", return_value=[]):
        first = evolve_bundle_serial(bundle(), sig, {"train"}, "L1", tmp_path)
        assert any({e["source"], e["target"]} == {"a", "b"} for e in first["edges"])
        assert len(list(tmp_path.glob("delta_public-serial-v1_L1_L1_*.json"))) == 1
        with patch.object(levers, "compute_l1_delta", side_effect=AssertionError("cache miss")):
            assert evolve_bundle_serial(bundle(), sig, {"train"}, "L1", tmp_path) == first
        sig["attempts"].append(trial(reward=1.0, attempt="two"))
        second = evolve_bundle_serial(bundle(), sig, {"train"}, "L1", tmp_path)
    assert not any({e["source"], e["target"]} == {"a", "b"} for e in second["edges"])


def test_serial_cache_respects_training_scope_and_avoid_threshold(tmp_path):
    sig = {"trials": {"train": trial()}, "attempts": [trial()]}
    graph = {"skills": [], "edges": []}
    with patch.object(levers, "compute_dep_edges", return_value=[]):
        first = evolve_bundle_serial(graph, sig, {"train"}, "L1", tmp_path)
        assert len(first["edges"]) == 1
        with patch.object(levers, "AVOID_FAIL_MIN", 2):
            assert evolve_bundle_serial(graph, sig, {"train"}, "L1", tmp_path)["edges"] == []
        assert evolve_bundle_serial(graph, sig, set(), "L1", tmp_path)["edges"] == []


def test_weight_cache_reuses_delta_and_invalidates_changed_rate(tmp_path):
    graph = {"skills": [], "edges": [
        dict(source="a", target="b", type="workflow", weight=0.6)]}
    sig = {"trials": {"train": trial(reward=1.0, used=("b",))}}
    first = evolve_bundle_serial(graph, sig, {"train"}, "L2", tmp_path)
    assert first["edges"][0]["weight"] == pytest.approx(0.7)
    with patch.object(levers, "compute_l2_delta", side_effect=AssertionError("cache miss")):
        assert evolve_bundle_serial(graph, sig, {"train"}, "L2", tmp_path) == first
    with patch.object(levers, "ETA", 0.2):
        second = evolve_bundle_serial(graph, sig, {"train"}, "L2", tmp_path)
    assert second["edges"][0]["weight"] == pytest.approx(0.8)


def test_extraction_finds_library_paths_outside_retrieval_and_ignores_outputs(tmp_path):
    (tmp_path / 'agent').mkdir()
    (tmp_path / 'result.json').write_text(json.dumps({
        'task_name': 'train', 'verifier_result': {'rewards': {'reward': 1.0}}}))
    # No retrieval query or returned bundle is needed to recognize a known path.
    (tmp_path / 'agent/trajectory.json').write_text(json.dumps({'steps': [
        {'tool_calls': [
            {'function_name': 'exec_command', 'arguments': {
                'cmd': 'cat /opt/graphskills/skills/b/SKILL.md'}},
            {'function_name': 'read_file', 'arguments': {
                'path': '/opt/graphskills/skills/a/SKILL.md'}},
        ], 'output': '/opt/graphskills/skills/a/SKILL.md'},
        {'tool_calls': [{'function_name': 'exec_command', 'arguments': {
            'cmd': 'cat /opt/graphskills/skills/c/SKILL.md /opt/graphskills/skills/b/SKILL.md'}}]},
    ]}))
    record = signals.extract_trial(tmp_path, {'a': 'A', 'b': 'B', 'c': 'C'})
    assert record['queries'] == []
    assert record['used_skills'] == ['B', 'C']
    assert record['used_skills_ordered'] == ['B', 'C']
