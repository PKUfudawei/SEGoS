"""Independent attempts retain evidence without multiplying task weight."""
import json
from unittest.mock import Mock, patch
import pytest
from benchmarks.skillsbench.evo import signals
from segos.evolution import updates as u
from segos.evolution.pipeline import evolve_bundle_serial


def trial(name, used=(), reward=1, queries=(), task="train"):
    return dict(task_name=task, trial_dir=name, used_skills=list(used),
                used_skills_ordered=list(used), reward=reward, queries=list(queries))


def sig(*records):
    return dict(evolution_attempt_policy="all_attempts", attempts=list(records),
                trials={r["task_name"]: r for r in records})


def graph():
    return {"skills": [dict(name=n, skill_dir=n, description="old", inputs=[], outputs=[])
                       for n in ("a", "b", "c")], "edges": []}


def test_unselected_success_without_cross_attempt_or_heldout_edges():
    s = sig(trial("a", ["b"], queries=["q1"]),
            trial("z", ["c"], reward=0, queries=["q2"]),
            trial("heldout", ["c"], queries=["q3"], task="eval"))
    with patch.object(u, "retrieve_bundle", return_value={"seeds": [{"name": "a"}]}):
        edges = u.compute_l1_delta(graph(), s, {"train"})["add"]
        assert [(e["source"], e["target"]) for e in edges] == [("a", "b")]
        assert edges[0]["_count"] == 0.5
        assert edges[0]["weight"] == u.NEW_EDGE_WEIGHT
        s["evolution_attempt_policy"] = "selected_attempt"
        assert u.compute_l1_delta(graph(), s, {"train"})["add"] == []


def test_repeated_attempts_do_not_double_workflow_or_reward_weight():
    a = trial("one", ["b"], queries=["q", "q"])
    b = dict(a, trial_dir="two")
    g = graph()
    g["edges"] = [dict(source="a", target="b", type="workflow", weight=0.6)]
    with patch.object(u, "retrieve_bundle", return_value={"seeds": [{"name": "a"}], "skills": []}):
        for fn in (u.compute_l1_delta, u.compute_l2_delta):
            assert fn(g, sig(a), {"train"}) == fn(g, sig(a, b), {"train"})
        s = sig(a, trial("two", ["c"], reward=0))
        assert u.compute_l2_delta(g, s, {"train"})["delta"]["a|b|workflow"] == pytest.approx(u.ETA / 2)


def test_dependency_stays_within_attempt_and_counts_distinct_tasks():
    with patch("segos.graph.schema_overlap_score", return_value=(1., ["schema"])):
        assert u.compute_dep_edges(graph(), sig(trial("one", ["a"]), trial("two", ["b"])), {"train"}) == []
        records = [trial("one", ["a", "b"]), trial("two", ["a", "b"])]
        assert u.compute_dep_edges(graph(), sig(*records), {"train"})[0]["_count"] == 1
        records.append(trial("one", ["a", "b"], task="other"))
        assert u.compute_dep_edges(graph(), sig(*records), {"train", "other"})[0]["_count"] == 2
        missing = trial("missing", ["a", "b"])
        del missing["used_skills_ordered"]
        assert u.compute_dep_edges(graph(), sig(missing), {"train"}) == []


def test_attenuation_counts_tasks_and_sees_use_in_either_attempt():
    g = graph()
    g["edges"] = [dict(source="a", target="b", type="semantic", weight=.6)]
    with patch.object(u, "retrieve_bundle", return_value={"skills": [{"name": "b"}]}):
        records = [trial("one", queries=["q"]), trial("two", queries=["q"])]
        assert u.compute_l2_delta(g, sig(*records), {"train"})["scale"] == {}
        records.append(trial("one", queries=["q"], task="other"))
        s = sig(*records)
        assert u.compute_l2_delta(g, s, {"train", "other"})["scale"] == {"a|b|semantic": u.PRUNE_SCALE}
        records[0]["used_skills"] = ["b"]
        assert u.compute_l2_delta(g, s, {"train", "other"})["scale"] == {}


def test_l3_uses_unselected_misses_and_guards_other_attempts():
    s = sig(trial("one", ["a"], reward=0, queries=["miss"]),
            trial("two", ["b"], queries=["protect"]))
    for lose_protected in (False, True):
        llm = Mock()
        llm.chat.side_effect = ["feedback", "new", ""]
        def retrieve(bundle, query):
            changed = bundle["skills"][0]["description"] == "new"
            names = (["a"] if changed else []) if query == "miss" else ([] if changed and lose_protected else ["b"])
            return {"skills": [{"name": n} for n in names]}
        with patch.object(u, "retrieve_bundle", side_effect=retrieve):
            result = u.compute_l3_delta(graph(), s, {"train"}, llm=llm, max_workers=1)
        assert result == ({} if lose_protected else {"a": "new"})
        assert llm.chat.call_count == 3


def test_l3_cache_includes_unselected_attempts_and_policy(tmp_path):
    s = sig(trial("one", ["a"], reward=0), trial("two", ["b"]))
    with patch.object(u, "compute_l3_delta", return_value={}) as compute:
        first = evolve_bundle_serial(graph(), s, {"train"}, "L3", tmp_path, benchmark_model="test")
        evolve_bundle_serial(graph(), s, {"train"}, "L3", tmp_path, benchmark_model="test")
        assert compute.call_count == 1
        s["attempts"][0]["queries"] = ["new evidence"]
        evolve_bundle_serial(graph(), s, {"train"}, "L3", tmp_path, benchmark_model="test")
        assert compute.call_count == 2
        s["evolution_attempt_policy"] = "selected_attempt"
        evolve_bundle_serial(graph(), s, {"train"}, "L3", tmp_path, benchmark_model="test")
        assert compute.call_count == 3
        assert first["segos_evolution_protocol"]["training_attempt_count"] == 2


def test_parser_policy_and_raw_evidence_validation(tmp_path):
    job = tmp_path / "job"
    for name in ("one", "two"):
        (job / name).mkdir(parents=True)
    bundle = tmp_path / "graph.json"
    bundle.write_text(json.dumps(graph()))
    records = [trial("one", ["a"]), trial("two", ["b"], reward=0)]
    with patch.object(signals, "extract_trial", side_effect=records):
        s = signals.parse_jobs([job], bundle, attempt_policy="all_attempts", merge_attempts=False)
    assert s["attempts"] == records
    assert len(u.training_attempts(s, {"train"})) == 2
    with pytest.raises(ValueError, match="cannot merge"):
        signals.parse_jobs([job], bundle, attempt_policy="all_attempts", merge_attempts=True)
    with pytest.raises(ValueError, match="selection manifest"):
        signals.parse_jobs([job], bundle, attempt_policy="all_attempts", selection={"train": "one"})
    with pytest.raises(ValueError, match="Duplicate"):
        u.training_attempts(sig(records[0], records[0]), {"train"})
    with pytest.raises(ValueError, match="unmerged raw attempts"):
        u.training_attempts(dict(evolution_attempt_policy="all_attempts", trials={}), {"train"})
    with pytest.raises(ValueError, match="unmerged raw attempts"):
        u.training_attempts(sig(dict(records[0], attempt_dirs=["one", "two"])), {"train"})


def test_static_needs_no_evidence_and_later_round_updates_protocol_metadata(tmp_path):
    assert evolve_bundle_serial(graph(), {}, {"train"}, "static") == graph()
    first = evolve_bundle_serial(graph(), sig(trial("one")), {"train"}, "L2", tmp_path)
    selected = {"trials": {"train": trial("two")}}
    second = evolve_bundle_serial(first, selected, {"train"}, "L2", tmp_path)
    assert second["segos_evolution_protocol"]["attempt_policy"] == "selected_attempt"
