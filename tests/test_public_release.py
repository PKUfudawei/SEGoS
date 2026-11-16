import json
from pathlib import Path
from unittest.mock import patch

import pytest

from segos.evolution import updates
from segos.evolution.pipeline import evolve_bundle_serial
from segos.evolution.validation import within_edit_budget
from segos.retrieval.settings import ALFWORLD, SKILLSBENCH, use_settings


def test_budget_profiles_are_separate_and_context_does_not_leak():
    observed = []
    with patch.object(updates.gq, "retrieve", side_effect=lambda b, q, **kw: observed.append(kw) or {}):
        with use_settings(ALFWORLD):
            updates.retrieve_bundle({}, "query")
        updates.retrieve_bundle({}, "query")
    assert {k: observed[0][k] for k in ALFWORLD} == ALFWORLD
    assert {k: observed[1][k] for k in SKILLSBENCH} == SKILLSBENCH


def test_alfworld_budget_applies_to_initial_and_dynamic_retrieval(tmp_path):
    from evaluation.skill import SkillModule
    module = SkillModule.__new__(SkillModule)
    module.bundle_path = tmp_path / "graph.json"
    assert module._effective_top_k("Your task is to: clean cup", 15) == 8
    with patch("evaluation.skill.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps({"skills": [], "summary": "empty"})
        module._run_graphskills_query("clean cup")
    argv = run.call_args.args[0]
    for flag, expected in [("--top-n", "8"), ("--seed-top-k", "5"),
                           ("--max-skill-chars", "2400"), ("--max-context-chars", "12000")]:
        assert argv[argv.index(flag)+1] == expected


def test_description_budget_is_checked_offline():
    assert within_edit_budget("original", "original parse spreadsheet")
    assert not within_edit_budget("original", "original " + "spreadsheet " * 100)


def test_l3_guard_ignores_heldout_queries():
    bundle = {"skills": [{"name": "a", "description": "original"}], "edges": []}
    sig = {"trials": {"train": {"reward": 0, "queries": ["train-q"], "used_skills": ["a"]},
                      "heldout": {"reward": 1, "queries": ["heldout-q"], "used_skills": ["a"]}}}
    queries = []
    def retrieve(b, q):
        queries.append(q)
        return {"skills": [{"name": "a"}]} if b["skills"][0]["description"] != "original" else {"skills": []}
    class LLM:
        def chat(self, *args, **kwargs): return "edited"
    with patch.object(updates, "retrieve_bundle", side_effect=retrieve):
        result = updates.compute_l3_delta(bundle, sig, {"train"}, llm=LLM(), max_workers=1)
    assert result == {"a": "edited"}
    assert "heldout-q" not in queries


def test_l3_combined_edits_recheck_guard():
    bundle = {"skills": [{"name": n, "description": "original"} for n in ["a", "b", "protected"]], "edges": []}
    sig = {"trials": {"train": {"reward": 0, "queries": ["q"], "used_skills": ["a", "b", "protected"]}}}
    def retrieve(b, q):
        edited = [s["name"] for s in b["skills"] if s["description"] != "original"]
        names = edited if len(edited) > 1 else ["protected"] + edited
        return {"skills": [{"name": n} for n in names]}
    class LLM:
        def chat(self, *args, **kwargs): return "edited"
    with patch.object(updates, "retrieve_bundle", side_effect=retrieve):
        result = updates.compute_l3_delta(bundle, sig, {"train"}, llm=LLM(), max_workers=1)
    assert result == {"a": "edited"}


def test_retrieval_budget_changes_invalidate_evolution_cache(tmp_path):
    base = {"skills": [], "edges": []}
    sig = {"trials": {}}
    with patch.object(updates, "compute_l2_delta", return_value={"delta": {}, "scale": {}}) as compute:
        evolve_bundle_serial(base, sig, set(), "L2", tmp_path, retrieval=SKILLSBENCH)
        evolve_bundle_serial(base, sig, set(), "L2", tmp_path, retrieval=SKILLSBENCH)
        assert compute.call_count == 1
        evolve_bundle_serial(base, sig, set(), "L2", tmp_path, retrieval=ALFWORLD)
        assert compute.call_count == 2


def test_chat_client_requires_explicit_credential(monkeypatch):
    from segos.clients.chat import LLMClient
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVO_LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set EVO_LLM_API_KEY"):
        LLMClient(model="model")


def test_public_agent_passes_credentials_via_environment_only(monkeypatch):
    from segos.clients.agent import run_agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("segos.clients.agent.shutil.which", return_value="/usr/bin/codex"), patch("segos.clients.agent.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}})
        assert run_agent([{"role": "user", "content": "test"}], "model")[0] == "ok"
        assert "test-key" not in str(run.call_args)


def test_all_twelve_tables_have_consistent_data_widths():
    root = Path(__file__).resolve().parents[1] / "paper/tables"
    index = json.loads((root / "index.json").read_text())
    assert [r["table_number"] for r in index["tables"]] == list(range(1, 13))
    for row in index["tables"]:
        table = json.loads((root / row["file"]).read_text())
        for panel in table["panels"]:
            assert panel["data"]
            assert len({len(r) for r in panel["data"]}) == 1


def test_alfworld_resume_rejects_changed_inputs(tmp_path):
    from benchmarks.alfworld.run_pipeline import check_resume
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"model": "a", "graph_sha256": "old"}))
    check_resume(manifest, {"model": "a", "graph_sha256": "old"})
    with pytest.raises(ValueError, match="inputs changed"):
        check_resume(manifest, {"model": "a", "graph_sha256": "new"})


def test_alfworld_signals_reject_overlapping_attempts(tmp_path, monkeypatch):
    from benchmarks.alfworld.alfworld_signals import main
    for attempt in (1, 2):
        folder = tmp_path / str(attempt)
        folder.mkdir()
        (folder / "idx_1.json").write_text(json.dumps({"name": "same episode", "reward": 0}))
    monkeypatch.setattr("sys.argv", ["alfworld_signals", "--results-dir", str(tmp_path),
                                    "--out", str(tmp_path / "signals.json")])
    with pytest.raises(ValueError, match="Multiple attempts"):
        main()


def test_graph_updates_refresh_all_edge_counts():
    from segos.graph import apply_bundle_overrides
    graph = {"skills": [], "edges": [{"source": "a", "target": "b", "type": "semantic", "weight": 0.6}],
             "metadata": {"total_edges": 123, "edge_count": 123, "semantic_edges": 123}}
    updated = apply_bundle_overrides(graph, {"remove": ["a|b|semantic"], "add": [
        {"source": "b", "target": "a", "type": "workflow", "weight": 0.6}]}, None)
    assert updated["metadata"]["edge_count"] == updated["metadata"]["total_edges"] == 1
    assert updated["metadata"]["semantic_edges"] == 0
    assert updated["metadata"]["workflow_edges"] == 1
    assert graph["metadata"]["total_edges"] == 123
