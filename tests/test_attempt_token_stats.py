"""Token means must pool every attempt, not one attempt per task.

`load_job_results` keeps a single entry per task, so averaging its token
counts silently averages one attempt per task and depends on which attempt
the glob reached last.  The paper's *T* column is the mean over all attempts,
which is what `attempt_token_stats` computes; these tests pin that difference.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.skillsbench.evo.analyze import (
    attempt_token_stats,
    load_job_results,
    pooled_reward_stats,
)


def _write_trial(job_dir: Path, task: str, suffix: str, tokens: int, reward: float | None) -> None:
    trial = job_dir / f"{task}__{suffix}"
    trial.mkdir(parents=True)
    payload = {
        "task_name": task,
        "agent_result": {"n_input_tokens": tokens, "n_output_tokens": 1},
        "verifier_result": {"rewards": {"reward": reward}},
    }
    (trial / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_token_mean_pools_every_attempt(tmp_path: Path) -> None:
    # Two tasks, two attempts each: the pooled mean (200) differs from every
    # one-attempt-per-task mean by construction.
    tokens = {"a": (100, 200), "b": (200, 300)}
    for task, (first, second) in tokens.items():
        _write_trial(tmp_path, task, "r1", first, 1.0)
        _write_trial(tmp_path, task, "r2", second, 0.0)

    stats = attempt_token_stats(tmp_path)

    assert stats["n_tokens"] == 4
    assert stats["n_trials"] == 4
    assert stats["T_M"] == (100 + 200 + 200 + 300) / 4 / 1e6


def test_load_job_results_collapses_to_one_entry_per_task(tmp_path: Path) -> None:
    for task in ("a", "b"):
        _write_trial(tmp_path, task, "r1", 100, 1.0)
        _write_trial(tmp_path, task, "r2", 200, 0.0)

    collapsed = load_job_results(tmp_path)

    assert sorted(collapsed) == ["a", "b"]
    # the pooling function is the one that sees both attempts
    assert attempt_token_stats(tmp_path)["n_tokens"] == 2 * len(collapsed)


def test_infrastructure_failures_are_dropped_from_both_stats(tmp_path: Path) -> None:
    _write_trial(tmp_path, "a", "r1", 100, 1.0)
    _write_trial(tmp_path, "a", "r2", 300, None)
    broken = tmp_path / "a__r2" / "result.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["exception_info"] = {"exception_type": "EnvironmentStartTimeoutError"}
    broken.write_text(json.dumps(payload), encoding="utf-8")

    stats = attempt_token_stats(tmp_path)

    assert stats["n_infra"] == 1
    assert stats["T_M"] == 100 / 1e6
    assert pooled_reward_stats(tmp_path)["n_infra"] == 1


def test_missing_job_dir_is_not_fatal(tmp_path: Path) -> None:
    stats = attempt_token_stats(tmp_path / "absent")

    assert stats["T_M"] is None
    assert stats["n_tokens"] == 0
