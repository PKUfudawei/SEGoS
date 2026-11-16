"""Phase A: parse execution signals from a completed harbor job.

For each trial directory under the job we extract:
  - reward / input tokens / runtime        (result.json)
  - retrieval queries                       (graphskills-query exec commands)
  - actually-used skill dirs                (exec commands referencing /opt/graphskills/skills/<dir>/)

Used-skill dirs are only harvested from *tool_call* commands (commands the agent
chose to run), never from instruction/system text or tool outputs, so the
AGENTS.md example paths never leak into the signals. First-use order is recorded
as ``used_skills_ordered``, which the dependency induction uses to certify
prerequisites from ordered usage plus I/O schema overlap.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from benchmarks.skillsbench.evo.llm import benchmark_model_from_jobs

SKILL_PATH_RE = re.compile(r"/opt/graphskills/skills/([^/\s\"'`]+)/")

SIGNAL_FIELDS = (
    "reward",
    "n_input_tokens",
    "n_output_tokens",
    "runtime_sec",
    "queries",
    "used_skills",
    "used_dirs",
)


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    return bundle


def dir_to_name_map(bundle: dict[str, Any]) -> dict[str, str]:
    return {skill["skill_dir"]: skill["name"] for skill in bundle["skills"]}


def _walk_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return trajectory.get("steps") or []


def extract_trial(trial_dir: Path, dir_to_name: dict[str, str]) -> dict[str, Any] | None:
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text(encoding="utf-8"))

    reward = (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
    if reward is None:
        return None
    agent_result = result.get("agent_result") or {}
    n_input_tokens = agent_result.get("n_input_tokens")
    n_output_tokens = agent_result.get("n_output_tokens")

    started = result.get("started_at")
    finished = result.get("finished_at")
    runtime_sec = None
    if started and finished:
        try:
            from datetime import datetime

            fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
            runtime_sec = max(
                0.0,
                (datetime.strptime(finished, fmt) - datetime.strptime(started, fmt)).total_seconds(),
            )
        except ValueError:
            runtime_sec = None

    trajectory_path = trial_dir / "agent" / "trajectory.json"
    queries: list[str] = []
    used_dirs: set[str] = set()
    used_dirs_ordered: list[str] = []
    if trajectory_path.exists():
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        for step in _walk_steps(trajectory):
            for call in step.get("tool_calls") or []:
                if call.get("function_name") != "exec_command":
                    continue
                arguments = call.get("arguments") or {}
                cmd = arguments.get("cmd", "") if isinstance(arguments, dict) else str(arguments)
                if not isinstance(cmd, str):
                    continue
                if "graphskills-query" in cmd:
                    for match in re.finditer(r'graphskills-query\s+("(?:\\.|[^"\\])*"|\S+)', cmd):
                        raw = match.group(1)
                        if raw.startswith('"') and raw.endswith('"'):
                            try:
                                queries.append(json.loads(raw))
                            except json.JSONDecodeError:
                                # Some agent queries embed raw backslashes (e.g.
                                # regex/LaTeX); json.loads rejects them, so keep
                                # the query text as-is minus the wrapping quotes.
                                queries.append(raw[1:-1])
                        else:
                            queries.append(raw.strip('"'))
                for match in SKILL_PATH_RE.finditer(cmd):
                    skill_dir = match.group(1)
                    if skill_dir not in used_dirs:
                        used_dirs.add(skill_dir)
                        used_dirs_ordered.append(skill_dir)  # first-use order

    used_skills = sorted(
        dir_to_name[skill_dir] for skill_dir in used_dirs if skill_dir in dir_to_name
    )
    # First-use order of skill usage (dep induction certifies prerequisites from
    # ordered usage + I/O schema overlap).
    used_skills_ordered = [
        dir_to_name[skill_dir]
        for skill_dir in used_dirs_ordered
        if skill_dir in dir_to_name
    ]
    task_name = result.get("task_name") or trial_dir.name
    return {
        "task_name": task_name,
        "trial_dir": trial_dir.name,
        "reward": float(reward),
        "n_input_tokens": n_input_tokens,
        "n_output_tokens": n_output_tokens,
        "runtime_sec": runtime_sec,
        "queries": queries,
        "used_skills": used_skills,
        "used_skills_ordered": used_skills_ordered,
        "used_dirs": sorted(used_dirs),
    }


def merge_trials(task_name: str, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse a task's attempts into a single trial.

    ``used_skills_ordered`` keeps first-use order across attempts, so an
    attempt that only used a skill the earlier attempt already used adds
    nothing. Reward, token counts and runtime are averaged; queries are
    concatenated. Every attempt of a task is kept in ``attempt_dirs`` so the
    merge is auditable.
    """
    ordered: list[str] = []
    dirs: list[str] = []
    seen_skills: set[str] = set()
    seen_dirs: set[str] = set()
    queries: list[str] = []
    for attempt in attempts:
        for skill in (attempt.get("used_skills_ordered")
                      or attempt.get("used_skills") or []):
            if skill not in seen_skills:
                seen_skills.add(skill)
                ordered.append(skill)
        for skill_dir in attempt.get("used_dirs") or []:
            if skill_dir not in seen_dirs:
                seen_dirs.add(skill_dir)
                dirs.append(skill_dir)
        queries.extend(attempt.get("queries") or [])

    def mean_of(key: str) -> float | None:
        values = [a[key] for a in attempts if a.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "task_name": task_name,
        "trial_dir": attempts[0]["trial_dir"],
        "attempt_dirs": [a["trial_dir"] for a in attempts],
        "reward": mean_of("reward"),
        "n_input_tokens": mean_of("n_input_tokens"),
        "n_output_tokens": mean_of("n_output_tokens"),
        "runtime_sec": mean_of("runtime_sec"),
        "queries": queries,
        "used_skills": sorted(seen_skills),
        "used_skills_ordered": ordered,
        "used_dirs": dirs,
    }


def parse_jobs(
    jobs_roots: list[Path],
    bundle_path: Path,
    out_path: Path | None = None,
    *,
    merge_attempts: bool | None = None,
    selection: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse one or more completed harbor jobs into per-task signals.

    A task that ran ``n_attempts`` times produces that many trial directories;
    by default only the **last** one is kept, so a task contributes one trial.
    Set ``SEGOS_MERGE_ATTEMPTS=1`` (or pass ``merge_attempts=True``) to pool a
    task's attempts instead, which keeps skills used in *any* attempt.

    Raw ``attempts`` are also retained for avoid induction/retraction. Those
    rules must see successful counterevidence and cannot use merged skill sets.
    The per-task selection still controls the other update channels.
    """
    if merge_attempts is None:
        merge_attempts = os.environ.get("SEGOS_MERGE_ATTEMPTS", "") == "1"

    bundle = load_bundle(bundle_path)
    dir_to_name = dir_to_name_map(bundle)

    attempts_by_task: dict[str, list[dict[str, Any]]] = {}
    for jobs_root in jobs_roots:
        root_attempts: dict[str, list[dict[str, Any]]] = {}
        for trial_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            trial = extract_trial(trial_dir, dir_to_name)
            if trial is None:
                continue
            trial["source_job"] = str(jobs_root)
            root_attempts.setdefault(trial["task_name"], []).append(trial)
        # later job roots replace earlier attempts for the same task
        attempts_by_task.update(root_attempts)

    if selection is not None:
        trials = {}
        for task, attempt_id in selection.items():
            matches = [a for a in attempts_by_task.get(task, []) if Path(a["trial_dir"]).name == attempt_id]
            if len(matches) != 1:
                raise ValueError(f"Selection must identify exactly one attempt: {task}/{attempt_id}")
            trials[task] = matches[0]
    elif merge_attempts:
        trials = {task: merge_trials(task, attempts)
                  for task, attempts in attempts_by_task.items()}
    else:
        trials = {task: attempts[-1]
                  for task, attempts in attempts_by_task.items()}

    n_attempts = sum(len(a) for a in attempts_by_task.values())
    if not merge_attempts and n_attempts != len(trials):
        print(f"[signals] collapsing {n_attempts} attempts -> {len(trials)} trials "
              f"(SEGOS_MERGE_ATTEMPTS=1 pools them instead)", flush=True)

    signals = {
        "source_jobs": [str(root) for root in jobs_roots],
        "benchmark_model": benchmark_model_from_jobs(jobs_roots),
        "bundle_path": str(bundle_path),
        "merge_attempts": merge_attempts,
        "attempt_count": n_attempts,
        "trial_count": len(trials),
        "selection": {task: Path(t["trial_dir"]).name for task, t in trials.items()},
        "selection_policy": "manifest" if selection is not None else "lexicographic_last",
        "trials": trials,
        "attempts": [trial for task in sorted(attempts_by_task)
                     for trial in attempts_by_task[task]],
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(signals, indent=2), encoding="utf-8")
    return signals


if __name__ == "__main__":
    jobs = [Path(arg) for arg in sys.argv[1:-2]]
    bundle = Path(sys.argv[-2])
    out = Path(sys.argv[-1])
    result = parse_jobs(jobs, bundle, out)
    print(f"parsed {result['trial_count']} trials from {len(jobs)} jobs")
