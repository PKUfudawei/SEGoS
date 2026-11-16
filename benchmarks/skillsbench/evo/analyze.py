"""Analyze the 8-condition factorial results and produce the paper table inputs.

Outputs to evo_data/analysis_summary.json:
  - per-condition: mean reward, pass@1, mean input/output tokens, mean runtime
  - per-task paired comparison vs static
  - lever main effects (containing-lever vs not, averaged over the 4 cells each)
"""
from __future__ import annotations

import json
import glob
import math
import os
import sys
from pathlib import Path
from typing import Any

SKILLSBENCH = Path(__file__).resolve().parent.parent

CONDITIONS = ["static", "L1", "L2", "L3", "L1L2", "L1L3", "L2L3", "L1L2L3"]
LEVERS = ["L1", "L2", "L3"]

# SkillDAG-aligned reward accounting (analysis/score_skillsbench_gos.py):
# startup/build infrastructure failures are excluded from the R denominator;
# agent timeouts / model failures / missing verifier reward stay in with 0.
INFRA_EXCEPTION_TYPES = {"AgentSetupTimeoutError", "EnvironmentStartTimeoutError"}
INFRA_MESSAGE_MARKERS = (
    "SkillDAG runtime failed to start",
    "Docker compose command failed",
    "docker compose command failed",
    "dockerfile parse error",
    "failed to solve:",
    "failed to resolve reference",
    "connect: connection refused",
    "RPC failed;",
    "fetch-pack: unexpected disconnect",
    "fatal: early EOF",
)

# Harbor-internal crash signatures: these attempts never reached the verifier
# because the harness itself died (e.g. the orchestrator process was killed
# mid-run).  They carry a result.json without any verifier reward, and are
# infrastructure failures rather than agent errors: count them as infra
# (excluded from the denominator) and let the runner re-schedule them.
HARBOR_CRASH_MARKERS = (
    "harbor/environments/base.py",
    "harbor/trial/trial.py",
    "created in a different Context",
    "exec_env_overlays",
    "GeneratorExit",
)


def is_harbor_crash(data: dict[str, Any]) -> bool:
    # Only attempts that never produced a verifier reward can be harness kills.
    # A stray GeneratorExit/teardown traceback on an otherwise scored attempt is
    # not a crash artifact -- that attempt stays in the scored pool.
    verifier = data.get("verifier_result")
    if isinstance(verifier, dict):
        reward = (verifier.get("rewards") or {}).get("reward")
        if reward is not None:
            return False
    ei = data.get("exception_info") or {}
    if not isinstance(ei, dict):
        return False
    text = " ".join(
        str(ei.get(k) or "") for k in ("exception_type", "exception_message", "exception_traceback")
    )
    return any(m in text for m in HARBOR_CRASH_MARKERS)


def is_infra_exception(data: dict[str, Any]) -> bool:
    ei = data.get("exception_info") or {}
    if not isinstance(ei, dict):
        return False
    etype = ei.get("exception_type") or ""
    emsg = ei.get("exception_message") or ""
    return (
        is_harbor_crash(data)
        or etype in INFRA_EXCEPTION_TYPES
        or any(m in emsg for m in INFRA_MESSAGE_MARKERS)
    )


def as_reward(value: Any) -> float | None:
    """Parse a verifier reward without binarizing or rounding it.

    SkillsBench/Harbor reward files may contain either binary rewards or a
    task-level fractional score. Keep the verifier's value intact so the
    scorer can aggregate either convention, matching SkillDAG's scorer.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def reward_from_trial(data: dict[str, Any], trial_dir: Path) -> float | None:
    """Read the raw reward from result.json, then Harbor reward artifacts."""
    verifier = data.get("verifier_result") or {}
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if isinstance(rewards, dict):
        reward = as_reward(rewards.get("reward"))
    else:
        reward = as_reward(rewards)
    if reward is not None:
        return reward

    reward_txt = trial_dir / "verifier" / "reward.txt"
    if reward_txt.exists():
        try:
            reward = as_reward(reward_txt.read_text(encoding="utf-8"))
        except OSError:
            reward = None
        if reward is not None:
            return reward

    reward_json = trial_dir / "verifier" / "reward.json"
    if reward_json.exists():
        try:
            payload = json.loads(reward_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(payload, dict):
            return as_reward(payload.get("reward"))
    return None


def pooled_reward_stats(job_dir: Path) -> dict[str, Any]:
    """SkillDAG-aligned pooled stats over *every* attempt trial (no per-task
    collapse). Returns mean raw verifier reward over scored attempts
    (error/infra handling as SkillDAG), scored/infra/pending counts, reward
    sum, exact pass/fail and fractional-reward counts. Rewards are not
    binarized or rounded here; the verifier defines their scale.
    """
    scored: list[float] = []
    n_infra = 0
    n_pending = 0
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir() and "__" in p.name):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            n_pending += 1
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            n_pending += 1
            continue
        if is_infra_exception(data):
            n_infra += 1
            continue
        reward = reward_from_trial(data, trial_dir)
        scored.append(reward if reward is not None else 0.0)
    reward_sum = sum(scored)
    n_exact_pass = sum(reward == 1.0 for reward in scored)
    n_exact_fail = sum(reward == 0.0 for reward in scored)
    n_fractional = len(scored) - n_exact_pass - n_exact_fail
    return {
        "R": mean(scored) if scored else None,
        "n_scored": len(scored),
        "n_infra": n_infra,
        "n_pending": n_pending,
        "reward_sum": reward_sum,
        "n_exact_pass": n_exact_pass,
        "n_exact_fail": n_exact_fail,
        "n_fractional": n_fractional,
        "rewards": scored,
    }



def load_signals(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_job_results(job_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-task view of a job: exactly one entry per task.

    A task runs several attempts, each in its own ``<task>__<suffix>`` trial
    directory.  This function keeps one of them per task, which is what
    per-task alignment needs, but it means the returned rewards and token
    counts are *one* attempt per task, not the pooled values the paper
    reports.  Use :func:`pooled_reward_stats` for reward means and
    :func:`attempt_token_stats` for token means; averaging the token fields
    here would average 87 samples instead of 174.
    """
    results: dict[str, dict[str, Any]] = {}
    for f in sorted(glob.glob(str(job_dir / "*__*" / "result.json"))):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        tn = data.get("task_name") or os.path.basename(os.path.dirname(f))
        agent = data.get("agent_result") or {}
        verifier = data.get("verifier_result") or {}
        reward = verifier.get("rewards", {}).get("reward")
        results[tn] = {
            # infrastructure errors / missing verifier reward are scored as 0
            "reward": reward if reward is not None else 0.0,
            "n_input_tokens": agent.get("n_input_tokens"),
            "n_output_tokens": agent.get("n_output_tokens"),
            "exception": data.get("exception_info"),
        }
    return results


def attempt_token_stats(job_dir: Path) -> dict[str, Any]:
    """Token statistics over *every* attempt trial of a job (no per-task collapse).

    This is the accounting the paper's *T* column uses: mean input tokens per
    attempt, over all scored attempts of the job.  Infrastructure failures are
    dropped, matching the reward accounting in :func:`pooled_reward_stats`.
    """
    inputs: list[int] = []
    outputs: list[int] = []
    n_trials = 0
    n_infra = 0
    n_pending = 0
    if not job_dir.is_dir():
        return {"T_M": None, "mean_output_tokens": None, "inputs": inputs,
                "outputs": outputs, "n_tokens": 0, "n_trials": 0,
                "n_infra": 0, "n_pending": 0}
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir() and "__" in p.name):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            n_pending += 1
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            n_pending += 1
            continue
        if is_infra_exception(data):
            n_infra += 1
            continue
        agent = data.get("agent_result") or {}
        n_trials += 1
        if agent.get("n_input_tokens") is not None:
            inputs.append(int(agent["n_input_tokens"]))
        if agent.get("n_output_tokens") is not None:
            outputs.append(int(agent["n_output_tokens"]))
    return {
        "T_M": (mean(inputs) / 1e6) if inputs else None,
        "mean_output_tokens": mean(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "n_tokens": len(inputs),
        "n_trials": n_trials,
        "n_infra": n_infra,
        "n_pending": n_pending,
    }


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    signals = load_signals(SKILLSBENCH / "evo_data" / "signals.json")
    split = load_split(SKILLSBENCH / "evo_data" / "split.json")
    eval_tasks = split["eval"]

    # static comes from the static job results (consistent with the condition
    # jobs below); falls back to signals.json if the static job is absent.
    static_job: dict[str, dict[str, Any]] = {}
    for root in ("jobs", "jobs_old"):
        static_job.update(load_job_results(SKILLSBENCH / root / "graphskills-codex-n1000"))
    if static_job:
        static: dict[str, float] = {
            t: (static_job.get(t) or {}).get("reward") for t in eval_tasks
        }
    else:
        static = {t: signals["trials"][t]["reward"] for t in eval_tasks}

    all_results: dict[str, dict[str, dict[str, Any]]] = {
        "static": {
            t: {"reward": static[t], "n_input_tokens": static_job.get(t, {}).get("n_input_tokens")}
            for t in eval_tasks
        }
    }
    for cond in CONDITIONS[1:]:
        results: dict[str, dict[str, Any]] = {}
        for root in ("jobs", "jobs_old"):
            results.update(
                load_job_results(
                    SKILLSBENCH / root / f"graphskills-codex-n1000-evo-{cond}"
                )
            )
        all_results[cond] = results

    # ---- per-condition summary ----
    summary: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS:
        r = all_results[cond]
        vals = [v["reward"] for v in r.values()]
        rewards = [v for v in vals if v is not None]
        tok_inputs: list[int] = []
        tok_outputs: list[int] = []
        for root in ("jobs", "jobs_old"):
            tok = attempt_token_stats(SKILLSBENCH / root / f"graphskills-codex-n1000-evo-{cond}")
            tok_inputs.extend(tok["inputs"])
            tok_outputs.extend(tok["outputs"])
        n = len(rewards)
        summary[cond] = {
            "n": n,
            "mean_reward": mean(rewards),
            "pass_at_1": (sum(1 for v in rewards if v >= 0.9) / n) if n else None,
            "mean_input_tokens_M": mean(tok_inputs) / 1e6 if tok_inputs else None,
            "mean_output_tokens": mean(tok_outputs),
            "n_attempts_with_tokens": len(tok_inputs),
            "delta_reward_vs_static": (mean(rewards) - static_mean(static, eval_tasks)) if n else None,
        }

    # ---- per-task paired comparison ----
    paired: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS[1:]:
        flipped: dict[str, tuple[float | None, float | None]] = {}
        for task in eval_tasks:
            base = static[task]
            evo = all_results[cond].get(task, {}).get("reward")
            if evo is not None and base is not None and evo != base:
                flipped[task] = (base, evo)
        improved = {t: (b, e) for t, (b, e) in flipped.items() if e > b}
        regressed = {t: (b, e) for t, (b, e) in flipped.items() if e < b}
        paired[cond] = {
            "improved": {t: {"static": b, "evo": e} for t, (b, e) in improved.items()},
            "regressed": {t: {"static": b, "evo": e} for t, (b, e) in regressed.items()},
            "n_improved": len(improved),
            "n_regressed": len(regressed),
        }

    # ---- lever main effects (factorial) ----
    main_effects: dict[str, dict[str, Any]] = {}
    all_cells = CONDITIONS  # includes the static baseline (empty lever set)
    for lever in LEVERS:
        with_cells = [c for c in all_cells if c != "static" and lever in c]
        without_cells = [c for c in all_cells if lever not in c]
        w = mean([summary[c]["mean_reward"] for c in with_cells if summary[c]["mean_reward"] is not None])
        wo = mean([summary[c]["mean_reward"] for c in without_cells if summary[c]["mean_reward"] is not None])
        w_tok = mean([summary[c]["mean_input_tokens_M"] for c in with_cells if summary[c]["mean_input_tokens_M"] is not None])
        wo_tok = mean([summary[c]["mean_input_tokens_M"] for c in without_cells if summary[c]["mean_input_tokens_M"] is not None])
        main_effects[lever] = {
            "with_cells": with_cells,
            "without_cells": without_cells,
            "mean_reward_with": w,
            "mean_reward_without": wo,
            "reward_main_effect": (w - wo) if w is not None and wo is not None else None,
            "tokens_with": w_tok,
            "tokens_without": wo_tok,
            "tokens_main_effect_M": (w_tok - wo_tok) if w_tok is not None and wo_tok is not None else None,
        }

    output = {
        "eval_tasks": eval_tasks,
        "summary": summary,
        "paired": paired,
        "main_effects": main_effects,
    }
    out_path = SKILLSBENCH / "evo_data" / "analysis_summary.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # ---- human-readable printout ----
    print("=" * 72)
    print("8-condition factorial on 37 eval tasks (1 attempt)")
    print("=" * 72)
    print(f"{'cond':10s} {'n':>3} {'reward':>7} {'pass@1':>7} {'in_tok(M)':>9} {'Δreward':>8}")
    for cond in CONDITIONS:
        s = summary[cond]
        d = s["delta_reward_vs_static"]
        print(
            f"{cond:10s} {s['n']:>3} {s['mean_reward']:>7.3f} "
            f"{s['pass_at_1']:>7.3f} {s['mean_input_tokens_M']:>9.2f} "
            f"{('' if d is None else f'{d:+.3f}'):>8}"
        )
    print()
    print("Lever main effects (mean over 4 cells):")
    for lever, me in main_effects.items():
        print(
            f"  {lever}: reward {me['mean_reward_with']:.3f} - {me['mean_reward_without']:.3f} "
            f"= {me['reward_main_effect']:+.3f} | tokens {me['tokens_with']:.2f}M - {me['tokens_without']:.2f}M "
            f"= {me['tokens_main_effect_M']:+.2f}M"
        )
    print()
    for cond in CONDITIONS[1:]:
        p = paired[cond]
        print(
            f"{cond:10s} flipped: {p['n_improved']} improved / {p['n_regressed']} regressed"
        )
    print(f"\nwrote {out_path}")


def static_mean(static: dict[str, float], eval_tasks: list[str]) -> float:
    vals = [static[t] for t in eval_tasks if static[t] is not None]
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    main()
