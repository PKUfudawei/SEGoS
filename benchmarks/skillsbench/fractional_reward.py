#!/usr/bin/env python3
"""Install/read the test-fraction reward wrapper used by local evaluations."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any

WRAPPER = """#!/usr/bin/env bash
# SE-GOS-FRACTIONAL-REWARD: run the canonical tests, then score their CTRF report.
set +e
mkdir -p /logs/verifier
rm -f /logs/verifier/ctrf.json /logs/verifier/reward.txt
cd /tests || exit 1
bash /tests/test.original.sh "$@"
python3 /tests/_fractional_reward.py \\
  --report /logs/verifier/ctrf.json \\
  --reward /logs/verifier/reward.txt \\
  --details /logs/verifier/fractional_reward.json
score_status=$?
if [ "$score_status" -ne 0 ]; then
  exit "$score_status"
fi
# Reward, not pytest's process exit code, carries task-level success.
exit 0
"""


def install_fractional_reward(task_dir: Path) -> None:
    """Wrap a copied SkillsBench task's pytest verifier in-place.

    Requires the task's existing tests/test.sh to produce a CTRF report. The
    source test script is preserved as test.original.sh; generated tasks are
    disposable outputs and can be regenerated from their canonical sources.
    """
    tests_dir = task_dir / "tests"
    test_script = tests_dir / "test.sh"
    if not test_script.is_file():
        raise FileNotFoundError(f"SkillsBench task has no tests/test.sh: {task_dir}")

    script = test_script.read_text(encoding="utf-8")
    if "SE-GOS-FRACTIONAL-REWARD" in script:
        return

    ctrf_match = re.search(r"--ctrf(?:=|\s+)([^\s]+)", script)
    if ctrf_match is None:
        raise ValueError(
            f"Task verifier does not emit a CTRF report; cannot compute test-fraction reward: {test_script}"
        )
    # Normalize the path so every generated task writes to the same location.
    script = script[: ctrf_match.start()] + "--ctrf /logs/verifier/ctrf.json" + script[ctrf_match.end() :]
    test_script.write_text(script, encoding="utf-8")

    original_script = tests_dir / "test.original.sh"
    if original_script.exists():
        raise FileExistsError(f"Refusing to overwrite preserved verifier: {original_script}")
    test_script.rename(original_script)
    executable_mode = original_script.stat().st_mode & 0o777
    test_script.write_text(WRAPPER, encoding="utf-8")
    test_script.chmod(executable_mode | stat.S_IXUSR)
    shutil.copy2(Path(__file__), tests_dir / "_fractional_reward.py")


def fractional_reward(report_path: Path) -> tuple[float, dict[str, int]]:
    """Return passed / all collected test cases; non-passes count in denominator."""
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    tests = ((report.get("results") or {}).get("tests"))
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"CTRF report has no test cases: {report_path}")

    counts = {"passed": 0, "failed": 0, "skipped": 0, "pending": 0, "other": 0}
    for case in tests:
        if not isinstance(case, dict) or case.get("status") not in counts:
            raise ValueError(f"CTRF report has a test with an unknown status: {case!r}")
        counts[case["status"]] += 1

    # Literal test-case pass fraction: skipped, pending, failed, and other
    # outcomes remain in the denominator and do not count as passed.
    return counts["passed"] / len(tests), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    try:
        reward, counts = fractional_reward(args.report)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"cannot compute fractional SkillsBench reward: {exc}", file=sys.stderr)
        return 2

    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{reward:.12g}\n", encoding="utf-8")
    args.details.write_text(
        json.dumps(
            {
                "reward": reward,
                "passed": counts["passed"],
                "total": sum(counts.values()),
                "counts_by_status": counts,
                "metric": "passed_test_cases / all_collected_test_cases",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"fractional reward={reward:.6f} "
        f"({counts['passed']}/{sum(counts.values())} test cases passed)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
