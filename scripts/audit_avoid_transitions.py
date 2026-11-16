"""Verify released multi-round avoid relations against prior raw attempts.

Run with --data-root pointing to a data release. This reads archived trajectories,
does not run agents, and does not regenerate graph topology, weights or rewards.
"""
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "benchmarks" / "skillsbench"
sys.path.insert(0, str(CODE))
from segos.evolution import updates as levers
from benchmarks.skillsbench.evo import signals


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pair(edge):
    return frozenset((edge["source"], edge["target"]))


def main():
    global ROOT
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("runs/avoid_transition_evidence.json"))
    args = ap.parse_args()
    ROOT = args.data_root.resolve()
    paths = {0: ROOT / "experiments/skillsbench/full87/coldstart/segos_coldstart_skills1000.json", **{
        r: ROOT / f"experiments/skillsbench/full87/graphs/segos_evolved_skills1000_round{r}.json"
        for r in (1, 2, 3)}}
    graphs = {r: json.loads(p.read_text()) for r, p in paths.items()}
    history = []
    report = {
        "policy_version": levers.AVOID_POLICY_VERSION,
        "scope": "Prior-attempt evidence for released avoid relations; not a replay of all graph updates or a new performance evaluation.",
        "rules": {"success_reward": levers.SUCCESS_REWARD,
                  "minimum_failed_tasks": levers.AVOID_FAIL_MIN,
                  "maximum_successful_tasks": levers.AVOID_SUCC_MAX,
                  "retract_after_successful_tasks": levers.AVOID_RETRACT_SUCC_MIN,
                  "history": "All actual attempts from rounds 0 through r-1, restricted to training tasks."},
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        "implementation": {p.name: sha(p) for p in
                           (Path(levers.__file__), Path(signals.__file__))},
        "source_rounds": {}, "transitions": {},
    }
    for r in (0, 1, 2):
        archive = ROOT / f"experiments/skillsbench/full87/traces/segos_traces_round{r}_{'train' if r == 0 else 'eval'}.tar.gz"
        report["inputs"][str(archive.relative_to(ROOT))] = sha(archive)
        records = []
        with tempfile.TemporaryDirectory(prefix="segos-avoid-evidence-") as tmp:
            root = Path(tmp)
            with tarfile.open(archive) as tar:
                for member in tar:
                    p = Path(member.name)
                    if not member.isfile() or p.is_absolute() or ".." in p.parts or p.name.startswith("._"):
                        continue
                    if ((p.name == "result.json" and len(p.parts) == 3)
                            or (p.name == "trajectory.json" and len(p.parts) == 4)):
                        dest = root / p
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(tar.extractfile(member).read())
            mapping = signals.dir_to_name_map(graphs[r])
            for result in sorted(root.glob("*/*/result.json")):
                trial = signals.extract_trial(result.parent, mapping)
                if trial:
                    records.append(dict(trial, round=r,
                                        source_job=str(archive.relative_to(ROOT))))
        tasks = {t["task_name"] for t in records}
        assert len(records) == 174 and len(tasks) == 87
        history.extend(records)
        report["source_rounds"][str(r)] = {"attempts": len(records), "tasks": len(tasks)}
        if r == 0:
            continue
        target = r + 1
        sig = {"trials": {}, "avoid_trials": history}
        evidence = levers.avoid_co_use_evidence(sig, tasks)
        previous = {pair(e) for e in graphs[r]["edges"] if e["type"] == "avoid"}
        current = {pair(e) for e in graphs[target]["edges"] if e["type"] == "avoid"}

        def describe(p):
            ev = evidence.get(p, {"failed_tasks": set(), "successful_tasks": set(),
                                  "failed_attempts": [], "successful_attempts": []})
            return {"pair": sorted(p),
                    "failed_task_count": len(ev["failed_tasks"]),
                    "successful_task_count": len(ev["successful_tasks"]),
                    "failed_attempts": ev["failed_attempts"],
                    "successful_attempts": ev["successful_attempts"]}

        groups = {"removed": previous-current, "retained": previous & current,
                  "added": current-previous}
        for p in groups["removed"]:
            assert evidence[p]["successful_tasks"], sorted(p)
        for p in current:
            assert evidence[p]["failed_tasks"] and not evidence[p]["successful_tasks"], sorted(p)
        eligible = {pair(e) for e in levers.compute_avoid_edges(sig, tasks, graphs[target])}
        assert current <= eligible
        report["transitions"][str(target)] = {
            "prior_trace_rounds": list(range(target)),
            "counts": {"previous": len(previous), **{k: len(v) for k,v in groups.items()},
                       "current": len(current)},
            **{name: [describe(p) for p in sorted(pairs, key=lambda p: sorted(p))]
               for name, pairs in groups.items()},
            "eligible_on_released_navigation_graph": len(eligible),
            "eligible_but_not_released": [sorted(p) for p in sorted(eligible-current, key=lambda p: sorted(p))],
        }
        print(f"Round {target}: {report['transitions'][str(target)]['counts']}", flush=True)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
