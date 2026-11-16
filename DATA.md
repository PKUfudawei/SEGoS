# Data and paper artifacts

This repository includes full SkillsBench task packages, ALFWorld task and
TextWorld game assets, and all four skill archives under `paper/data/`, alongside
graph snapshots, selected per-attempt
scores, evolution evidence and all 12 manuscript tables.

## Download and verify

The anonymous repository is https://anonymous.4open.science/r/SEGoS/.
Its ZIP download and file endpoints resolve Git LFS content. If an archive is
missing or is still an LFS pointer after downloading a checkout, run:

```bash
bash scripts/download_data.sh
python scripts/verify_data.py --data-root paper
```

The downloader uses only the anonymous URL, verifies each archive's size and
SHA-256, and skips files that already match. It requires no GitHub account or
model API. In a Git checkout with the original remote configured, `git lfs pull`
is also supported. The verification command detects unhydrated pointer files.

## Included archives

| Archive | Contents |
|---|---|
| `paper/data/benchmarks/skillsbench/tasks.tar.gz` | All 87 evaluated task packages, including instructions, verifiers, data, Dockerfiles and build inputs; 93 Dockerfiles in total |
| `paper/data/benchmarks/alfworld/alfworld_textworld.tar.gz` | Official ALFWorld task JSON, initial-state PDDL, executable TextWorld games, domain logic and language grammar |
| `paper/data/skillsets/archives/skills_200.tar.gz` | Complete 200-skill library used for ALFWorld |
| `paper/data/skillsets/archives/skills_1000.tar.gz` | Complete 1,000-skill library used for SkillsBench |
| `paper/data/skillsets/archives/skills_500.tar.gz` and `skills_2000.tar.gz` | Additional released upstream library sizes |

The task archive removes local Git reflogs from a cached upstream checkout and
normalizes archive owner fields. All retained file contents are preserved,
including task instructions, verifiers, configurations and build inputs.
`paper/data/benchmarks/skillsbench/archive_manifest.json` records its checksum and
packaging transformation. Skill archives retain upstream attribution and the
checksums in `paper/data/skillsets/archive_manifest.json`.

## Unpack and run

Use Python 3.14 and install the benchmark extra as described in [README.md](README.md).
From the repository root:

```bash
python scripts/prepare_skillsets.py --sizes 200 1000
TASK_SHA256=$(python -c 'import json; print(json.load(open("paper/data/benchmarks/skillsbench/archive_manifest.json"))["sha256"])')
python -m segos unpack paper/data/benchmarks/skillsbench/tasks.tar.gz \
  --sha256 "$TASK_SHA256" --output runs/inputs
```

The task directory is `runs/inputs/tasks`. Skill libraries are unpacked to
`paper/data/skillsets/skills_200` and `paper/data/skillsets/skills_1000`.
The extraction command requires a new output directory. Follow the fixed-graph
example in [README.md](README.md) to prepare and run all 87 tasks; use `--task`
for a single-task smoke run. Held-out task names are listed in
`configs/paper/splits/train50_eval37.json`.

### ALFWorld task and game assets

Install `.[alfworld]`, then unpack the included archive:

```bash
ALFWORLD_SHA256=$(python -c 'import json; print(json.load(open("paper/data/benchmarks/alfworld/archive_manifest.json"))["sha256"])')
python -m segos unpack paper/data/benchmarks/alfworld/alfworld_textworld.tar.gz \
  --sha256 "$ALFWORLD_SHA256" --output runs/alfworld-inputs
export ALFWORLD_DATA="$PWD/runs/alfworld-inputs/alfworld"
```

The archive contains the official `json_2.1.1` task tree, executable
`game.tw-pddl` files, `initial_state.pddl`, `traj_data.json`, and
`logic/alfred.pddl` / `logic/alfred.twl2`. It covers all released `train`,
`valid_train`, `valid_seen` and `valid_unseen` assets, including 3,553 train and
140 valid-seen TextWorld games. The paper pipeline selects 420 training episodes
with shuffle seed 2026 and evaluates all 140 valid-seen episodes twice.

These assets support the paper's `AlfredTWEnv` text environment. AI2-THOR visual
simulator binaries, visual detector weights and pretrained BUTLER checkpoints are
not needed by this environment. Original task/game contents and the MIT license
are retained; `paper/data/benchmarks/alfworld/archive_manifest.json` records
upstream URLs, versions, sizes and SHA-256 values. An internal `manifest.json`
records each extracted payload file.

The paper's ALFWorld skill library is `skills_200`. The included `skills_1000`
is the upstream general-purpose library, not a separate ALFWorld-specific release.
See the [ALFWorld adapter](benchmarks/alfworld/README.md) for the execution pipeline.

## Evidence and scope

`paper/data/experiments/skillsbench/{full87,heldout}/` contains the SE-GoS and
SkillDAG score files for paired statistics, evolved graphs and evolution records.
The typed-initialization control includes its evaluation scores. Other reported
conditions are represented in the table JSON and available graph artifacts;
this snapshot does not contain every condition's raw execution trajectory.

Run `python scripts/paired_statistics.py` to regenerate the paired comparisons
from the included scores, without an API key or agent execution.
