# SE-GoS: Self-Evolving Graph-of-Skills

SE-GoS updates a skill-retrieval graph offline from execution feedback. It learns
topology, edge weights and retrieval descriptions while retaining source skill
files and model weights.

## Install

Use Python 3.14 and install the benchmark you need:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[skillsbench]'
python -m segos --help
```

For ALFWorld, install `.[alfworld]` and the public Codex CLI, then unpack the
included task and TextWorld game assets following [DATA.md](DATA.md). SkillsBench additionally needs Docker and Compose.
Copy `.env.example` to `.env`, fill in your endpoint and key, then export it:

```bash
set -a
source .env
set +a
```

The task-solving CLI uses a Responses-compatible endpoint. Offline description
rewriting uses Chat Completions and can have its own endpoint and key. Set the
execution model explicitly; rewriting follows that same model ID. Graph
inspection, existing-table inspection and structural updates need no model API.

## Paper materials

[paper/tables/index.json](paper/tables/index.json) indexes all 12 manuscript
tables, including both panels of the held-out table. JSON preserves column
headings, units in captions, numeric values, display precision and cell spans.
[paper/README.md](paper/README.md) describes the included graph snapshots,
scores, evolution records and task split. Complete SkillsBench tasks, ALFWorld task and TextWorld game assets, and all four
skill archives are included with Git LFS. See [DATA.md](DATA.md) for anonymous
download, checksum verification and unpacking commands.

```bash
segos inspect paper/data/experiments/skillsbench/full87/graphs/segos_evolved_skills1000_round1.json
segos query paper/data/experiments/skillsbench/full87/graphs/segos_evolved_skills1000_round1.json 'edit an xlsx workbook'
```

## Run a fixed graph on SkillsBench

Follow [DATA.md](DATA.md) to unpack the included tasks and skill library, then run:

```bash
segos prepare --tasks-root runs/inputs/tasks --skills-root paper/data/skillsets/skills_1000 \
  --graph paper/data/experiments/skillsbench/full87/graphs/segos_evolved_skills1000_round1.json \
  --model openai/gpt-5.2-codex --attempts 2 --concurrency 4 --output runs/skillsbench-eval
segos run runs/skillsbench-eval/run.yaml
```

`prepare` writes the task tree, Harbor configuration and an input manifest. Use
`--task TASK_NAME` for a smoke run. It refuses an existing output directory.
Full87 evaluation has 87 tasks × 2 attempts; held-out evaluation uses the 37
tasks listed in `configs/paper/splits/train50_eval37.json`.

## Evolve a graph

```bash
segos signals --jobs /path/to/completed-job --graph /path/to/coldstart.json --output runs/signals.json
segos evolve --graph /path/to/coldstart.json --signals runs/signals.json \
  --split configs/paper/splits/train50_eval37.json --condition L1L2L3 \
  --model gpt-5.2-codex --output runs/heldout/graph.json
```

New `segos signals` commands default to `--attempt-policy all_attempts`:
workflow, dependency, weights and descriptions use all valid training attempts
independently. Rewards and skill-use orders are never merged. For historical
reproduction use `--attempt-policy selected_attempt`, optionally with
`--selection /path/to/manifest.json`. Signals without a policy and the archived
benchmark runner entry points retain selected-attempt behavior. The parser keeps
the selected task view in either mode; `evolution_attempt_policy` determines
which evidence the updates consume.

The all-attempt protocol is a new implementation option, not the protocol behind
the published scores. Its graph outputs carry a protocol version and are marked
as not performance-evaluated. Existing graphs, scores and tables are unchanged.
See [the protocol specification](benchmarks/skillsbench/attempt_protocol.md). Each factorial condition recomputes enabled updates
in L1 → L2 → L3 order on the preceding updated graph; caches include input hashes
and retrieval budgets. See [benchmarks/alfworld/README.md](benchmarks/alfworld/README.md)
for its separate train/evaluation pipeline.

## Retrieval configurations

| Benchmark | Seeds | Candidate limit | Characters per skill / bundle |
|---|---:|---:|---:|
| SkillsBench | 4 | 5 | 1,800 / 9,000 |
| ALFWorld | 5 | 8 | 2,400 / 12,000 |

SkillsBench follows GoS's released lightweight retriever defaults. The reported
ALFWorld experiments use GoS v4's retrieval settings. Evaluation and offline
evolution use the same retrieval configuration within each benchmark.

## Repository layout

`src/segos/` contains shared graph operations, retrieval, evolution and clients.
`benchmarks/` contains environment adapters. `configs/paper/` contains explicit
protocol settings. `evaluation/` contains shared skill-loading and analysis
helpers. Runtime outputs belong under `runs/` and are not release data.

Run `python -m pytest -q` for the regression suite. Original upstream licenses
are retained; see `LICENSE` and `benchmarks/skillsbench/LICENSE`.

The SkillsBench task archive includes the task Dockerfiles and build inputs.
Task preparation copies those environments and adds graph retrieval.
