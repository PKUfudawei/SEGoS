# ALFWorld adapter

Install `.[alfworld]` and the public Codex CLI. Unpack the included ALFWorld task
and TextWorld game archive and set `ALFWORLD_DATA` as described in
[DATA.md](../../DATA.md#alfworld-task-and-game-assets). The archive includes
task JSON, PDDL, executable text games and domain/grammar logic.
Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` for a Responses-compatible endpoint.
For L3, set `EVO_LLM_API_KEY` and `EVO_LLM_BASE` for Chat Completions serving
the same explicit model ID.

```bash
python -m benchmarks.alfworld.run_pipeline \
  --model gpt-5.2-codex --graph paper/data/experiments/alfworld/coldstart/segos_coldstart_skills200.json \
  --skills-root paper/data/skillsets/skills_200 --output runs/alfworld --workers 4
```

The pipeline collects 420 training episodes once, evolves offline, then runs
two separately identified attempts over all 140 `valid_seen` evaluation episodes.
Use `--factorial` to include all seven nonempty component combinations.
Incomplete runs fail the expected-count check. New configurations require new
output directories; existing result identities are validated before resuming.

The reported experiments use GoS v4's retrieval settings: seed-K=5, top-N=8,
2,400 characters per skill and 12,000 per bundle. Both initial and agent-requested
retrieval use this budget, as does offline evolution. PPR restart remains 0.2.
There is no hidden top-4 clamp. Original skill files are unchanged.
