# SkillsBench adapter

Install the `skillsbench` extra, Docker and Compose. Use `segos prepare` with
explicit task, skill and graph paths, then `segos run` on the generated YAML.
The adapter uses Harbor 0.26.0's public Codex agent and your configured API.
The endpoint must support the Responses protocol; offline L3 rewriting uses
the separately configured Chat Completions endpoint.

The fixed retrieval configuration is 4 lexical seeds, 5 candidates, reverse-aware
PPR (restart 0.2), and 1,800/9,000 character budgets. The bundled standalone
query script is copied from `segos.retrieval.query` into task images.

`evo/signals.py` extracts rewards and actual skill use from structured traces.
`segos.evolution.pipeline` is the shared evolution implementation. Benchmark entry points use this shared implementation.
Use `segos evolve` for each condition, with the same base graph and training
signals; do not combine deltas computed for another condition.

The verifier wrapper computes fractional unit-test reward. Original task tests
and instructions remain the comparison reference. Task generation records input
hashes; no deployment-specific Docker overlay is required.

This adapter derives from the upstream SkillsBench harness. Its license is
retained in `LICENSE`.

For new evolution runs, `segos signals --attempt-policy all_attempts` retains
each attempt independently for all update channels. The historical runner
scripts keep selected-attempt behavior. See [attempt protocols](attempt_protocol.md)
for normalization, evidence boundaries and reproduction commands.
