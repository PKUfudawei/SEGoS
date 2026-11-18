# SkillsBench evolution attempt protocols

The new CLI default is `all_attempts` (version `all-attempts-v1`). The published
results remain associated with the historical selected-attempt protocol; no
benchmark execution or result regeneration accompanies this code change.

## Evidence and aggregation

Only valid parsed attempts from the requested training tasks are used. The two
attempts of a task remain separate records: reward, queries, used skills and
first-use order are never pooled into a synthetic trajectory. As before, when
multiple supplied jobs contain the same task, the later job replaces that task's
attempts; pass the intended training jobs, not unrelated graph versions. Avoid
history is a separate input and is not used for other updates.

For task t with n_t valid attempts:

- Workflow witnesses pair a query seed and used skill within the same successful
  attempt. Each query occurrence contributes 1/n_t support; any positive support
  creates an edge. Initial weight is min(0.9, 0.6 + 0.05 max(0, support - 1)),
  so a single witness retains the base weight even with fractional support.
- Dependency requires successful ordered use within one attempt and qualifying
  schema overlap. Support counts distinct tasks, not repeated attempts. Missing
  first-use order produces no dependency evidence in this mode.
- Reinforcement adds eta/n_t times the attempt's reward to each non-avoid edge
  targeting a skill used in that attempt. n_t includes zero-reward attempts.
- Semantic attenuation counts distinct tasks that surfaced a skill across any
  attempt, and only attenuates skills never used in any included training attempt.
- Description updates collect miss queries from all failed training attempts.
  Exact duplicate queries are removed per skill. The no-eviction check uses all
  training queries and used skills, including successful attempts. The existing
  per-skill candidate and edit budgets are unchanged; more eligible skills can
  increase total rewriting and retrieval cost.
- Avoid retains its existing distinct-task, actual within-attempt co-use rules
  and cumulative successful counterevidence. It never merges skill sets across
  attempts.

The normalization limits changes caused solely by repeated sampling of a task;
it does not establish better performance. New graph versions require evaluation
before assigning performance claims. Caches include the attempt policy, protocol
version and consumed evidence, including all description-update inputs.

## Commands and compatibility

```bash
# New protocol, also the default for the signals CLI:
segos signals --jobs /path/to/train-job --graph /path/to/base.json \
  --attempt-policy all_attempts --output runs/all-attempts-signals.json

# Historical per-task selection (lexicographically last valid attempt):
segos signals --jobs /path/to/train-job --graph /path/to/base.json \
  --attempt-policy selected_attempt --output runs/selected-signals.json

# Fixed historical selection, instead of selecting by directory name:
segos signals --jobs /path/to/train-job --graph /path/to/base.json \
  --attempt-policy selected_attempt --selection /path/to/selection.json \
  --output runs/selected-signals.json
```

`segos evolve` reads the policy from the signals. Existing signals without a
policy, `parse_jobs` library callers and the historical full87/heldout/multiround
runner scripts retain selected-attempt behavior. Programmatic callers opt in via
`parse_jobs(..., attempt_policy="all_attempts", merge_attempts=False)`.
Do not enable `SEGOS_MERGE_ATTEMPTS`: merging changes reward and order semantics.
Re-extract raw attempts instead of relabeling merged or selected-only records.
The training/evaluation split and scored-attempt accounting are unchanged.
