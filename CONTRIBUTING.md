# Development

Install the repository in editable mode with its `skillsbench` extra and pytest.
Run `python -m pytest -q`. Core graph/retrieval/evolution tests run offline.
Benchmark smoke runs need their environment and API settings; keep outputs under
`runs/`. Do not commit credentials, local caches or deployment overlays.

Preserve graph inputs and use a new output version when a protocol or budget
changes. Per-attempt scores, table records and graph provenance have separate
roles; publish them using the manifests rather than copying runtime directories.
