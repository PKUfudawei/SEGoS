# Paper materials

`tables/index.json` maps all 12 manuscript tables to JSON, with captions,
column headings, values, units and display precision.

`data/` includes graph snapshots, five per-attempt score files, evolution
signals and deltas, avoid-transition evidence, and the explicit 50/37 task split.
It also contains complete SkillsBench tasks, ALFWorld task JSON, PDDL and
TextWorld games with their logic files, and the upstream skill libraries with
200, 500, 1,000 and 2,000 skills, stored with Git LFS.

See [DATA.md](../DATA.md) for anonymous download and extraction instructions.
`manifest.json` records checksums of all paper files; `data/manifest.json` covers
the data files. Full raw agent trajectory archives are separate from this snapshot;
parsed evolution evidence is included in the two evolution archives.

Recompute the primary DeepSeek paired comparisons without a model API:

```bash
python scripts/paired_statistics.py
```

The script reads the included full87 and held-out scores and writes CSV, JSON,
Markdown and LaTeX under `build/paired_statistics/`.

Regenerate table JSON from the manuscript with
`python scripts/export_tables.py --paper-root /path/to/manuscript --output paper/tables`.
Refresh evidence and skill archives with
`python scripts/export_paper_data.py --data-root /path/to/data --output paper`.
The exporter preserves the reviewed task archive, whose packaging is described in
`data/benchmarks/skillsbench/archive_manifest.json`.

The reported ALFWorld experiments use 5 seeds, 8 candidates, 2,400 characters
per skill and 12,000 per bundle for both evaluation and offline evolution.
