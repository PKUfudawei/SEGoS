# Paper tables

The 12 JSON files record the tables in the manuscript, in publication order.
`index.json` maps table numbers and LaTeX labels to files. Each table contains its
caption, panels, column headings, numeric values, display text and cell spans.
Units and interpretation follow the headings and captions; R is reward percent,
T is input tokens in millions and S is runtime in seconds.

Regenerate from the manuscript with `python3 scripts/export_tables.py`.
These table records are separate from per-attempt experiment records.
