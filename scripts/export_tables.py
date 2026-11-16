#!/usr/bin/env python3
"""Export every manuscript table to JSON, preserving display text and numbers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def group(text, offset):
    while text[offset].isspace():
        offset += 1
    if text[offset] != "{":
        raise ValueError(f"Expected group at {text[offset:offset+40]!r}")
    start = offset + 1
    depth = 1
    offset += 1
    while depth:
        if text[offset] == "\\":
            offset += 2
            continue
        depth += (text[offset] == "{") - (text[offset] == "}")
        offset += 1
    return text[start:offset-1], offset


def plain(text):
    text = re.sub(r"\\allowbreak\s*", "", text)
    text = re.sub(r"\\ablationrowmark\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:allowbreak|raggedright|arraybackslash)\b", "", text)
    text = re.sub(r"\\(?:textbf|underline|texttt|emph|text|mathrm|textnormal)\{([^{}]*)\}", r"\1", text)
    for a, b in {r"\%": "%", r"\_": "_", r"\checkmark": "✓", r"\uparrow": "↑", r"\downarrow": "↓", r"\le": "≤", r"\ge": "≥", r"\alpha": "α", r"\gamma": "γ", r"\eta": "η", r"\theta": "θ", r"\zeta": "ζ", r"\propto": "∝", r"\to": "→", r"\;": " ", r"\,": "", "~": " "}.items():
        text = text.replace(a, b)
    text = re.sub(r"\\ref\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("^{*}", "*").replace("{,}", ",")
    text = text.replace("{", "").replace("}", "")
    return " ".join(text.split())


def cell(raw):
    raw = raw.strip()
    body = raw
    result = {"latex": raw, "row_span": 1, "column_span": 1}
    for command, key in [(r"\multirow", "row_span"), (r"\multicolumn", "column_span")]:
        if body.startswith(command):
            span, pos = group(body, len(command))
            _, pos = group(body, pos)
            body, _ = group(body, pos)
            result[key] = int(span)
    result["text"] = plain(body)
    number = result["text"].replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", number):
        result["value"] = float(number) if "." in number else int(number)
    else:
        result["value"] = result["text"]
    if r"\textbf" in body:
        result["emphasis"] = "bold"
    elif r"\underline" in body:
        result["emphasis"] = "underline"
    return result


def panels(block):
    output = []
    for match in re.finditer(r"\\begin\{(tabular\*?)\}", block):
        env = match[1]
        _, pos = group(block, match.end())
        if env.endswith("*"):
            _, pos = group(block, pos)
        end = block.index(r"\end{" + env + "}", pos)
        body = block[pos:end]
        body = re.sub(r"(?m)(?<!\\)%[^\n]*", "", body)
        # The manuscript's cells use grouped commands; split only at depth zero.
        depth = 0
        start = 0
        rows = []
        row = []
        i = 0
        is_header = True
        while i < len(body):
            if depth == 0 and body.startswith(r"\midrule", i):
                is_header = False
                i += len(r"\midrule")
                start = i
                continue
            rule = re.match(r"\\(?:toprule|bottomrule|cmidrule(?:\([^)]*\))?\{[^}]*\})", body[i:]) if depth == 0 else None
            if rule:
                i += rule.end()
                start = i
                continue
            if depth == 0 and body[i] == "&":
                row.append(cell(body[start:i])); start = i + 1
            elif depth == 0 and body.startswith("\\\\", i):
                row.append(cell(body[start:i]))
                if any(c["text"] for c in row):
                    rows.append({"kind": "header" if is_header else "data", "cells": row})
                row = []
                i += 2
                start = i
                continue
            elif body[i] == "\\":
                i += 2
                continue
            elif body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            i += 1
        # Expand grouped model labels and other spans for machine-readable rows.
        active = {}
        matrix = []
        for row in rows:
            values = []
            col = 0
            for c in row["cells"]:
                value = active[col][0] if col in active and not c["text"] else c["value"]
                values.extend([value] * c["column_span"])
                if c["row_span"] > 1:
                    active[col] = (value, c["row_span"])
                col += c["column_span"]
            active = {k: (v, n-1) for k, (v, n) in active.items() if n > 1}
            if row["kind"] == "data":
                matrix.append(values)
        headers = [r for r in rows if r["kind"] == "header"]
        columns = [c["text"] for c in headers[-1]["cells"]] if headers else []
        output.append({"panel": chr(97+len(output)), "columns": columns, "rows": rows, "data": matrix})
    return output


def export(root, output):
    main = (root / "iclr2027_conference.tex").read_text()
    paths = [root / (p + ".tex") for p in re.findall(r"\\input\{(sections/[^}]+)\}", main)]
    index = []
    for path in paths:
        source = path.read_text()
        for match in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", source, re.S):
            block = match[0]
            labels = re.findall(r"\\label\{(tab:[^}]+)\}", block)
            if not labels:
                raise ValueError(f"Unlabelled table in {path}")
            caption_start = re.search(r"\\caption\s*\{", block)
            caption, _ = group(block, caption_start.end()-1)
            number = len(index) + 1
            name = f"table_{number:02d}_{labels[0].split(':')[1]}.json"
            table = {"schema_version": 1, "table_number": number, "labels": labels,
                     "caption": plain(caption), "caption_latex": caption,
                     "source": {"file": str(path.relative_to(root)), "line": source[:match.start()].count("\n")+1,
                                "table_sha256": hashlib.sha256(block.encode()).hexdigest()},
                     "panels": panels(block)}
            output.mkdir(parents=True, exist_ok=True)
            (output / name).write_text(json.dumps(table, ensure_ascii=False, indent=2)+"\n")
            index.append({"table_number": number, "labels": labels, "file": name, "caption": table["caption"]})
    (output / "index.json").write_text(json.dumps({"schema_version": 1, "tables": index}, ensure_ascii=False, indent=2)+"\n")
    print(f"Exported {len(index)} tables to {output}")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "tables")
    args = parser.parse_args()
    export(args.paper_root, args.output)
