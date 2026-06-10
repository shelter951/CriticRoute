#!/usr/bin/env python3
"""Summarize one or more hetero-router JSONL result files."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def summarize(rows):
    if not rows:
        return {
            "n": 0,
            "sr": 0.0,
            "oracle_sr": 0.0,
            "spl": 0.0,
            "nav_error": 0.0,
            "cloud_call_rate": 0.0,
            "cloud_calls_per_ep": 0.0,
        }
    return {
        "n": len(rows),
        "sr": 100 * float(np.mean([r.get("success", 0.0) for r in rows])),
        "oracle_sr": 100 * float(np.mean([r.get("oracle_success", 0.0) for r in rows])),
        "spl": 100 * float(np.mean([r.get("spl", 0.0) for r in rows])),
        "nav_error": float(np.mean([r.get("nav_error", 0.0) for r in rows])),
        "cloud_call_rate": 100 * float(np.mean([r.get("cloud_call_rate", 0.0) for r in rows])),
        "cloud_calls_per_ep": float(np.mean([r.get("cloud_calls", 0.0) for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    rows = load_rows(args.inputs)
    summary = summarize(rows)
    summary["name"] = args.name
    summary["inputs"] = args.inputs
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

