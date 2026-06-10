#!/usr/bin/env python3
"""Summarize sharded continuation-verified object-router JSONL outputs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows matched {args.input_glob!r}")

    summary = {
        "name": args.name,
        "n": len(rows),
        "input_files": paths,
        "task": rows[0].get("task"),
        "sr": 100 * float(np.mean([r.get("success", 0.0) for r in rows])),
        "oracle_sr": 100 * float(np.mean([r.get("oracle_success", 0.0) for r in rows])),
        "spl": 100 * float(np.mean([r.get("spl", 0.0) for r in rows])),
        "nav_error": float(np.mean([r.get("nav_error", 0.0) for r in rows])),
        "cloud_call_rate": 100 * float(np.mean([r.get("cloud_call_rate", 0.0) for r in rows])),
        "cloud_calls_per_ep": float(np.mean([r.get("cloud_calls", 0.0) for r in rows])),
        "steps_per_ep": float(np.mean([r.get("total_steps", len(r.get("steps", []))) for r in rows])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
