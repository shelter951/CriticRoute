#!/usr/bin/env python3
"""Summarize continuation-router samples or eval result JSONL files."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def iter_jsonl(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def summarize_samples(paths):
    rows = list(iter_jsonl(paths))
    if not rows:
        return {"type": "samples", "n": 0}
    labels = [float(r.get("label", 0.0)) for r in rows]
    one = [float(r.get("one_step_label", 0.0)) for r in rows]
    utility = [float(r.get("cv_utility", 0.0) or 0.0) for r in rows]
    both_pos = sum(1 for a, b in zip(labels, one) if a > 0.5 and b > 0.5)
    cv_only = sum(1 for a, b in zip(labels, one) if a > 0.5 and b <= 0.5)
    one_only = sum(1 for a, b in zip(labels, one) if a <= 0.5 and b > 0.5)
    return {
        "type": "samples",
        "n": len(rows),
        "episodes": len({r.get("instr_id") for r in rows}),
        "positive_rate": float(np.mean(labels)),
        "one_step_positive_rate": float(np.mean(one)),
        "mean_cv_utility": float(np.mean(utility)),
        "label_counts": dict(Counter(r.get("label_reason", "unknown") for r in rows).most_common()),
        "one_step_counts": dict(Counter(r.get("one_step_reason", "unknown") for r in rows).most_common()),
        "agreement": {
            "both_positive": both_pos,
            "cv_only_positive": cv_only,
            "one_step_only_positive": one_only,
            "both_negative": len(rows) - both_pos - cv_only - one_only,
        },
    }


def summarize_results(paths):
    rows = list(iter_jsonl(paths))
    if not rows:
        return {"type": "results", "n": 0}
    return {
        "type": "results",
        "n": len(rows),
        "sr": 100 * float(np.mean([float(r.get("success", 0.0)) for r in rows])),
        "oracle_sr": 100 * float(np.mean([float(r.get("oracle_success", 0.0)) for r in rows])),
        "spl": 100 * float(np.mean([float(r.get("spl", 0.0)) for r in rows])),
        "nav_error": float(np.mean([float(r.get("nav_error", 0.0)) for r in rows])),
        "cloud_call_rate": 100 * float(np.mean([float(r.get("cloud_call_rate", 0.0)) for r in rows])),
        "cloud_calls_per_ep": float(np.mean([float(r.get("cloud_calls", 0.0)) for r in rows])),
        "total_steps_per_ep": float(np.mean([float(r.get("total_steps", 0.0)) for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["samples", "results"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    paths = [str(Path(p)) for p in args.inputs]
    summary = summarize_samples(paths) if args.kind == "samples" else summarize_results(paths)
    summary["inputs"] = paths
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

