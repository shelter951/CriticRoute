#!/usr/bin/env python3
"""Analyze where a hetero-router actually calls the cloud."""
import argparse
import json
from collections import Counter, defaultdict


def iter_rows(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def bucket_step(step, total=15):
    if step < total / 3:
        return "early"
    if step < 2 * total / 3:
        return "middle"
    return "late"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    episodes = list(iter_rows(args.inputs))
    call_reasons = Counter()
    all_reasons = Counter()
    call_steps = Counter()
    all_steps = Counter()
    called = 0
    total = 0
    safe_calls = 0
    critical_calls = 0
    by_success = defaultdict(lambda: {"episodes": 0, "calls": 0, "steps": 0})

    for ep in episodes:
        success_key = "success" if float(ep.get("success", 0.0)) > 0.5 else "failure"
        by_success[success_key]["episodes"] += 1
        for st in ep.get("steps", []):
            total += 1
            reason = st.get("label_reason", "unknown")
            step_bucket = bucket_step(int(st.get("step", 0)))
            all_reasons[reason] += 1
            all_steps[step_bucket] += 1
            by_success[success_key]["steps"] += 1
            if st.get("used_cloud"):
                called += 1
                by_success[success_key]["calls"] += 1
                call_reasons[reason] += 1
                call_steps[step_bucket] += 1
                if reason == "safe":
                    safe_calls += 1
                else:
                    critical_calls += 1

    out = {
        "episodes": len(episodes),
        "steps": total,
        "cloud_calls": called,
        "cloud_call_rate": called / max(total, 1),
        "safe_call_rate_among_calls": safe_calls / max(called, 1),
        "critical_call_rate_among_calls": critical_calls / max(called, 1),
        "call_reasons": dict(call_reasons.most_common()),
        "all_reasons": dict(all_reasons.most_common()),
        "call_step_buckets": dict(call_steps),
        "all_step_buckets": dict(all_steps),
        "by_success": dict(by_success),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()

