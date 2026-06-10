#!/usr/bin/env python3
"""Summarize official NaviLLM online-cloud intervention probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_json", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    preds = {str(r["instr_id"]): r for r in json.load(open(args.pred_json, encoding="utf-8"))}
    meta = json.load(open(args.meta_json, encoding="utf-8"))
    rows = []
    first_action_match = 0
    has_decisions = 0
    off_path = 0
    for m in meta:
        pred = preds.get(m["synthetic_instr_id_saved"])
        if pred is None:
            continue
        decisions = pred.get("decisions") or []
        if decisions:
            has_decisions += 1
        if m.get("off_teacher_path"):
            off_path += 1
        first = decisions[0] if decisions else {}
        online_action_vpid = first.get("action_vpid")
        advisor_action = m.get("advisor_action")
        # advisor_action is an index in the Qwen candidate list, so exact
        # viewpoint matching is unavailable here.  Keep the online action and
        # state metadata for later candidate-level comparison.
        rows.append(
            {
                "source_instr_id": m["source_instr_id"],
                "synthetic_instr_id": m["synthetic_instr_id_saved"],
                "source_step": m["step"],
                "start_viewpoint": m["viewpoint"],
                "goal": m["goal"],
                "off_teacher_path": bool(m["off_teacher_path"]),
                "router_score": m["router_score"],
                "label": m["label"],
                "label_reason": m["label_reason"],
                "online_first_action_vpid": online_action_vpid,
                "online_first_is_stop": bool(first.get("is_stop", False)) if first else None,
                "online_path_len": len(pred.get("trajectory", [])),
                "has_decisions": bool(decisions),
            }
        )

    summary = {
        "n_meta": len(meta),
        "n_preds": len(preds),
        "n_joined": len(rows),
        "has_decisions": has_decisions,
        "off_teacher_path": off_path,
        "note": "Official eval metrics are in the NaviLLM run log; this file checks decision-log availability for online cloud probes.",
        "examples": rows[:10],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
