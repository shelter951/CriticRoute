#!/usr/bin/env python3
"""Build R2R-like probe episodes from mixed edge-cloud intervention states.

The probe asks the official NaviLLM cloud model to re-plan online from states
visited by the edge-cloud system.  Each synthetic item starts from a router
cloud-call viewpoint and keeps the original R2R instruction/goal.  This is not
used for training; it is a systems sanity check for whether an online cloud
model can replace the fast trajectory-replay advisor.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_instruction_map(r2r_path: Path) -> dict[str, dict]:
    data = json.load(open(r2r_path, encoding="utf-8"))
    out = {}
    for item in data:
        for idx, instr in enumerate(item["instructions"]):
            instr_id = f"{item['path_id']}_{idx}"
            out[instr_id] = {
                "instruction": instr,
                "scan": item["scan"],
                "heading": float(item.get("heading", 0.0) or 0.0),
                "goal": item["path"][-1],
                "orig_path": item["path"],
            }
    return out


def iter_router_call_states(result_paths: list[Path]):
    for path in result_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                teacher_path = set(rec.get("teacher_path") or [])
                for step in rec.get("steps", []):
                    if not step.get("used_cloud"):
                        continue
                    yield {
                        "source_result": str(path),
                        "source_instr_id": rec["instr_id"],
                        "scan": rec["scan"],
                        "viewpoint": step["viewpoint"],
                        "goal": rec["gt_path"][-1],
                        "step": int(step["step"]),
                        "router_score": float(step.get("router_score", 0.0) or 0.0),
                        "label": float(step.get("label", 0.0) or 0.0),
                        "label_reason": step.get("label_reason", ""),
                        "qwen_action": step.get("qwen_action"),
                        "advisor_action": step.get("cloud_action"),
                        "off_teacher_path": step["viewpoint"] not in teacher_path,
                    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router_results", nargs="+", required=True)
    ap.add_argument("--r2r_file", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--max_items", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--prefer_off_path", action="store_true")
    args = ap.parse_args()

    instr_map = load_instruction_map(Path(args.r2r_file))
    states = list(iter_router_call_states([Path(p) for p in args.router_results]))
    states = [s for s in states if s["source_instr_id"] in instr_map]
    if args.prefer_off_path:
        states.sort(key=lambda x: (not x["off_teacher_path"], -x["router_score"]))
    else:
        rng = random.Random(args.seed)
        rng.shuffle(states)
        states.sort(key=lambda x: -x["router_score"])

    seen = set()
    items = []
    meta = []
    for state in states:
        key = (state["source_instr_id"], state["viewpoint"])
        if key in seen:
            continue
        seen.add(key)
        base = instr_map[state["source_instr_id"]]
        # The official R2R loader expands each item into per-instruction records.
        # Use one instruction per synthetic path to keep accounting exact.
        syn_path_id = 900000000 + len(items)
        item = {
            "distance": 0.0,
            "scan": state["scan"],
            "path_id": syn_path_id,
            "path": [state["viewpoint"], state["goal"]],
            "heading": base["heading"],
            "instructions": [base["instruction"]],
            "instr_encodings": [[]],
        }
        state.update(
            {
                "synthetic_path_id": syn_path_id,
                "synthetic_instr_id_official": f"r2r_{syn_path_id}_0",
                "synthetic_instr_id_saved": f"{syn_path_id}_0",
                "instruction": base["instruction"],
                "orig_path": base["orig_path"],
            }
        )
        items.append(item)
        meta.append(state)
        if len(items) >= args.max_items:
            break

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(meta, open(args.meta_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "items": len(items),
                "states_considered": len(states),
                "off_path_items": sum(1 for m in meta if m["off_teacher_path"]),
                "out_json": args.out_json,
                "meta_json": args.meta_json,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
