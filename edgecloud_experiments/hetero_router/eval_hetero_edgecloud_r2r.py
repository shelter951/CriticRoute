#!/usr/bin/env python3
"""Evaluate and collect data for heterogeneous R2R edge-cloud routing.

The edge model is Qwen2.5-VL-R2R-panoramic.  The cloud side is represented by a
trusted NaviLLM teacher trajectory file.  This lets us debug the routing problem
without mixing the fragile official NaviLLM environment into the Qwen runtime.

Router modes:
- small: always use Qwen2.5-VL.
- cloud: always use the cloud trajectory advisor.
- random: call cloud with a fixed budget probability.
- heuristic: uncertainty/loop based hand-written policy.
- oracle: use the intervention label at each state, for an upper bound.
- trained: use a trained budget-aware MLP router.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edgecloud_experiments.eval_qwen25vl_r2r_panoramic import (  # noqa: E402
    QwenWorker,
    gather_candidates,
    load_nav_graphs,
    make_sim,
    save_panorama,
    scan_36,
    shortest_distances,
)


def load_r2r_episodes(data_dir, split, sample_seed=-1, max_episodes=0, start_index=0, end_index=0):
    data_file = Path(data_dir) / f"R2R_{split}_enc.json"
    if not data_file.exists() and split == "train":
        # This project snapshot does not always keep the canonical R2R_train_enc.json.
        # FGR2R_train.json preserves the R2R train scan/path/heading/instructions fields
        # and its path_id/instr_id convention matches the collected NaviLLM teacher traces.
        fallback = Path(data_dir) / "FGR2R_train.json"
        if fallback.exists():
            data_file = fallback
    if not data_file.exists():
        raise FileNotFoundError(f"R2R split file not found: {data_file}")
    data = json.load(open(data_file))
    episodes = []
    for item in data:
        for i, instr in enumerate(item["instructions"]):
            ep = dict(item)
            ep["instr_id"] = f"{item['path_id']}_{i}"
            ep["instruction"] = instr
            episodes.append(ep)
    if sample_seed is not None and sample_seed >= 0:
        rng = random.Random(sample_seed)
        rng.shuffle(episodes)
    if (start_index and start_index > 0) or (end_index and end_index > 0):
        start = max(0, int(start_index or 0))
        end = int(end_index) if end_index and end_index > 0 else None
        episodes = episodes[start:end]
    if max_episodes and max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def load_teacher_trajectories(path):
    data = json.load(open(path))
    out = {}
    for rec in data:
        traj = rec.get("trajectory", [])
        vpids = []
        for x in traj:
            if isinstance(x, (list, tuple)) and x:
                vpids.append(x[0])
            elif isinstance(x, str):
                vpids.append(x)
        out[str(rec["instr_id"])] = vpids
    return out


def path_len(dists, scan, path):
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        try:
            total += float(dists[scan][a][b])
        except Exception:
            pass
    return total


def shortest_next_hop(graph, src, dst):
    if src == dst:
        return "stop"
    try:
        sp = nx.shortest_path(graph, src, dst, weight="weight")
    except Exception:
        return None
    if len(sp) < 2:
        return "stop"
    return sp[1]


def choose_cloud_action(scan, current_vp, candidates, teacher_path, graph):
    """Return a candidate index or 'stop' for a trajectory-replay cloud advisor.

    The advisor exactly follows the teacher trajectory when the current viewpoint
    is on that trajectory.  If the edge model has drifted away, it re-anchors to
    the teacher-predicted final viewpoint through the navigation graph.  This is
    a bounded surrogate: always-cloud reproduces the teacher destination rather
    than the ground-truth shortest path.
    """
    if not teacher_path:
        return None
    cand_vps = [c["viewpointId"] for c in candidates]

    target = None
    if current_vp in teacher_path:
        pos = max(i for i, vp in enumerate(teacher_path) if vp == current_vp)
        if pos >= len(teacher_path) - 1:
            return "stop"
        target = teacher_path[pos + 1]
    else:
        target = teacher_path[-1]

    if target == current_vp:
        return "stop"
    if target in cand_vps:
        return cand_vps.index(target)

    hop = shortest_next_hop(graph, current_vp, target)
    if hop == "stop":
        return "stop"
    if hop in cand_vps:
        return cand_vps.index(hop)

    # Last-resort safe projection: choose the candidate closest to the target.
    best_idx, best_dist = None, float("inf")
    lengths = dict(nx.single_source_dijkstra_path_length(graph, target, weight="weight"))
    for i, cand in enumerate(candidates):
        d = lengths.get(cand["viewpointId"], float("inf"))
        if d < best_dist:
            best_idx, best_dist = i, d
    return best_idx


def parse_qwen_choice(choice, candidates):
    if choice == "stop":
        return "stop"
    if choice is None:
        return None
    try:
        idx = int(choice)
    except Exception:
        return None
    if 0 <= idx < len(candidates):
        return idx
    return None


def action_next_vp(action, current_vp, candidates):
    if action == "stop" or action is None:
        return current_vp
    return candidates[int(action)]["viewpointId"]


def candidate_stats(candidates):
    if not candidates:
        return {
            "cand_count": 0.0,
            "cand_dist_mean": 0.0,
            "cand_dist_min": 0.0,
            "cand_dist_max": 0.0,
            "cand_angle_abs_mean": 0.0,
            "cand_angle_abs_min": 0.0,
        }
    dists = [float(c.get("distance", 0.0) or 0.0) for c in candidates]
    angles = [abs(float(c.get("relative_angle", 0.0) or 0.0)) for c in candidates]
    return {
        "cand_count": float(len(candidates)),
        "cand_dist_mean": float(np.mean(dists)),
        "cand_dist_min": float(np.min(dists)),
        "cand_dist_max": float(np.max(dists)),
        "cand_angle_abs_mean": float(np.mean(angles)),
        "cand_angle_abs_min": float(np.min(angles)),
    }


def extract_features(ep, step, current_vp, path, cumul, candidates, qwen_out, qwen_action):
    stats = candidate_stats(candidates)
    probs = qwen_out.get("candidate_probs") or []
    entropy = float(qwen_out.get("candidate_entropy", 0.0) or 0.0)
    margin = float(qwen_out.get("candidate_margin", 0.0) or 0.0)
    max_prob = float(qwen_out.get("candidate_max_prob", 0.0) or 0.0)
    stop_prob = float(qwen_out.get("stop_prob", 0.0) or 0.0)
    selected_prob = 0.0
    if isinstance(qwen_action, int) and qwen_action < len(probs):
        selected_prob = float(probs[qwen_action])
    elif qwen_action == "stop" and probs:
        selected_prob = float(probs[-1])

    chosen_angle = 0.0
    chosen_dist = 0.0
    if isinstance(qwen_action, int) and qwen_action < len(candidates):
        chosen_angle = abs(float(candidates[qwen_action].get("relative_angle", 0.0) or 0.0))
        chosen_dist = float(candidates[qwen_action].get("distance", 0.0) or 0.0)
    qwen_next = action_next_vp(qwen_action, current_vp, candidates)
    visited = set(path[:-1])
    return {
        "step_norm": float(step) / 15.0,
        "path_len_norm": float(cumul) / 20.0,
        "instruction_len_norm": min(len(ep["instruction"].split()) / 80.0, 2.0),
        "cand_count_norm": stats["cand_count"] / 8.0,
        "cand_dist_mean_norm": stats["cand_dist_mean"] / 5.0,
        "cand_dist_min_norm": stats["cand_dist_min"] / 5.0,
        "cand_dist_max_norm": stats["cand_dist_max"] / 8.0,
        "cand_angle_abs_mean_norm": stats["cand_angle_abs_mean"] / 180.0,
        "cand_angle_abs_min_norm": stats["cand_angle_abs_min"] / 180.0,
        "qwen_entropy": entropy,
        "qwen_margin": margin,
        "qwen_max_prob": max_prob,
        "qwen_stop_prob": stop_prob,
        "qwen_selected_prob": selected_prob,
        "qwen_is_stop": 1.0 if qwen_action == "stop" else 0.0,
        "qwen_invalid": 1.0 if qwen_action is None else 0.0,
        "qwen_chosen_angle_norm": chosen_angle / 180.0,
        "qwen_chosen_dist_norm": chosen_dist / 5.0,
        "qwen_backtracks": 1.0 if qwen_next in visited else 0.0,
        "current_revisit_count_norm": float(path.count(current_vp)) / 5.0,
    }


def compute_intervention_label(scan, goal, current_vp, path, candidates, qwen_action, cloud_action, dists, min_gain):
    cur_err = float(dists[scan][current_vp][goal])
    qwen_next = action_next_vp(qwen_action, current_vp, candidates)
    cloud_next = action_next_vp(cloud_action, current_vp, candidates)
    qwen_err = float(dists[scan].get(qwen_next, {}).get(goal, cur_err))
    cloud_err = float(dists[scan].get(cloud_next, {}).get(goal, cur_err))

    qwen_success = qwen_err < 3.0
    cloud_success = cloud_err < 3.0
    qwen_loop = qwen_next in set(path[:-1])
    cloud_loop = cloud_next in set(path[:-1])

    reasons = []
    if qwen_action in ("stop", None) and cur_err >= 3.0 and cloud_action not in ("stop", None):
        reasons.append("bad_stop")
    if cloud_success and not qwen_success:
        reasons.append("success_flip")
    if qwen_err - cloud_err >= min_gain:
        reasons.append("naverr_gain")
    if qwen_loop and not cloud_loop and cloud_err <= qwen_err + 0.5:
        reasons.append("loop_break")
    if cloud_action is None:
        reasons = []
    return {
        "label": 1.0 if reasons else 0.0,
        "reason": "|".join(reasons) if reasons else "safe",
        "cur_err": cur_err,
        "qwen_next_err": qwen_err,
        "cloud_next_err": cloud_err,
        "qwen_success_next": float(qwen_success),
        "cloud_success_next": float(cloud_success),
    }


class RouterPolicy:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.feature_names = None
        self.mean = None
        self.std = None
        self.threshold = args.threshold
        if args.router_mode == "trained":
            import torch

            ckpt = torch.load(args.router_ckpt, map_location="cpu")
            self.feature_names = ckpt["feature_names"]
            self.mean = np.array(ckpt["mean"], dtype=np.float32)
            self.std = np.array(ckpt["std"], dtype=np.float32)
            hidden = int(ckpt.get("hidden", 64))
            self.model = torch.nn.Sequential(
                torch.nn.Linear(len(self.feature_names), hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden, 1),
            )
            state = ckpt["model"]
            # Training saves RouterMLP.net.* keys.  Strip that prefix so this
            # evaluator can stay self-contained and not import the train module.
            if any(k.startswith("net.") for k in state):
                state = {k.replace("net.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(state)
            self.model.eval()
            budget_thresholds = ckpt.get("budget_thresholds", {})
            if args.budget_key and args.budget_key in budget_thresholds:
                self.threshold = float(budget_thresholds[args.budget_key])

    def score(self, features):
        if self.args.router_mode == "trained":
            import torch

            x = np.array([features.get(k, 0.0) for k in self.feature_names], dtype=np.float32)
            x = (x - self.mean) / np.maximum(self.std, 1e-6)
            with torch.no_grad():
                logit = self.model(torch.from_numpy(x).float().unsqueeze(0)).squeeze().item()
            return float(1.0 / (1.0 + math.exp(-logit)))
        # Heuristic score is intentionally interpretable.
        score = 0.0
        score += 0.45 * float(features.get("qwen_is_stop", 0.0))
        score += 0.25 * float(features.get("qwen_backtracks", 0.0))
        score += 0.20 * max(0.0, 0.35 - float(features.get("qwen_margin", 0.35))) / 0.35
        score += 0.20 * max(0.0, float(features.get("qwen_entropy", 0.0)) - 1.0)
        score += 0.10 * float(features.get("step_norm", 0.0))
        return min(score, 1.0)

    def should_call(self, features, label_info=None, rng=None):
        mode = self.args.router_mode
        if mode == "small":
            return False, 0.0
        if mode == "cloud":
            return True, 1.0
        if mode == "oracle":
            label = bool(label_info and label_info.get("label", 0.0) > 0.5)
            return label, 1.0 if label else 0.0
        if mode == "random":
            rng = rng or random
            p = float(self.args.budget)
            return rng.random() < p, p
        score = self.score(features)
        return score >= float(self.threshold), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen", choices=["train", "val_seen", "val_unseen"])
    ap.add_argument("--max_episodes", type=int, default=20, help="0 means all")
    ap.add_argument("--start_index", type=int, default=0, help="Start offset after optional shuffle, before max_episodes.")
    ap.add_argument("--end_index", type=int, default=0, help="End offset after optional shuffle, before max_episodes. 0 means no end limit.")
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--sample_seed", type=int, default=-1)
    ap.add_argument("--data_dir", default="data/R2R")
    ap.add_argument("--connectivity_dir", default="data/connectivity")
    ap.add_argument("--dataset_path", default="data/v1/scans")
    ap.add_argument("--model_dir", default="models/Qwen2.5-VL-3B-R2R-panoramic")
    ap.add_argument("--qwen_python", default="python")
    ap.add_argument("--worker_py", default="edgecloud_experiments/qwen25vl_r2r_worker.py")
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--mode", default="forward", choices=["forward", "generate"])
    ap.add_argument("--history", type=int, default=2)
    ap.add_argument("--teacher_json", required=True)
    ap.add_argument("--router_mode", default="small", choices=["small", "cloud", "random", "heuristic", "oracle", "trained"])
    ap.add_argument("--router_ckpt", default="")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--budget_key", default="")
    ap.add_argument("--min_gain", type=float, default=1.0)
    ap.add_argument("--out_dir", default="build/hetero_router/eval")
    ap.add_argument("--samples_out", default="", help="Optional JSONL path for per-step router samples.")
    ap.add_argument("--keep_images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = out_dir / f"images_{args.split}_{args.router_mode}_{int(time.time())}"
    img_root.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"hetero_{args.router_mode}_{args.split}_max{args.max_episodes or 'all'}_{int(time.time())}.jsonl"
    samples_path = Path(args.samples_out) if args.samples_out else None
    if samples_path:
        samples_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = load_r2r_episodes(
        args.data_dir,
        args.split,
        args.sample_seed,
        args.max_episodes,
        args.start_index,
        args.end_index,
    )
    print(
        f"LOADED_EPISODES split={args.split} count={len(episodes)} "
        f"start_index={args.start_index} end_index={args.end_index} max_episodes={args.max_episodes}",
        flush=True,
    )
    scans = sorted({e["scan"] for e in episodes})
    graphs = load_nav_graphs(args.connectivity_dir, scans)
    dists = shortest_distances(graphs)
    teacher = load_teacher_trajectories(args.teacher_json)
    sim = make_sim(args.dataset_path, args.connectivity_dir, rendering=True)
    worker = QwenWorker(args.worker_py, args.qwen_python, args.model_dir, args.gpu, args.mode)
    system_prompt = Path(args.model_dir, "system_prompt.txt").read_text()
    router = RouterPolicy(args)
    rng = random.Random(args.sample_seed if args.sample_seed >= 0 else 20260502)

    metrics = []
    sample_f = open(samples_path, "w") if samples_path else None
    try:
        with open(results_path, "w") as fout:
            for ep_idx, ep in enumerate(episodes, start=1):
                scan = ep["scan"]
                goal = ep["path"][-1]
                vp = ep["path"][0]
                heading = float(ep.get("heading", 0.0) or 0.0)
                path = [vp]
                cumul = 0.0
                pano_history = []
                raw_steps = []
                cloud_calls = 0
                ep_dir = img_root / ep["instr_id"]
                ep_dir.mkdir(parents=True, exist_ok=True)
                teacher_path = teacher.get(ep["instr_id"]) or teacher.get(f"r2r_{ep['instr_id']}") or []

                for step in range(args.max_steps):
                    states = scan_36(sim, scan, vp, heading)
                    pano_path = ep_dir / f"pano_{step}.jpg"
                    save_panorama(states, pano_path)
                    pano_history.append(str(pano_path))
                    kept_panos = pano_history[-(args.history + 1):]
                    candidates = gather_candidates(states, heading)
                    if not candidates:
                        break

                    cand_paths = []
                    for ci, cand in enumerate(candidates):
                        cp = ep_dir / f"candidate_{step}_{ci}.jpg"
                        Image.fromarray(cand["rgb"]).save(cp, quality=92)
                        cand_paths.append(str(cp))

                    parts = ["<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n"]
                    parts.append(f"Route instruction: {ep['instruction']}\nCurrent step: {step}\nCumulative Distance Traveled: {cumul:.2f} meters\n\nPanorama Images from Previous Steps:")
                    history_only = kept_panos[:-1]
                    if not history_only:
                        parts.append("[]")
                    for hi, _p in enumerate(history_only):
                        parts.append(f"\n\tPanorama at step: {hi}: <|vision_start|><|image_pad|><|vision_end|>")
                    parts.append("\n\nCurrent Panorama Image:\n\t<|vision_start|><|image_pad|><|vision_end|>")
                    parts.append("\n\nCandidate Directions:")
                    for ci, cand in enumerate(candidates):
                        angle = round(float(cand["relative_angle"]), 0)
                        direction = "Left" if angle < 0 else "Right"
                        parts.append(f"\n\tCandidate: {ci}:\n\t\tRelative angle: {abs(angle):.0f} degrees to the {direction}\n\t\tDistance: {float(cand['distance']):.2f} meters\n\t\tview: <|vision_start|><|image_pad|><|vision_end|>")
                    parts.append("\n\tCandidate: Stop\n\nNow, analyze the route instruction, your current position, and the available candidate directions. Select the candidate that best matches the instruction and helps you continue along the correct path. Answer on the format: Candidate: (and then the number)<|im_end|>\n")
                    qwen_out = worker.ask("".join(parts), kept_panos, cand_paths)
                    qwen_action = parse_qwen_choice(qwen_out.get("choice"), candidates)
                    cloud_action = choose_cloud_action(scan, vp, candidates, teacher_path, graphs[scan])
                    features = extract_features(ep, step, vp, path, cumul, candidates, qwen_out, qwen_action)
                    label_info = compute_intervention_label(scan, goal, vp, path, candidates, qwen_action, cloud_action, dists, args.min_gain)
                    use_cloud, router_score = router.should_call(features, label_info, rng)
                    action = cloud_action if use_cloud and cloud_action is not None else qwen_action
                    if use_cloud and cloud_action is not None:
                        cloud_calls += 1

                    if sample_f:
                        rec = {
                            "instr_id": ep["instr_id"],
                            "scan": scan,
                            "step": step,
                            "viewpoint": vp,
                            "features": features,
                            "label": label_info["label"],
                            "label_reason": label_info["reason"],
                            "qwen_action": qwen_action,
                            "cloud_action": cloud_action,
                            "qwen_raw": qwen_out.get("raw"),
                            "teacher_path_available": bool(teacher_path),
                            **{f"label_{k}": v for k, v in label_info.items() if k not in ["label", "reason"]},
                        }
                        sample_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        sample_f.flush()

                    raw_steps.append({
                        "step": step,
                        "viewpoint": vp,
                        "qwen_raw": qwen_out.get("raw"),
                        "qwen_choice": qwen_out.get("choice"),
                        "qwen_action": qwen_action,
                        "cloud_action": cloud_action,
                        "used_cloud": bool(use_cloud and cloud_action is not None),
                        "router_score": router_score,
                        "label": label_info["label"],
                        "label_reason": label_info["reason"],
                        "num_candidates": len(candidates),
                    })

                    if action == "stop" or action is None:
                        break
                    next_vp = candidates[int(action)]["viewpointId"]
                    if next_vp == vp:
                        break
                    cumul += float(dists[scan][vp][next_vp])
                    vp = next_vp
                    heading = float(candidates[int(action)]["abs_heading"])
                    path.append(vp)

                nav_error = float(dists[scan][vp][goal])
                oracle_error = min(float(dists[scan][p][goal]) for p in path)
                success = 1.0 if nav_error < 3.0 else 0.0
                oracle_success = 1.0 if oracle_error < 3.0 else 0.0
                gt_len = path_len(dists, scan, ep["path"])
                pred_len = path_len(dists, scan, path)
                spl = success * gt_len / max(pred_len, gt_len, 1e-6)
                rec = {
                    "instr_id": ep["instr_id"],
                    "scan": scan,
                    "path_id": ep["path_id"],
                    "pred_path": path,
                    "gt_path": ep["path"],
                    "teacher_path": teacher_path,
                    "nav_error": nav_error,
                    "oracle_error": oracle_error,
                    "success": success,
                    "oracle_success": oracle_success,
                    "spl": float(spl),
                    "path_len": float(pred_len),
                    "gt_len": float(gt_len),
                    "steps": raw_steps,
                    "cloud_calls": cloud_calls,
                    "total_steps": max(len(raw_steps), 1),
                    "cloud_call_rate": cloud_calls / max(len(raw_steps), 1),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                metrics.append(rec)
                if not args.keep_images:
                    shutil.rmtree(ep_dir, ignore_errors=True)

                sr = 100 * np.mean([m["success"] for m in metrics])
                spl_mean = 100 * np.mean([m["spl"] for m in metrics])
                ne = np.mean([m["nav_error"] for m in metrics])
                ccr = 100 * np.mean([m["cloud_call_rate"] for m in metrics])
                print(f"[{ep_idx}/{len(episodes)}] {ep['instr_id']} SR={sr:.2f} SPL={spl_mean:.2f} NE={ne:.2f} Cloud={ccr:.2f} last={raw_steps[-1] if raw_steps else {}}", flush=True)
    finally:
        if sample_f:
            sample_f.close()
        worker.close()

    summary = {
        "n": len(metrics),
        "router_mode": args.router_mode,
        "sr": 100 * float(np.mean([m["success"] for m in metrics])) if metrics else 0.0,
        "oracle_sr": 100 * float(np.mean([m["oracle_success"] for m in metrics])) if metrics else 0.0,
        "spl": 100 * float(np.mean([m["spl"] for m in metrics])) if metrics else 0.0,
        "nav_error": float(np.mean([m["nav_error"] for m in metrics])) if metrics else 0.0,
        "cloud_call_rate": 100 * float(np.mean([m["cloud_call_rate"] for m in metrics])) if metrics else 0.0,
        "cloud_calls_per_ep": float(np.mean([m["cloud_calls"] for m in metrics])) if metrics else 0.0,
        "results_path": str(results_path),
        "samples_path": str(samples_path) if samples_path else "",
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(out_dir / f"summary_{args.router_mode}_{args.split}_{int(time.time())}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
