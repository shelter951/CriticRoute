#!/usr/bin/env python3
"""R2R edge-cloud evaluation with continuation-verified critical labels.

This script is isolated from `hetero_router/eval_hetero_edgecloud_r2r.py`.
It keeps the same edge/cloud execution stack but changes the label oracle:

old: one-step edge-vs-cloud utility on the current rollout state.
new: action difference -> cloud continuation verification.

The new label asks whether accepting the edge action still preserves
cloud-continuation navigation quality. If yes, the difference is neutral. If
not, the state is divergent/critical and the oracle-corrected rollout executes
the cloud action.
"""

import argparse
import json
import math
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


def graph_candidates(graph, current_vp):
    out = []
    for nb in sorted(graph.neighbors(current_vp)):
        data = graph.get_edge_data(current_vp, nb, default={})
        out.append(
            {
                "viewpointId": nb,
                "distance": float(data.get("weight", 1.0) or 1.0),
                "relative_angle": 0.0,
                "abs_heading": 0.0,
            }
        )
    return out


def choose_cloud_action(scan, current_vp, candidates, teacher_path, graph):
    """Trajectory-replay cloud advisor with graph re-anchoring."""
    if not teacher_path:
        return None
    cand_vps = [c["viewpointId"] for c in candidates]
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
    lengths = dict(nx.single_source_dijkstra_path_length(graph, target, weight="weight"))
    best_idx, best_dist = None, float("inf")
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


def action_distance(dists, scan, current_vp, action, candidates):
    nxt = action_next_vp(action, current_vp, candidates)
    if nxt == current_vp:
        return 0.0
    return float(dists[scan][current_vp].get(nxt, 0.0))


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


def compute_one_step_label(scan, goal, current_vp, path, candidates, qwen_action, cloud_action, dists, min_gain):
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


def rollout_cloud_continuation(scan, goal, start_vp, start_path, prefix_len, teacher_path, graph, dists, gt_len, max_steps):
    """Continue from `start_vp` with the same cloud advisor for both branches."""
    vp = start_vp
    path = list(start_path)
    total_len = float(prefix_len)
    stopped = False
    for _ in range(max_steps):
        candidates = graph_candidates(graph, vp)
        if not candidates:
            break
        action = choose_cloud_action(scan, vp, candidates, teacher_path, graph)
        if action == "stop" or action is None:
            stopped = True
            break
        next_vp = action_next_vp(action, vp, candidates)
        if next_vp == vp:
            break
        total_len += float(dists[scan][vp].get(next_vp, 0.0))
        vp = next_vp
        path.append(vp)
    nav_error = float(dists[scan][vp][goal])
    success = 1.0 if nav_error < 3.0 else 0.0
    spl = success * gt_len / max(total_len, gt_len, 1e-6)
    return {
        "final_vp": vp,
        "path": path,
        "path_len": total_len,
        "nav_error": nav_error,
        "success": success,
        "spl": float(spl),
        "stopped": stopped,
        "loop_count": len(path) - len(set(path)),
    }


def branch_then_continue(scan, goal, current_vp, path, cumul, candidates, action, teacher_path, graph, dists, gt_len, max_steps):
    if action == "stop" or action is None:
        nav_error = float(dists[scan][current_vp][goal])
        success = 1.0 if nav_error < 3.0 else 0.0
        spl = success * gt_len / max(cumul, gt_len, 1e-6)
        return {
            "final_vp": current_vp,
            "path": list(path),
            "path_len": float(cumul),
            "nav_error": nav_error,
            "success": success,
            "spl": float(spl),
            "stopped": True,
            "loop_count": len(path) - len(set(path)),
        }
    next_vp = action_next_vp(action, current_vp, candidates)
    step_dist = action_distance(dists, scan, current_vp, action, candidates)
    new_path = list(path)
    if next_vp != current_vp:
        new_path.append(next_vp)
    return rollout_cloud_continuation(
        scan,
        goal,
        next_vp,
        new_path,
        float(cumul) + float(step_dist),
        teacher_path,
        graph,
        dists,
        gt_len,
        max_steps=max(0, int(max_steps) - 1),
    )


def compute_cv_label(
    scan,
    goal,
    current_vp,
    path,
    cumul,
    candidates,
    qwen_action,
    cloud_action,
    teacher_path,
    graph,
    dists,
    gt_len,
    max_steps,
    min_gain=1.0,
    spl_gap=0.12,
    extra_len=2.5,
):
    one = compute_one_step_label(scan, goal, current_vp, path, candidates, qwen_action, cloud_action, dists, min_gain)
    if cloud_action is None:
        return {
            **one,
            "label": 0.0,
            "reason": "no_cloud_action",
            "cv_utility": 0.0,
            "cv_edge": {},
            "cv_cloud": {},
            "cv_mode": "continuation_verified",
        }
    q_next = action_next_vp(qwen_action, current_vp, candidates)
    c_next = action_next_vp(cloud_action, current_vp, candidates)
    if qwen_action == cloud_action or q_next == c_next:
        edge = branch_then_continue(scan, goal, current_vp, path, cumul, candidates, qwen_action, teacher_path, graph, dists, gt_len, max_steps)
        return {
            **one,
            "label": 0.0,
            "reason": "identical",
            "cv_utility": 0.0,
            "cv_edge": edge,
            "cv_cloud": edge,
            "cv_mode": "continuation_verified",
        }
    edge = branch_then_continue(scan, goal, current_vp, path, cumul, candidates, qwen_action, teacher_path, graph, dists, gt_len, max_steps)
    cloud = branch_then_continue(scan, goal, current_vp, path, cumul, candidates, cloud_action, teacher_path, graph, dists, gt_len, max_steps)
    reasons = []
    if qwen_action in ("stop", None) and edge["success"] <= 0 and cloud["success"] > 0:
        reasons.append("bad_stop_cont")
    if cloud["success"] > 0 and edge["success"] <= 0:
        reasons.append("success_divergent")
    if edge["nav_error"] - cloud["nav_error"] >= min_gain:
        reasons.append("naverr_divergent")
    if cloud["spl"] - edge["spl"] >= spl_gap:
        reasons.append("spl_divergent")
    if edge["path_len"] - cloud["path_len"] >= extra_len and cloud["success"] >= edge["success"]:
        reasons.append("pathlen_divergent")
    if edge["loop_count"] > cloud["loop_count"] and cloud["nav_error"] <= edge["nav_error"] + 0.5:
        reasons.append("loop_divergent")

    # Utility is long-horizon and bounded. It rewards terminal success/SPL gains
    # more than small local distance changes.
    utility = 0.0
    utility += 0.50 * max(0.0, cloud["success"] - edge["success"])
    utility += 0.25 * max(0.0, cloud["spl"] - edge["spl"])
    utility += 0.20 * min(max(0.0, edge["nav_error"] - cloud["nav_error"]) / 6.0, 1.0)
    utility += 0.05 * min(max(0.0, edge["path_len"] - cloud["path_len"]) / 6.0, 1.0)
    if not reasons:
        utility = 0.0
    return {
        **one,
        "label": 1.0 if reasons else 0.0,
        "reason": "|".join(reasons) if reasons else "neutral",
        "cv_utility": float(min(1.0, utility)),
        "cv_edge": edge,
        "cv_cloud": cloud,
        "cv_mode": "continuation_verified",
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
            hidden = int(ckpt.get("hidden", 128))
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
            if any(k.startswith("net.") for k in state):
                state = {k.replace("net.", "", 1): v for k, v in state.items()}
            self.model.load_state_dict(state)
            self.model.eval()
            thresholds = ckpt.get("budget_thresholds", {})
            if args.budget_key and args.budget_key in thresholds:
                self.threshold = float(thresholds[args.budget_key])

    def score(self, features):
        if self.args.router_mode == "trained":
            import torch

            x = np.array([features.get(k, 0.0) for k in self.feature_names], dtype=np.float32)
            x = (x - self.mean) / np.maximum(self.std, 1e-6)
            with torch.no_grad():
                logit = self.model(torch.from_numpy(x).float().unsqueeze(0)).squeeze().item()
            if logit >= 0:
                z = math.exp(-logit)
                return float(1.0 / (1.0 + z))
            z = math.exp(logit)
            return float(z / (1.0 + z))
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


def build_qwen_prompt(ep, step, cumul, candidates, kept_panos, system_prompt):
    parts = ["<|im_start|>system\n" + system_prompt + "<|im_end|>\n<|im_start|>user\n"]
    parts.append(
        f"Route instruction: {ep['instruction']}\n"
        f"Current step: {step}\n"
        f"Cumulative Distance Traveled: {cumul:.2f} meters\n\n"
        "Panorama Images from Previous Steps:"
    )
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
        parts.append(
            f"\n\tCandidate: {ci}:\n\t\tRelative angle: {abs(angle):.0f} degrees to the {direction}"
            f"\n\t\tDistance: {float(cand['distance']):.2f} meters"
            "\n\t\tview: <|vision_start|><|image_pad|><|vision_end|>"
        )
    parts.append(
        "\n\tCandidate: Stop\n\n"
        "Now, analyze the route instruction, your current position, and the available candidate directions. "
        "Select the candidate that best matches the instruction and helps you continue along the correct path. "
        "Answer on the format: Candidate: (and then the number)<|im_end|>\n"
    )
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_unseen", choices=["train", "val_seen", "val_unseen"])
    ap.add_argument("--max_episodes", type=int, default=20, help="0 means all")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--continuation_horizon", type=int, default=15)
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
    ap.add_argument("--spl_gap", type=float, default=0.12)
    ap.add_argument("--extra_len", type=float, default=2.5)
    ap.add_argument("--out_dir", default="build/continuation_router/eval")
    ap.add_argument("--samples_out", default="")
    ap.add_argument("--keep_images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = out_dir / f"images_{args.split}_{args.router_mode}_{int(time.time())}"
    img_root.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"cv_{args.router_mode}_{args.split}_max{args.max_episodes or 'all'}_{int(time.time())}.jsonl"
    samples_path = Path(args.samples_out) if args.samples_out else None
    if samples_path:
        samples_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = load_r2r_episodes(args.data_dir, args.split, args.sample_seed, args.max_episodes, args.start_index, args.end_index)
    print(f"LOADED_EPISODES split={args.split} count={len(episodes)} start={args.start_index} end={args.end_index}", flush=True)
    scans = sorted({e["scan"] for e in episodes})
    graphs = load_nav_graphs(args.connectivity_dir, scans)
    dists = shortest_distances(graphs)
    teacher = load_teacher_trajectories(args.teacher_json)
    sim = make_sim(args.dataset_path, args.connectivity_dir, rendering=True)
    worker = None if args.router_mode == "cloud" else QwenWorker(args.worker_py, args.qwen_python, args.model_dir, args.gpu, args.mode)
    system_prompt = "" if worker is None else Path(args.model_dir, "system_prompt.txt").read_text()
    router = RouterPolicy(args)
    rng = random.Random(args.sample_seed if args.sample_seed >= 0 else 20260512)
    metrics = []
    sample_f = open(samples_path, "w") if samples_path else None
    try:
        with open(results_path, "w") as fout:
            for ep_idx, ep in enumerate(episodes, start=1):
                scan = ep["scan"]
                goal = ep["path"][-1]
                gt_len = path_len(dists, scan, ep["path"])
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
                    candidates = gather_candidates(states, heading)
                    if not candidates:
                        break
                    cloud_action = choose_cloud_action(scan, vp, candidates, teacher_path, graphs[scan])
                    if args.router_mode == "cloud":
                        qwen_out = {"raw": "", "choice": None}
                        qwen_action = None
                        features = {}
                        label_info = {"label": 0.0, "reason": "cloud_only", "cv_utility": 0.0, "cv_edge": {}, "cv_cloud": {}}
                        use_cloud, router_score = True, 1.0
                    else:
                        pano_path = ep_dir / f"pano_{step}.jpg"
                        save_panorama(states, pano_path)
                        pano_history.append(str(pano_path))
                        kept_panos = pano_history[-(args.history + 1) :]
                        cand_paths = []
                        for ci, cand in enumerate(candidates):
                            cp = ep_dir / f"candidate_{step}_{ci}.jpg"
                            Image.fromarray(cand["rgb"]).save(cp, quality=92)
                            cand_paths.append(str(cp))
                        prompt = build_qwen_prompt(ep, step, cumul, candidates, kept_panos, system_prompt)
                        qwen_out = worker.ask(prompt, kept_panos, cand_paths)
                        qwen_action = parse_qwen_choice(qwen_out.get("choice"), candidates)
                        features = extract_features(ep, step, vp, path, cumul, candidates, qwen_out, qwen_action)
                        label_info = compute_cv_label(
                            scan,
                            goal,
                            vp,
                            path,
                            cumul,
                            candidates,
                            qwen_action,
                            cloud_action,
                            teacher_path,
                            graphs[scan],
                            dists,
                            gt_len,
                            max_steps=args.continuation_horizon,
                            min_gain=args.min_gain,
                            spl_gap=args.spl_gap,
                            extra_len=args.extra_len,
                        )
                        use_cloud, router_score = router.should_call(features, label_info, rng)
                    action = cloud_action if use_cloud and cloud_action is not None else qwen_action
                    if use_cloud and cloud_action is not None:
                        cloud_calls += 1
                    if sample_f:
                        one = compute_one_step_label(scan, goal, vp, path, candidates, qwen_action, cloud_action, dists, args.min_gain)
                        rec = {
                            "instr_id": ep["instr_id"],
                            "scan": scan,
                            "step": step,
                            "viewpoint": vp,
                            "features": features,
                            "label": label_info["label"],
                            "label_reason": label_info["reason"],
                            "cv_utility": label_info.get("cv_utility", 0.0),
                            "cv_edge": label_info.get("cv_edge", {}),
                            "cv_cloud": label_info.get("cv_cloud", {}),
                            "one_step_label": one["label"],
                            "one_step_reason": one["reason"],
                            "qwen_action": qwen_action,
                            "cloud_action": cloud_action,
                            "qwen_raw": qwen_out.get("raw"),
                            "teacher_path_available": bool(teacher_path),
                            **{f"label_{k}": v for k, v in label_info.items() if k not in ["label", "reason", "cv_edge", "cv_cloud"]},
                            **{f"one_step_{k}": v for k, v in one.items() if k not in ["label", "reason"]},
                        }
                        sample_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        sample_f.flush()
                    raw_steps.append(
                        {
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
                            "cv_utility": label_info.get("cv_utility", 0.0),
                            "num_candidates": len(candidates),
                        }
                    )
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
                print(
                    f"[{ep_idx}/{len(episodes)}] {ep['instr_id']} SR={sr:.2f} SPL={spl_mean:.2f} "
                    f"NE={ne:.2f} Cloud={ccr:.2f} last={raw_steps[-1] if raw_steps else {}}",
                    flush=True,
                )
    finally:
        if sample_f:
            sample_f.close()
        if worker is not None:
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
        "continuation_horizon": args.continuation_horizon,
        "spl_gap": args.spl_gap,
        "extra_len": args.extra_len,
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(out_dir / f"summary_{args.router_mode}_{args.split}_{int(time.time())}.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
