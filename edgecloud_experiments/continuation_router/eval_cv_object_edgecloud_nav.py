#!/usr/bin/env python3
"""Continuation-verified edge-cloud routing for REVERIE/SOON object navigation.

This evaluator is isolated from the earlier one-step object router.  It keeps
the same edge/cloud candidate-action interface but mines labels by asking
whether an edge-first branch remains equivalent to a cloud-first branch after a
shared continuation.  It intentionally reports navigation SR/SPL/NavErr and
cloud-call cost; object grounding metrics are outside this edge-cloud routing
study.
"""

from __future__ import annotations

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
from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import (  # noqa: E402
    build_qwen_prompt,
    load_episodes,
    min_dist_to_goals,
    path_length,
)


def load_teacher_trajectories(path: str | Path) -> dict[str, list[str]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for rec in data:
        instr_id = str(rec.get("instr_id"))
        traj = rec.get("trajectory", [])
        if traj and isinstance(traj[0], dict) and "path" in traj[0]:
            traj = traj[0]["path"]
        vpids = []
        for x in traj:
            if isinstance(x, (list, tuple)) and x:
                vpids.append(str(x[0]))
            elif isinstance(x, str):
                vpids.append(x)
        if not vpids:
            continue
        out[instr_id] = vpids
        out[f"reverie_{instr_id}"] = vpids
        out[f"soon_{instr_id}"] = vpids
    return out


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


def choose_cloud_action(current_vp, candidates, teacher_path, graph):
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


def extract_features(ep, step, current_vp, path, cumul, candidates, qwen_out, qwen_action, max_steps):
    probs = qwen_out.get("candidate_probs") or []
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
    return {
        "step_norm": float(step) / max(float(max_steps), 1.0),
        "path_len_norm": float(cumul) / 20.0,
        "instruction_len_norm": min(len(ep["instruction"].split()) / 100.0, 2.0),
        "cand_count_norm": float(len(candidates)) / 10.0,
        "qwen_entropy": float(qwen_out.get("candidate_entropy", 0.0) or 0.0),
        "qwen_margin": float(qwen_out.get("candidate_margin", 0.0) or 0.0),
        "qwen_max_prob": float(qwen_out.get("candidate_max_prob", 0.0) or 0.0),
        "qwen_selected_prob": selected_prob,
        "qwen_is_stop": 1.0 if qwen_action == "stop" else 0.0,
        "qwen_invalid": 1.0 if qwen_action is None else 0.0,
        "qwen_chosen_angle_norm": chosen_angle / 180.0,
        "qwen_chosen_dist_norm": chosen_dist / 5.0,
        "qwen_backtracks": 1.0 if qwen_next in set(path[:-1]) else 0.0,
        "current_revisit_count_norm": float(path.count(current_vp)) / 5.0,
    }


def branch_then_continue(
    scan,
    goals,
    current_vp,
    path,
    cumul,
    candidates,
    action,
    teacher_path,
    graph,
    dists,
    gt_len,
    max_steps,
):
    if action == "stop" or action is None:
        nav_error = min_dist_to_goals(dists, scan, current_vp, goals)
        success = 1.0 if nav_error < 3.0 else 0.0
        spl = success * gt_len / max(float(cumul), gt_len, 1e-6)
        return {
            "final_vp": current_vp,
            "path": list(path),
            "path_len": float(cumul),
            "nav_error": float(nav_error),
            "success": success,
            "spl": float(spl),
            "loop_count": len(path) - len(set(path)),
            "stopped": True,
        }
    next_vp = action_next_vp(action, current_vp, candidates)
    new_path = list(path)
    step_dist = action_distance(dists, scan, current_vp, action, candidates)
    if next_vp != current_vp:
        new_path.append(next_vp)
    vp = next_vp
    total_len = float(cumul) + float(step_dist)
    stopped = False
    for _ in range(max(0, int(max_steps) - 1)):
        cands = graph_candidates(graph, vp)
        if not cands:
            break
        cont_action = choose_cloud_action(vp, cands, teacher_path, graph)
        if cont_action == "stop" or cont_action is None:
            stopped = True
            break
        nxt = action_next_vp(cont_action, vp, cands)
        if nxt == vp:
            break
        total_len += float(dists[scan][vp].get(nxt, 0.0))
        vp = nxt
        new_path.append(vp)
    nav_error = min_dist_to_goals(dists, scan, vp, goals)
    success = 1.0 if nav_error < 3.0 else 0.0
    spl = success * gt_len / max(total_len, gt_len, 1e-6)
    return {
        "final_vp": vp,
        "path": new_path,
        "path_len": float(total_len),
        "nav_error": float(nav_error),
        "success": success,
        "spl": float(spl),
        "loop_count": len(new_path) - len(set(new_path)),
        "stopped": stopped,
    }


def compute_cv_label(
    scan,
    goals,
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
    spl_gap=0.10,
    extra_len=2.5,
):
    cur_err = min_dist_to_goals(dists, scan, current_vp, goals)
    qwen_next = action_next_vp(qwen_action, current_vp, candidates)
    cloud_next = action_next_vp(cloud_action, current_vp, candidates)
    qwen_err = min_dist_to_goals(dists, scan, qwen_next, goals)
    cloud_err = min_dist_to_goals(dists, scan, cloud_next, goals)
    one_step_reasons = []
    if qwen_action in ("stop", None) and cur_err >= 3.0 and cloud_action not in ("stop", None):
        one_step_reasons.append("bad_stop")
    if cloud_err < 3.0 <= qwen_err:
        one_step_reasons.append("success_flip")
    if qwen_err - cloud_err >= min_gain:
        one_step_reasons.append("naverr_gain")
    if qwen_next in set(path[:-1]) and cloud_next not in set(path[:-1]) and cloud_err <= qwen_err + 0.5:
        one_step_reasons.append("loop_break")

    if cloud_action is None:
        return {
            "label": 0.0,
            "reason": "no_cloud_action",
            "cv_utility": 0.0,
            "cur_err": float(cur_err),
            "qwen_next_err": float(qwen_err),
            "cloud_next_err": float(cloud_err),
            "one_step_reason": "|".join(one_step_reasons) if one_step_reasons else "safe",
        }

    if qwen_action == cloud_action or qwen_next == cloud_next:
        edge = branch_then_continue(scan, goals, current_vp, path, cumul, candidates, qwen_action, teacher_path, graph, dists, gt_len, max_steps)
        return {
            "label": 0.0,
            "reason": "identical",
            "cv_utility": 0.0,
            "cur_err": float(cur_err),
            "qwen_next_err": float(qwen_err),
            "cloud_next_err": float(cloud_err),
            "one_step_reason": "|".join(one_step_reasons) if one_step_reasons else "safe",
            "cv_edge": edge,
            "cv_cloud": edge,
            "cv_mode": "continuation_verified",
        }

    edge = branch_then_continue(scan, goals, current_vp, path, cumul, candidates, qwen_action, teacher_path, graph, dists, gt_len, max_steps)
    cloud = branch_then_continue(scan, goals, current_vp, path, cumul, candidates, cloud_action, teacher_path, graph, dists, gt_len, max_steps)
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

    utility = 0.0
    utility += 0.50 * max(0.0, cloud["success"] - edge["success"])
    utility += 0.25 * max(0.0, cloud["spl"] - edge["spl"])
    utility += 0.20 * min(max(0.0, edge["nav_error"] - cloud["nav_error"]) / 6.0, 1.0)
    utility += 0.05 * min(max(0.0, edge["path_len"] - cloud["path_len"]) / 6.0, 1.0)
    if not reasons:
        utility = 0.0
    return {
        "label": 1.0 if reasons else 0.0,
        "reason": "|".join(reasons) if reasons else "neutral",
        "cv_utility": float(min(1.0, utility)),
        "cur_err": float(cur_err),
        "qwen_next_err": float(qwen_err),
        "cloud_next_err": float(cloud_err),
        "one_step_reason": "|".join(one_step_reasons) if one_step_reasons else "safe",
        "cv_edge": edge,
        "cv_cloud": cloud,
        "cv_mode": "continuation_verified",
    }


class RouterPolicy:
    def __init__(self, args):
        self.args = args
        self.trained = None
        if args.router_mode == "trained":
            self.trained = self._load_trained_router(args)

    def _load_trained_router(self, args):
        import torch
        import torch.nn as nn

        if not args.router_checkpoint:
            raise ValueError("--router_checkpoint is required when --router_mode trained")
        ckpt = torch.load(args.router_checkpoint, map_location="cpu")
        feature_names = ckpt["feature_names"]
        hidden = int(ckpt.get("hidden", 128))

        class RouterMLP(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(-1)

        model = RouterMLP(len(feature_names), hidden)
        model.load_state_dict(ckpt["model"])
        model.eval()
        threshold = args.trained_threshold
        if threshold is None:
            thresholds = ckpt.get("budget_thresholds") or {}
            if args.budget_key not in thresholds:
                raise KeyError(f"budget_key={args.budget_key!r} not found; available={sorted(thresholds)}")
            threshold = thresholds[args.budget_key]
        return {
            "torch": torch,
            "model": model,
            "feature_names": feature_names,
            "mean": np.array(ckpt["mean"], dtype=np.float32),
            "std": np.maximum(np.array(ckpt["std"], dtype=np.float32), 1e-6),
            "threshold": float(threshold),
        }

    def score(self, features):
        if self.trained is not None:
            values = np.array([[float(features.get(k, 0.0)) for k in self.trained["feature_names"]]], dtype=np.float32)
            values = (values - self.trained["mean"]) / self.trained["std"]
            with self.trained["torch"].no_grad():
                x = self.trained["torch"].from_numpy(values)
                return float(self.trained["torch"].sigmoid(self.trained["model"](x))[0].item())
        score = 0.0
        score += 0.45 * float(features.get("qwen_is_stop", 0.0))
        score += 0.25 * float(features.get("qwen_backtracks", 0.0))
        score += 0.25 * max(0.0, 0.35 - float(features.get("qwen_margin", 0.35))) / 0.35
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
            label = bool(label_info and float(label_info.get("label", 0.0)) > 0.5)
            return label, 1.0 if label else 0.0
        if mode == "random":
            rng = rng or random
            p = float(self.args.budget)
            return rng.random() < p, p
        score = self.score(features)
        threshold = self.trained["threshold"] if self.trained is not None else float(self.args.threshold)
        return score >= threshold, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REVERIE", choices=["REVERIE", "SOON"])
    ap.add_argument("--split", default="val_unseen", choices=["train", "val_seen", "val_unseen", "test"])
    ap.add_argument("--max_episodes", type=int, default=300)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--end_index", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=15)
    ap.add_argument("--continuation_horizon", type=int, default=15)
    ap.add_argument("--sample_seed", type=int, default=20260507)
    ap.add_argument("--data_root", default="data")
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
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--router_checkpoint", default="")
    ap.add_argument("--budget_key", default="b40")
    ap.add_argument("--trained_threshold", type=float, default=None)
    ap.add_argument("--min_gain", type=float, default=1.0)
    ap.add_argument("--spl_gap", type=float, default=0.10)
    ap.add_argument("--extra_len", type=float, default=2.5)
    ap.add_argument("--out_dir", default="build/continuation_router/object_eval")
    ap.add_argument("--samples_out", default="")
    ap.add_argument("--strict_teacher_paths", action="store_true")
    ap.add_argument("--keep_images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.task.lower() / args.router_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = out_dir / f"images_{args.split}_{int(time.time())}"
    img_root.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"cv_object_{args.task.lower()}_{args.router_mode}_{args.split}_max{args.max_episodes}_{int(time.time())}.jsonl"

    episodes = load_episodes(args.task, args.data_root, args.split, args.sample_seed, 0)
    if args.start_index or args.end_index:
        end = args.end_index if args.end_index and args.end_index > 0 else None
        episodes = episodes[max(args.start_index, 0):end]
    if args.max_episodes and args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
    scans = sorted({e["scan"] for e in episodes})
    graphs = load_nav_graphs(args.connectivity_dir, scans)
    dists = shortest_distances(graphs)
    teacher = load_teacher_trajectories(args.teacher_json)
    sim = make_sim(args.dataset_path, args.connectivity_dir, rendering=True)
    worker = None if args.router_mode == "cloud" else QwenWorker(args.worker_py, args.qwen_python, args.model_dir, args.gpu, args.mode)
    system_prompt = "" if worker is None else Path(args.model_dir, "system_prompt.txt").read_text()
    router = RouterPolicy(args)
    rng = random.Random(args.sample_seed)

    metrics = []
    teacher_path_missing = 0
    samples_fout = open(args.samples_out, "w", encoding="utf-8") if args.samples_out else None
    try:
        with open(results_path, "w", encoding="utf-8") as fout:
            for ep_idx, ep in enumerate(episodes, start=1):
                scan = ep["scan"]
                goals = ep.get("goal_set") or [ep["path"][-1]]
                vp = ep["path"][0]
                heading = float(ep.get("heading", 0.0) or 0.0)
                path = [vp]
                cumul = 0.0
                pano_history = []
                steps = []
                cloud_calls = 0
                ep_dir = img_root / str(ep["instr_id"])
                ep_dir.mkdir(parents=True, exist_ok=True)
                teacher_path = teacher.get(str(ep["instr_id"]))
                if teacher_path is None and str(ep["instr_id"]).startswith("reverie_"):
                    teacher_path = teacher.get(str(ep["instr_id"])[8:])
                if teacher_path is None and str(ep["instr_id"]).startswith("soon_"):
                    teacher_path = teacher.get(str(ep["instr_id"])[5:])
                teacher_path = teacher_path or []
                if not teacher_path:
                    teacher_path_missing += 1
                    if args.strict_teacher_paths:
                        raise RuntimeError(f"missing teacher trajectory for instr_id={ep['instr_id']}")
                gt_len = path_length(dists, scan, ep["path"])

                for step in range(args.max_steps):
                    states = scan_36(sim, scan, vp, heading)
                    candidates = gather_candidates(states, heading)
                    if not candidates:
                        break
                    cloud_action = choose_cloud_action(vp, candidates, teacher_path, graphs[scan])
                    if args.router_mode == "cloud":
                        qwen_out = {"raw": "", "choice": None}
                        qwen_action = None
                        features = {}
                        label_info = {"label": 0.0, "reason": "cloud_only", "cv_utility": 0.0}
                        use_cloud, router_score = True, 1.0
                    else:
                        pano_path = ep_dir / f"pano_{step}.jpg"
                        save_panorama(states, pano_path)
                        pano_history.append(str(pano_path))
                        kept_panos = pano_history[-(args.history + 1):]
                        cand_paths = []
                        for ci, cand in enumerate(candidates):
                            cp = ep_dir / f"candidate_{step}_{ci}.jpg"
                            Image.fromarray(cand["rgb"]).save(cp, quality=92)
                            cand_paths.append(str(cp))
                        prompt = build_qwen_prompt(system_prompt, ep["instruction"], step, cumul, kept_panos, candidates)
                        qwen_out = worker.ask(prompt, kept_panos, cand_paths)
                        qwen_action = parse_qwen_choice(qwen_out.get("choice"), candidates)
                        features = extract_features(ep, step, vp, path, cumul, candidates, qwen_out, qwen_action, args.max_steps)
                        # Continuation verification is required for offline sample
                        # mining and oracle analysis. Deployable trained/random/
                        # heuristic routers must not need outcome labels at test
                        # time, so skip this expensive branch-rollout work there.
                        if args.router_mode == "oracle" or samples_fout is not None:
                            label_info = compute_cv_label(
                                scan,
                                goals,
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
                                args.continuation_horizon,
                                min_gain=args.min_gain,
                                spl_gap=args.spl_gap,
                                extra_len=args.extra_len,
                            )
                        else:
                            label_info = {
                                "label": 0.0,
                                "reason": "not_computed_online",
                                "one_step_reason": "",
                                "cv_utility": 0.0,
                                "cur_err": 0.0,
                                "qwen_next_err": 0.0,
                                "cloud_next_err": 0.0,
                            }
                        use_cloud, router_score = router.should_call(features, label_info, rng)
                        if samples_fout is not None:
                            samples_fout.write(
                                json.dumps(
                                    {
                                        "task": args.task,
                                        "split": args.split,
                                        "instr_id": ep["instr_id"],
                                        "scan": scan,
                                        "step": step,
                                        "viewpoint": vp,
                                        "features": features,
                                        "teacher_path_available": bool(teacher_path),
                                        "label": label_info.get("label", 0.0),
                                        "label_reason": label_info.get("reason", ""),
                                        "one_step_reason": label_info.get("one_step_reason", ""),
                                        "cv_utility": label_info.get("cv_utility", 0.0),
                                        "label_cur_err": label_info.get("cur_err", 0.0),
                                        "label_qwen_next_err": label_info.get("qwen_next_err", 0.0),
                                        "label_cloud_next_err": label_info.get("cloud_next_err", 0.0),
                                        "qwen_action": qwen_action,
                                        "cloud_action": cloud_action,
                                        "qwen_choice": qwen_out.get("choice"),
                                        "num_candidates": len(candidates),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            samples_fout.flush()

                    action = cloud_action if use_cloud and cloud_action is not None else qwen_action
                    if use_cloud and cloud_action is not None:
                        cloud_calls += 1
                    steps.append(
                        {
                            "step": step,
                            "viewpoint": vp,
                            "qwen_choice": qwen_out.get("choice"),
                            "qwen_action": qwen_action,
                            "cloud_action": cloud_action,
                            "used_cloud": bool(use_cloud and cloud_action is not None),
                            "router_score": router_score,
                            "label": label_info.get("label", 0.0),
                            "label_reason": label_info.get("reason", ""),
                            "one_step_reason": label_info.get("one_step_reason", ""),
                            "cv_utility": label_info.get("cv_utility", 0.0),
                            "label_cur_err": label_info.get("cur_err", 0.0),
                            "label_qwen_next_err": label_info.get("qwen_next_err", 0.0),
                            "label_cloud_next_err": label_info.get("cloud_next_err", 0.0),
                            "features": features,
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

                nav_error = min_dist_to_goals(dists, scan, vp, goals)
                oracle_error = min(min_dist_to_goals(dists, scan, p, goals) for p in path)
                success = 1.0 if nav_error < 3.0 else 0.0
                oracle_success = 1.0 if oracle_error < 3.0 else 0.0
                pred_len = path_length(dists, scan, path)
                spl = success * gt_len / max(pred_len, gt_len, 1e-6)
                rec = {
                    "task": args.task,
                    "instr_id": ep["instr_id"],
                    "scan": scan,
                    "instruction": ep["instruction"],
                    "pred_path": path,
                    "gt_path": ep["path"],
                    "goal_set": goals,
                    "teacher_path": teacher_path,
                    "nav_error": float(nav_error),
                    "oracle_error": float(oracle_error),
                    "success": success,
                    "oracle_success": oracle_success,
                    "spl": float(spl),
                    "path_len": float(pred_len),
                    "gt_len": float(gt_len),
                    "steps": steps,
                    "cloud_calls": cloud_calls,
                    "total_steps": max(len(steps), 1),
                    "cloud_call_rate": cloud_calls / max(len(steps), 1),
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
                    f"[{ep_idx}/{len(episodes)}] {ep['instr_id']} SR={sr:.2f} SPL={spl_mean:.2f} NE={ne:.2f} Cloud={ccr:.2f}",
                    flush=True,
                )
    finally:
        if worker is not None:
            worker.close()
        if samples_fout is not None:
            samples_fout.close()

    summary = {
        "task": args.task,
        "split": args.split,
        "n": len(metrics),
        "router_mode": args.router_mode,
        "cv_mode": "continuation_verified",
        "teacher_path_missing": int(teacher_path_missing),
        "teacher_path_missing_rate": float(teacher_path_missing / max(len(metrics), 1)),
        "sr": 100 * float(np.mean([m["success"] for m in metrics])) if metrics else 0.0,
        "oracle_sr": 100 * float(np.mean([m["oracle_success"] for m in metrics])) if metrics else 0.0,
        "spl": 100 * float(np.mean([m["spl"] for m in metrics])) if metrics else 0.0,
        "nav_error": float(np.mean([m["nav_error"] for m in metrics])) if metrics else 0.0,
        "cloud_call_rate": 100 * float(np.mean([m["cloud_call_rate"] for m in metrics])) if metrics else 0.0,
        "cloud_calls_per_ep": float(np.mean([m["cloud_calls"] for m in metrics])) if metrics else 0.0,
        "steps_per_ep": float(np.mean([len(m["steps"]) for m in metrics])) if metrics else 0.0,
        "results_path": str(results_path),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(out_dir / f"summary_{args.router_mode}_{args.split}_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
