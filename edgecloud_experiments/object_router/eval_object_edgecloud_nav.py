#!/usr/bin/env python3
"""Navigation-only edge-cloud routing for REVERIE/SOON-style object VLN.

This evaluator intentionally focuses on navigation SR/SPL/NavErr and cloud-call
cost. Object grounding metrics are dataset-specific and are not the core signal
for the edge-cloud feasibility question.
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
from edgecloud_experiments.object_router.eval_qwen25vl_objectnav_smoke import (  # noqa: E402
    build_qwen_prompt,
    load_episodes,
    min_dist_to_goals,
    path_length,
)


def load_teacher_trajectories(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    out = {}
    for rec in data:
        instr_id = str(rec.get("instr_id"))
        traj = rec.get("trajectory", [])
        vpids = []
        # REVERIE uses a flat trajectory list; SOON may wrap path metadata.
        if traj and isinstance(traj[0], dict) and "path" in traj[0]:
            traj = traj[0]["path"]
        for x in traj:
            if isinstance(x, (list, tuple)) and x:
                vpids.append(x[0])
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


def compute_label(scan, goals, current_vp, path, candidates, qwen_action, cloud_action, dists, min_gain):
    cur_err = min_dist_to_goals(dists, scan, current_vp, goals)
    qwen_next = action_next_vp(qwen_action, current_vp, candidates)
    cloud_next = action_next_vp(cloud_action, current_vp, candidates)
    qwen_err = min_dist_to_goals(dists, scan, qwen_next, goals)
    cloud_err = min_dist_to_goals(dists, scan, cloud_next, goals)
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
            budget_thresholds = ckpt.get("budget_thresholds")
            if not budget_thresholds or args.budget_key not in budget_thresholds:
                available = sorted((budget_thresholds or {}).keys())
                raise KeyError(
                    f"budget_key={args.budget_key!r} not found in checkpoint thresholds; "
                    f"available={available}. Pass --trained_threshold explicitly to override."
                )
            threshold = budget_thresholds[args.budget_key]
        return {
            "torch": torch,
            "model": model,
            "feature_names": feature_names,
            "mean": np.array(ckpt["mean"], dtype=np.float32),
            "std": np.maximum(np.array(ckpt["std"], dtype=np.float32), 1e-6),
            "threshold": float(threshold),
            "budget_key": args.budget_key,
        }

    def score(self, features):
        if self.trained is not None:
            values = np.array(
                [[float(features.get(k, 0.0)) for k in self.trained["feature_names"]]],
                dtype=np.float32,
            )
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
            label = bool(label_info and label_info.get("label", 0.0) > 0.5)
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
    ap.add_argument(
        "--cloud_policy",
        default="teacher_replay",
        choices=["teacher_replay"],
        help="Cloud action source. teacher_replay is an offline teacher trajectory surrogate and must not be confused with online cloud inference.",
    )
    ap.add_argument("--router_mode", default="small", choices=["small", "cloud", "random", "heuristic", "oracle", "trained"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--router_checkpoint", default="")
    ap.add_argument("--budget_key", default="b40")
    ap.add_argument("--trained_threshold", type=float, default=None)
    ap.add_argument("--min_gain", type=float, default=1.0)
    ap.add_argument("--out_dir", default="build/object_router/edgecloud_nav_eval")
    ap.add_argument("--samples_out", default="")
    ap.add_argument("--strict_teacher_paths", action="store_true")
    ap.add_argument("--keep_images", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.task.lower() / args.router_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = out_dir / f"images_{args.split}_{int(time.time())}"
    img_root.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"object_edgecloud_{args.task.lower()}_{args.router_mode}_{args.split}_max{args.max_episodes}_{int(time.time())}.jsonl"

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
                teacher_path = teacher_path or []
                if not teacher_path:
                    teacher_path_missing += 1
                    if args.strict_teacher_paths:
                        raise RuntimeError(f"missing teacher trajectory for instr_id={ep['instr_id']}")

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
                        label_info = {"label": 0.0, "reason": "cloud_only"}
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
                        label_info = compute_label(scan, goals, vp, path, candidates, qwen_action, cloud_action, dists, args.min_gain)
                        use_cloud, router_score = router.should_call(features, label_info, rng)
                        if samples_fout is not None:
                            samples_fout.write(json.dumps({
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
                                "label_cur_err": label_info.get("cur_err", 0.0),
                                "label_qwen_next_err": label_info.get("qwen_next_err", 0.0),
                                "label_cloud_next_err": label_info.get("cloud_next_err", 0.0),
                                "qwen_action": qwen_action,
                                "cloud_action": cloud_action,
                                "qwen_choice": qwen_out.get("choice"),
                                "num_candidates": len(candidates),
                            }, ensure_ascii=False) + "\n")
                            samples_fout.flush()

                    action = cloud_action if use_cloud and cloud_action is not None else qwen_action
                    if use_cloud and cloud_action is not None:
                        cloud_calls += 1
                    steps.append({
                        "step": step,
                        "viewpoint": vp,
                        "qwen_choice": qwen_out.get("choice"),
                        "qwen_action": qwen_action,
                        "cloud_action": cloud_action,
                        "used_cloud": bool(use_cloud and cloud_action is not None),
                        "router_score": router_score,
                        "label": label_info.get("label", 0.0),
                        "label_reason": label_info.get("reason", ""),
                        "label_cur_err": label_info.get("cur_err", 0.0),
                        "label_qwen_next_err": label_info.get("qwen_next_err", 0.0),
                        "label_cloud_next_err": label_info.get("cloud_next_err", 0.0),
                        "features": features,
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

                nav_error = min_dist_to_goals(dists, scan, vp, goals)
                oracle_error = min(min_dist_to_goals(dists, scan, p, goals) for p in path)
                success = 1.0 if nav_error < 3.0 else 0.0
                oracle_success = 1.0 if oracle_error < 3.0 else 0.0
                gt_len = path_length(dists, scan, ep["path"])
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
                    "nav_error": nav_error,
                    "oracle_error": oracle_error,
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
                print(f"[{ep_idx}/{len(episodes)}] {ep['instr_id']} SR={sr:.2f} SPL={spl_mean:.2f} NE={ne:.2f} Cloud={ccr:.2f}", flush=True)
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
        "cloud_policy": args.cloud_policy,
        "cloud_policy_note": "teacher_replay is an offline teacher-trajectory surrogate, not an online cloud model",
        "teacher_path_missing": int(teacher_path_missing),
        "teacher_path_missing_rate": float(teacher_path_missing / max(len(metrics), 1)),
        "sr": 100 * float(np.mean([m["success"] for m in metrics])) if metrics else 0.0,
        "oracle_sr": 100 * float(np.mean([m["oracle_success"] for m in metrics])) if metrics else 0.0,
        "spl": 100 * float(np.mean([m["spl"] for m in metrics])) if metrics else 0.0,
        "nav_error": float(np.mean([m["nav_error"] for m in metrics])) if metrics else 0.0,
        "cloud_call_rate": 100 * float(np.mean([m["cloud_call_rate"] for m in metrics])) if metrics else 0.0,
        "cloud_calls_per_ep": float(np.mean([m["cloud_calls"] for m in metrics])) if metrics else 0.0,
        "results_path": str(results_path),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    with open(out_dir / f"summary_{args.router_mode}_{args.split}_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
