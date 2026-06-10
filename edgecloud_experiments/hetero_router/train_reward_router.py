#!/usr/bin/env python3
"""Train a reward-aware router for heterogeneous R2R edge-cloud routing.

This script is intentionally router-only.  It does not fine-tune the edge or
cloud navigators; instead it treats each collected step as a contextual bandit
state and optimizes a Bernoulli deferral policy with reward:

    r = call_cloud * (intervention_utility - cost_lambda)

The learned actor is saved in the same checkpoint format consumed by
eval_hetero_edgecloud_r2r.py, so it can be evaluated exactly like the supervised
binary/critical routers.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_samples(paths):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def split_by_episode(rows, val_ratio, seed):
    rng = random.Random(seed)
    ids = sorted({r["instr_id"] for r in rows})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = [r for r in rows if r["instr_id"] not in val_ids]
    val = [r for r in rows if r["instr_id"] in val_ids]
    return train, val


def parse_csv_set(value):
    return {x.strip() for x in str(value).split(",") if x.strip()}


def compute_utility(row, components=None):
    if components is None:
        components = {"naverr", "success_flip", "bad_stop", "loop_break"}
    reason = row.get("label_reason", "")
    qerr = float(row.get("label_qwen_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    cerr = float(row.get("label_cloud_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    gain = max(0.0, qerr - cerr)
    utility = min(gain / 6.0, 0.55) if "naverr" in components else 0.0
    if "success_flip" in components and "success_flip" in reason:
        utility += 0.45
    if "bad_stop" in components and "bad_stop" in reason:
        utility += 0.30
    if "loop_break" in components and "loop_break" in reason:
        utility += 0.15
    if float(row.get("label", 0.0)) <= 0.0:
        utility = 0.0
    return float(min(utility, 1.0))


def compute_critical(row):
    if float(row.get("label", 0.0)) <= 0.0:
        return 0.0
    reason = row.get("label_reason", "")
    qerr = float(row.get("label_qwen_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    cerr = float(row.get("label_cloud_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    gain = max(0.0, qerr - cerr)
    return float(
        "success_flip" in reason
        or "bad_stop" in reason
        or "loop_break" in reason
        or gain >= 2.0
    )


class RouterMLP(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CriticMLP(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def rows_to_arrays(rows, feature_names, utility_components=None):
    x = np.array([[float(r["features"].get(k, 0.0)) for k in feature_names] for r in rows], dtype=np.float32)
    utility = np.array([compute_utility(r, utility_components) for r in rows], dtype=np.float32)
    critical = np.array([compute_critical(r) for r in rows], dtype=np.float32)
    binary = np.array([float(r.get("label", 0.0)) for r in rows], dtype=np.float32)
    return x, utility, critical, binary


def threshold_for_budget(scores, budget):
    if len(scores) == 0:
        return 1.0
    budget = min(max(float(budget), 0.0), 1.0)
    if budget <= 0:
        return float(scores.max() + 1e-6)
    if budget >= 1:
        return float(scores.min() - 1e-6)
    return float(np.quantile(scores, 1.0 - budget))


def budget_metrics(scores, utility, critical, binary, budgets):
    out = {}
    for b in budgets:
        key = f"b{int(round(b * 100)):02d}"
        th = threshold_for_budget(scores, b)
        pred = scores >= th
        call_rate = float(pred.mean()) if len(pred) else 0.0
        out[key] = {
            "threshold": th,
            "call_rate": call_rate,
            "avg_utility_called": float(utility[pred].mean()) if pred.any() else 0.0,
            "utility_sum_per_100_steps": float(utility[pred].sum() / max(len(utility), 1) * 100.0),
            "critical_precision": float(critical[pred].mean()) if pred.any() else 0.0,
            "binary_precision": float(binary[pred].mean()) if pred.any() else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260505)
    ap.add_argument("--cost_lambda", type=float, default=0.35)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--critic_coef", type=float, default=0.5)
    ap.add_argument("--budgets", default="0.10,0.20,0.30,0.40,0.50")
    ap.add_argument("--feature_names", default="", help="Optional comma-separated feature subset for ablations.")
    ap.add_argument("--drop_feature_prefixes", default="", help="Optional comma-separated prefixes to remove from the feature list.")
    ap.add_argument(
        "--utility_components",
        default="naverr,success_flip,bad_stop,loop_break",
        help="Comma-separated reward components: naverr,success_flip,bad_stop,loop_break.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_samples(args.samples) if r.get("teacher_path_available", True)]
    if not rows:
        raise RuntimeError("No samples loaded")

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    if args.feature_names.strip():
        requested = [x.strip() for x in args.feature_names.split(",") if x.strip()]
        available = set(rows[0]["features"].keys())
        missing = [x for x in requested if x not in available]
        if missing:
            raise ValueError(f"Requested features are missing from samples: {missing}")
        feature_names = requested
    else:
        feature_names = sorted(rows[0]["features"].keys())
    drop_prefixes = [x.strip() for x in args.drop_feature_prefixes.split(",") if x.strip()]
    if drop_prefixes:
        feature_names = [f for f in feature_names if not any(f.startswith(p) for p in drop_prefixes)]
    utility_components = parse_csv_set(args.utility_components)
    allowed_components = {"naverr", "success_flip", "bad_stop", "loop_break"}
    if not utility_components.issubset(allowed_components):
        raise ValueError(f"Unknown utility components: {sorted(utility_components - allowed_components)}")
    train_rows, val_rows = split_by_episode(rows, args.val_ratio, args.seed)
    x_train, u_train, c_train, b_train = rows_to_arrays(train_rows, feature_names, utility_components)
    x_val, u_val, c_val, b_val = rows_to_arrays(val_rows, feature_names, utility_components)
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor = RouterMLP(len(feature_names), args.hidden, args.dropout)
    critic = CriticMLP(len(feature_names), args.hidden, args.dropout)
    opt = torch.optim.AdamW(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, weight_decay=1e-4)

    xtr = torch.from_numpy(x_train)
    utr = torch.from_numpy(u_train)
    xva = torch.from_numpy(x_val)
    indices = np.arange(len(x_train))
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        actor.train()
        critic.train()
        np.random.shuffle(indices)
        losses = []
        call_rates = []
        reward_means = []
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            xb = xtr[idx]
            ub = utr[idx]
            logits = actor(xb)
            probs = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
            dist = torch.distributions.Bernoulli(probs=probs)
            action = dist.sample()
            reward = action * (ub - float(args.cost_lambda))
            value = critic(xb)
            advantage = reward - value.detach()

            actor_loss = -(dist.log_prob(action) * advantage).mean()
            critic_loss = F.mse_loss(value, reward.detach())
            entropy = dist.entropy().mean()
            loss = actor_loss + args.critic_coef * critic_loss - args.entropy_coef * entropy

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            call_rates.append(float(action.mean().item()))
            reward_means.append(float(reward.mean().item()))

        actor.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(actor(xva)).cpu().numpy()
        metrics = budget_metrics(val_scores, u_val, c_val, b_val, budgets)
        # Prefer high-utility low/mid budget ranking because that is where the
        # critical-step story matters most.
        score = (
            metrics.get("b20", {}).get("utility_sum_per_100_steps", 0.0)
            + metrics.get("b30", {}).get("utility_sum_per_100_steps", 0.0)
            + 0.5 * metrics.get("b40", {}).get("utility_sum_per_100_steps", 0.0)
        )
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "sampled_call_rate": float(np.mean(call_rates)),
            "sampled_reward": float(np.mean(reward_means)),
            "selection_score": float(score),
            "budget_metrics": metrics,
        }
        history.append(rec)
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}, metrics)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    actor.load_state_dict(best[2])
    actor.eval()
    with torch.no_grad():
        val_scores = torch.sigmoid(actor(xva)).cpu().numpy()
    budget_thresholds = {}
    final_metrics = budget_metrics(val_scores, u_val, c_val, b_val, budgets)
    for b in budgets:
        key = f"b{int(round(b * 100)):02d}"
        budget_thresholds[key] = final_metrics[key]["threshold"]

    torch.save(
        {
            "model": actor.state_dict(),
            "feature_names": feature_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "hidden": args.hidden,
            "budget_thresholds": budget_thresholds,
            "best_epoch": best[1],
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "router_training": "reward_actor_critic",
            "cost_lambda": args.cost_lambda,
            "utility_components": sorted(utility_components),
        },
        out_dir / "hetero_router.pt",
    )

    label_counts = Counter([r.get("label_reason", "unknown") for r in rows])
    summary = {
        "num_samples": len(rows),
        "num_episodes": len({r["instr_id"] for r in rows}),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "cost_lambda": args.cost_lambda,
        "best_epoch": best[1],
        "feature_names": feature_names,
        "utility_components": sorted(utility_components),
        "utility_mean": float(np.mean([compute_utility(r, utility_components) for r in rows])),
        "critical_positive_rate": float(np.mean([compute_critical(r) for r in rows])),
        "binary_positive_rate": float(np.mean([float(r.get("label", 0.0)) for r in rows])),
        "label_counts": dict(label_counts),
        "budget_thresholds": budget_thresholds,
        "budget_metrics": final_metrics,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
