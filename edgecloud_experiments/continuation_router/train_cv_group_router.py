#!/usr/bin/env python3
"""Train a group-relative router from continuation-verified samples."""

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


def compute_utility(row):
    if "cv_utility" in row:
        return float(row.get("cv_utility") or 0.0)
    if float(row.get("label", 0.0)) <= 0.0:
        return 0.0
    reason = row.get("label_reason", "")
    utility = 0.0
    if "success" in reason:
        utility += 0.50
    if "spl" in reason:
        utility += 0.25
    if "naverr" in reason:
        utility += 0.20
    if "loop" in reason or "pathlen" in reason or "bad_stop" in reason:
        utility += 0.15
    return float(min(1.0, utility))


def compute_critical(row):
    return float(float(row.get("label", 0.0)) > 0.5)


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


def rows_to_arrays(rows, feature_names):
    x = np.array([[float(r["features"].get(k, 0.0)) for k in feature_names] for r in rows], dtype=np.float32)
    utility = np.array([compute_utility(r) for r in rows], dtype=np.float32)
    critical = np.array([compute_critical(r) for r in rows], dtype=np.float32)
    binary = np.array([float(r.get("label", 0.0)) for r in rows], dtype=np.float32)
    return x, utility, critical, binary


def group_episodes(rows, feature_names, mean, std):
    groups = []
    by_id = {}
    for r in rows:
        by_id.setdefault(r["instr_id"], []).append(r)
    for instr_id, ep_rows in by_id.items():
        ep_rows = sorted(ep_rows, key=lambda r: int(r.get("step", 0)))
        x, u, c, b = rows_to_arrays(ep_rows, feature_names)
        x = (x - mean) / np.maximum(std, 1e-6)
        groups.append(
            {
                "instr_id": instr_id,
                "x": torch.from_numpy(x),
                "utility": torch.from_numpy(u),
                "critical": torch.from_numpy(c),
                "binary": torch.from_numpy(b),
                "length": len(ep_rows),
            }
        )
    return groups


def positive_rank_loss(logits, labels, utility):
    pos = labels > 0.5
    neg = labels <= 0.5
    if int(pos.sum().item()) == 0 or int(neg.sum().item()) == 0:
        return logits.new_tensor(0.0)
    pos_logits = logits[pos]
    neg_logits = logits[neg]
    pos_u = utility[pos]
    neg_u = utility[neg]
    diff = pos_logits[:, None] - neg_logits[None, :]
    weights = (pos_u[:, None] - neg_u[None, :]).clamp(min=0.05, max=1.0)
    return (F.softplus(-diff) * weights).mean()


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
        out[key] = {
            "threshold": th,
            "call_rate": float(pred.mean()) if len(pred) else 0.0,
            "avg_utility_called": float(utility[pred].mean()) if pred.any() else 0.0,
            "utility_sum_per_100_steps": float(utility[pred].sum() / max(len(utility), 1) * 100.0),
            "critical_precision": float(critical[pred].mean()) if pred.any() else 0.0,
            "binary_precision": float(binary[pred].mean()) if pred.any() else 0.0,
        }
    return out


def score_on_rows(actor, rows, feature_names, mean, std, budgets):
    x, utility, critical, binary = rows_to_arrays(rows, feature_names)
    x = (x - mean) / np.maximum(std, 1e-6)
    actor.eval()
    with torch.no_grad():
        scores = torch.sigmoid(actor(torch.from_numpy(x))).cpu().numpy()
    return scores, budget_metrics(scores, utility, critical, binary, budgets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--episodes_per_batch", type=int, default=64)
    ap.add_argument("--rollouts_per_episode", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260512)
    ap.add_argument("--cost_lambda", type=float, default=0.30)
    ap.add_argument("--target_budget", type=float, default=0.40)
    ap.add_argument("--budget_penalty", type=float, default=0.30)
    ap.add_argument("--policy_coef", type=float, default=1.0, help="Weight for the group-relative policy objective. Use 0 for aligned supervised/ranking baselines.")
    ap.add_argument("--entropy_coef", type=float, default=0.005)
    ap.add_argument("--supervised_coef", type=float, default=0.0)
    ap.add_argument("--rank_coef", type=float, default=0.0)
    ap.add_argument("--utility_reg_coef", type=float, default=0.0)
    ap.add_argument("--budgets", default="0.10,0.20,0.30,0.40,0.50")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_samples(args.samples) if r.get("teacher_path_available", True)]
    rows = [r for r in rows if r.get("features")]
    if not rows:
        raise RuntimeError("No usable samples loaded")

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    feature_names = sorted(rows[0]["features"].keys())
    train_rows, val_rows = split_by_episode(rows, args.val_ratio, args.seed)
    x_train, _, _, _ = rows_to_arrays(train_rows, feature_names)
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-6)
    train_groups = group_episodes(train_rows, feature_names, mean, std)
    train_positive_rate = float(np.mean([compute_critical(r) for r in train_rows]))
    pos_weight = (1.0 - train_positive_rate) / max(train_positive_rate, 1e-6)
    utility_scale = max(float(np.percentile([compute_utility(r) for r in train_rows], 95)), 1e-3)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    actor = RouterMLP(len(feature_names), args.hidden, args.dropout)
    opt = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        actor.train()
        rng.shuffle(train_groups)
        losses = []
        returns = []
        call_rates = []
        for start in range(0, len(train_groups), args.episodes_per_batch):
            batch = train_groups[start : start + args.episodes_per_batch]
            opt.zero_grad()
            batch_loss = 0.0
            batch_entropy = 0.0
            batch_items = 0
            for ep in batch:
                x = ep["x"]
                u = ep["utility"]
                labels = ep["binary"]
                logits = actor(x)
                probs = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
                dist = torch.distributions.Bernoulli(probs=probs)
                actions = dist.sample((args.rollouts_per_episode,))
                log_probs = dist.log_prob(actions).sum(dim=1)
                call_rate = actions.mean(dim=1)
                step_reward = actions * (u.unsqueeze(0) - float(args.cost_lambda))
                seq_return = step_reward.sum(dim=1)
                over_budget = torch.relu(call_rate - float(args.target_budget))
                seq_return = seq_return - float(args.budget_penalty) * over_budget.pow(2) * max(float(ep["length"]), 1.0)
                adv = seq_return - seq_return.mean()
                adv = adv / (seq_return.std(unbiased=False) + 1e-6)
                if args.policy_coef > 0:
                    batch_loss = batch_loss - float(args.policy_coef) * (log_probs * adv.detach()).mean()
                batch_entropy = batch_entropy + dist.entropy().mean()
                if args.supervised_coef > 0:
                    bce = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=logits.new_tensor(pos_weight),
                    )
                    batch_loss = batch_loss + float(args.supervised_coef) * bce
                if args.rank_coef > 0:
                    batch_loss = batch_loss + float(args.rank_coef) * positive_rank_loss(logits, labels, u)
                if args.utility_reg_coef > 0:
                    target = (u / utility_scale).clamp(0.0, 1.0)
                    batch_loss = batch_loss + float(args.utility_reg_coef) * F.mse_loss(probs, target)
                batch_items += 1
                returns.append(float(seq_return.mean().item()))
                call_rates.append(float(call_rate.mean().item()))
            loss = batch_loss / max(batch_items, 1) - float(args.entropy_coef) * batch_entropy / max(batch_items, 1)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        val_scores, metrics = score_on_rows(actor, val_rows, feature_names, mean, std, budgets)
        score = (
            1.25 * metrics.get("b30", {}).get("utility_sum_per_100_steps", 0.0)
            + 1.00 * metrics.get("b40", {}).get("utility_sum_per_100_steps", 0.0)
            + 0.50 * metrics.get("b50", {}).get("utility_sum_per_100_steps", 0.0)
        )
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "sampled_return": float(np.mean(returns)),
            "sampled_call_rate": float(np.mean(call_rates)),
            "selection_score": float(score),
            "budget_metrics": metrics,
        }
        history.append(rec)
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}, metrics)
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    actor.load_state_dict(best[2])
    scores, final_metrics = score_on_rows(actor, val_rows, feature_names, mean, std, budgets)
    budget_thresholds = {
        f"b{int(round(b * 100)):02d}": final_metrics[f"b{int(round(b * 100)):02d}"]["threshold"]
        for b in budgets
    }
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
            "router_training": "continuation_verified_group_relative_policy_gradient",
            "cost_lambda": args.cost_lambda,
            "target_budget": args.target_budget,
            "budget_penalty": args.budget_penalty,
            "policy_coef": args.policy_coef,
            "supervised_coef": args.supervised_coef,
            "rank_coef": args.rank_coef,
            "utility_reg_coef": args.utility_reg_coef,
            "utility_scale": utility_scale,
        },
        out_dir / "hetero_router.pt",
    )
    label_counts = Counter([r.get("label_reason", "unknown") for r in rows])
    one_step_counts = Counter([r.get("one_step_reason", "unknown") for r in rows])
    summary = {
        "num_samples": len(rows),
        "num_episodes": len({r["instr_id"] for r in rows}),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "cost_lambda": args.cost_lambda,
        "target_budget": args.target_budget,
        "budget_penalty": args.budget_penalty,
        "policy_coef": args.policy_coef,
        "supervised_coef": args.supervised_coef,
        "rank_coef": args.rank_coef,
        "utility_reg_coef": args.utility_reg_coef,
        "train_positive_rate": train_positive_rate,
        "pos_weight": pos_weight,
        "utility_scale": utility_scale,
        "rollouts_per_episode": args.rollouts_per_episode,
        "best_epoch": best[1],
        "feature_names": feature_names,
        "label_counts": dict(label_counts),
        "one_step_counts": dict(one_step_counts),
        "positive_rate": float(np.mean([float(r.get("label", 0.0)) for r in rows])),
        "mean_cv_utility": float(np.mean([compute_utility(r) for r in rows])),
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
