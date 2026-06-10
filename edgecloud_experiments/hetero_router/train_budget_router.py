#!/usr/bin/env python3
"""Train a budget-aware heterogeneous R2R router.

The router is deliberately small and transparent.  It predicts whether a Qwen
edge decision should be deferred to the NaviLLM cloud advisor.  Thresholds are
selected after training to hit target teacher-call budgets.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
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


class RouterMLP(nn.Module):
    def __init__(self, dim, hidden=64, dropout=0.1):
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


def rows_to_xy(rows, feature_names):
    x = np.array([[float(r["features"].get(k, 0.0)) for k in feature_names] for r in rows], dtype=np.float32)
    y = np.array([float(r["label"]) for r in rows], dtype=np.float32)
    return x, y


def compute_utility(row):
    """Continuous intervention utility used for ranking critical steps.

    Binary labels are enough to say "cloud may help", but routing under a fixed
    budget needs the most valuable calls first.  We therefore give extra score
    to success flips, bad stops, and large one-step navigation-error gains.
    """
    reason = row.get("label_reason", "")
    qerr = float(row.get("label_qwen_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    cerr = float(row.get("label_cloud_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    gain = max(0.0, qerr - cerr)
    utility = 0.0
    utility += min(gain / 6.0, 0.55)
    if "success_flip" in reason:
        utility += 0.45
    if "bad_stop" in reason:
        utility += 0.30
    if "loop_break" in reason:
        utility += 0.15
    if float(row.get("label", 0.0)) <= 0.0:
        utility = 0.0
    return float(min(utility, 1.0))


def compute_critical_target(row):
    """Sharper target for ranking scarce cloud calls.

    The broad binary label is useful for high-budget routing, but it treats many
    mild one-step navigation-error gains like true make-or-break decisions.  For
    the method's core claim we need the top budget slots to focus on critical
    steps: bad stops, success flips, loop escapes, and large recovery gains.
    """
    if float(row.get("label", 0.0)) <= 0.0:
        return 0.0, 1.0
    reason = row.get("label_reason", "")
    qerr = float(row.get("label_qwen_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    cerr = float(row.get("label_cloud_next_err", row.get("label_cur_err", 0.0)) or 0.0)
    gain = max(0.0, qerr - cerr)

    target = 0.0
    weight = 1.0
    if "success_flip" in reason:
        target = max(target, 1.0)
        weight += 5.0
    if "bad_stop" in reason:
        target = max(target, 0.95)
        weight += 4.0
    if "loop_break" in reason:
        target = max(target, 0.85)
        weight += 3.0
    if gain >= 4.0:
        target = max(target, 0.90)
        weight += 3.0
    elif gain >= 2.0:
        target = max(target, 0.70)
        weight += 2.0
    elif gain >= 1.0:
        # Mild one-step gains remain positive evidence, but should not dominate
        # the low-budget critical-step frontier.
        target = max(target, 0.25)
        weight += 0.5
    return float(min(target, 1.0)), float(weight)


def rows_to_targets(rows, target_mode):
    if target_mode == "binary":
        y = np.array([float(r["label"]) for r in rows], dtype=np.float32)
        w = np.ones_like(y, dtype=np.float32)
        return y, w
    if target_mode == "critical":
        pairs = [compute_critical_target(r) for r in rows]
        y = np.array([p[0] for p in pairs], dtype=np.float32)
        w = np.array([p[1] for p in pairs], dtype=np.float32)
        return y, w
    utilities = np.array([compute_utility(r) for r in rows], dtype=np.float32)
    # Keep a small floor for positive binary labels whose immediate utility is
    # not large; they can still matter later, but should rank below clear flips.
    labels = np.array([float(r["label"]) for r in rows], dtype=np.float32)
    y = np.maximum(utilities, labels * 0.15)
    w = 1.0 + 4.0 * utilities + 1.0 * labels
    return y.astype(np.float32), w.astype(np.float32)


def metrics_at_threshold(scores, labels, threshold):
    pred = scores >= threshold
    labels = labels.astype(bool)
    tp = int(np.logical_and(pred, labels).sum())
    fp = int(np.logical_and(pred, ~labels).sum())
    fn = int(np.logical_and(~pred, labels).sum())
    tn = int(np.logical_and(~pred, ~labels).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "call_rate": float(pred.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def threshold_for_budget(scores, budget):
    if len(scores) == 0:
        return 1.0
    budget = min(max(float(budget), 0.0), 1.0)
    if budget <= 0:
        return float(scores.max() + 1e-6)
    if budget >= 1:
        return float(scores.min() - 1e-6)
    return float(np.quantile(scores, 1.0 - budget))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--budgets", default="0.05,0.10,0.20,0.30,0.40,0.50")
    ap.add_argument("--target_mode", default="binary", choices=["binary", "utility", "critical"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_samples(args.samples)
    rows = [r for r in rows if r.get("teacher_path_available", True)]
    if not rows:
        raise RuntimeError("No samples loaded")

    feature_names = sorted(rows[0]["features"].keys())
    train_rows, val_rows = split_by_episode(rows, args.val_ratio, args.seed)
    x_train, y_train = rows_to_xy(train_rows, feature_names)
    x_val, y_val = rows_to_xy(val_rows, feature_names)
    y_train_target, w_train = rows_to_targets(train_rows, args.target_mode)
    y_val_target, _ = rows_to_targets(val_rows, args.target_mode)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.maximum(std, 1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = RouterMLP(len(feature_names), args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([min(max(neg / max(pos, 1.0), 1.0), 20.0)], dtype=torch.float32)
    xtr = torch.from_numpy(x_train)
    ytr = torch.from_numpy(y_train_target)
    wtr = torch.from_numpy(w_train)
    xva = torch.from_numpy(x_val)
    yva = torch.from_numpy(y_val_target)

    best = None
    history = []
    indices = np.arange(len(x_train))
    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(indices)
        losses = []
        for start in range(0, len(indices), args.batch_size):
            idx = indices[start : start + args.batch_size]
            logits = model(xtr[idx])
            if args.target_mode == "binary":
                bce = F.binary_cross_entropy_with_logits(logits, ytr[idx], pos_weight=pos_weight)
            else:
                bce_raw = F.binary_cross_entropy_with_logits(logits, ytr[idx], reduction="none")
                bce = (bce_raw * wtr[idx]).mean()
            probs = torch.sigmoid(logits)
            # Budget regularizer discourages the degenerate "call almost never" or
            # "call everything" behavior while still letting thresholds set budgets.
            call_prior = probs.mean()
            reg = 0.02 * (call_prior - min(max(pos / max(pos + neg, 1.0), 0.1), 0.5)) ** 2
            loss = bce + reg
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_logits = model(xva)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = F.binary_cross_entropy_with_logits(val_logits, yva).item()
        eval_labels = y_val if args.target_mode == "binary" else (y_val_target >= 0.5).astype(np.float32)
        th = threshold_for_budget(val_probs, min(max(float(eval_labels.mean()), 0.05), 0.5))
        val_metrics = metrics_at_threshold(val_probs, eval_labels, th)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(val_loss),
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_call_rate": val_metrics["call_rate"],
        }
        history.append(rec)
        score = rec["val_f1"] - 0.05 * rec["val_loss"]
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    model.load_state_dict(best[2])
    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(xtr)).cpu().numpy()
        val_scores = torch.sigmoid(model(xva)).cpu().numpy()

    budget_thresholds = {}
    budget_metrics = {}
    eval_val_labels = y_val if args.target_mode == "binary" else (y_val_target >= 0.5).astype(np.float32)
    for b in [float(x) for x in args.budgets.split(",") if x.strip()]:
        key = f"b{int(round(b * 100)):02d}"
        th = threshold_for_budget(val_scores, b)
        budget_thresholds[key] = th
        budget_metrics[key] = metrics_at_threshold(val_scores, eval_val_labels, th)

    ckpt = {
        "model": model.state_dict(),
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "hidden": args.hidden,
        "budget_thresholds": budget_thresholds,
        "best_epoch": best[1],
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }
    torch.save(ckpt, out_dir / "hetero_router.pt")

    label_counts = Counter([r.get("label_reason", "unknown") for r in rows])
    critical_targets = [compute_critical_target(r)[0] for r in rows]
    summary = {
        "num_samples": len(rows),
        "num_episodes": len({r["instr_id"] for r in rows}),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "positive_rate": float(np.mean([r["label"] for r in rows])),
        "target_mode": args.target_mode,
        "utility_mean": float(np.mean([compute_utility(r) for r in rows])),
        "utility_positive_mean": float(np.mean([compute_utility(r) for r in rows if float(r["label"]) > 0.0])) if any(float(r["label"]) > 0.0 for r in rows) else 0.0,
        "critical_positive_rate": float(np.mean([t >= 0.5 for t in critical_targets])),
        "critical_target_mean": float(np.mean(critical_targets)),
        "best_epoch": best[1],
        "feature_names": feature_names,
        "label_counts": dict(label_counts),
        "budget_thresholds": budget_thresholds,
        "budget_metrics": budget_metrics,
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
