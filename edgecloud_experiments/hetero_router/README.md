# Heterogeneous Router for Edge-Cloud R2R

This directory is a clean router stack for the current project direction:

- Edge model: `Qwen2.5-VL-3B-R2R-panoramic`, using RGB panorama and candidate-view prompts.
- Cloud model: trusted NaviLLM teacher trajectory / future online NaviLLM service.
- Shared interface: both models act on the same R2R simulator state and choose the next candidate viewpoint or `Stop`.

The goal is not to preserve the old Qwen3 distillation router.  The goal is to
learn a budget-aware deferral policy that improves `SR/SPL` over Qwen2.5-VL
while using NaviLLM only at critical steps.

## Pipeline

1. Run `eval_hetero_edgecloud_r2r.py --router_mode small --samples_out ...` to collect Qwen decisions, confidence features, and intervention labels.
2. Train a router with `train_budget_router.py`.
3. Evaluate Pareto points with `eval_hetero_edgecloud_r2r.py --router_mode trained --router_ckpt ... --budget ...`.
4. Compare against `small`, `cloud`, `random`, `heuristic`, and `oracle` modes.

## Why This Is Separate

The old router assumes homogeneous action logits/hidden states from a distilled
NaviLLM-like student.  Qwen2.5-VL-R2R is a stronger but heterogeneous edge model,
so the router must use model-agnostic state/confidence features and evaluate in
the real simulator loop.

