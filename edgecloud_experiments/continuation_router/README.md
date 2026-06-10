# Continuation-Verified Critical-Step Router

This directory is intentionally isolated from `edgecloud_experiments/hetero_router`.
It tests a new data-generation and router-training path designed to reduce the
off-policy mismatch between edge-only hindsight mining and deployed mixed
edge-cloud navigation.

## Core Idea

The old router labels a state mostly from one-step edge-vs-cloud differences on
an edge rollout. This new path follows the Roads-to-Rome style idea:

1. At a pre-action navigation state, compare edge and cloud actions.
2. If actions are identical, mark the state as non-critical.
3. If actions differ, execute each action as the first step and continue both
   branches with the same cloud continuation policy.
4. Mark the edge action as neutral if the edge-first branch keeps cloud-level
   navigation quality; mark it as divergent/critical if the cloud-first branch
   succeeds or preserves path quality while the edge-first branch fails,
   drifts, loops, or becomes inefficient.
5. In `oracle` mode, execute edge on neutral states and cloud on divergent
   states, producing an oracle-corrected mixed trajectory distribution.

This does not claim to fully solve long-horizon credit assignment. It is a
practical continuation-verification approximation that directly attacks the
main weakness of edge-only hindsight utility mining.

## Files

- `eval_cv_edgecloud_r2r.py`: R2R evaluator and sample collector with
  continuation-verified labels.
- `train_cv_group_router.py`: group-relative router trainer using
  continuation-verified utilities.
- `run_cv_smoke_r2r_v1.sh`: small end-to-end smoke run.
- `run_cv_collect_train_r2r_2000_v1.sh`: 4-shard train sample collection.
- `run_cv_train_eval_r2r_v1.sh`: merge samples, train routers, and launch
  validation evaluation.

## Outputs

Default output root:

`build/continuation_router`

Important subdirectories:

- `cv_smoke_r2r_v1`
- `cv_train_r2r_2000_v1`
- `cv_router_r2r_2000_v1`
- `cv_eval_val_unseen_v1`

