# Edge-Cloud VLN Routing Supplementary Code

This repository contains the anonymized supplementary code for an edge-cloud
vision-and-language navigation (VLN) routing system. It includes the core model
adaptations, distillation utilities, router implementations, and experiment
drivers needed to reproduce the reported workflows once the external datasets
and checkpoints are placed in the expected locations.

## Repository Layout

```text
configs/                    Training and evaluation configs.
models/                     NaviLLM/Qwen model adaptations.
tasks/                      VLN task agents, datasets, and feature loading.
tools/                      Shared parsing, optimization, and metric helpers.
distill_code/               Student-model distillation code and scripts.
edgecloud_experiments/
  eval_edgecloud.py         Main edge-cloud evaluation entrypoint.
  routers/                  Entropy, divergence, off-course, and PPO routers.
  continuation_router/      Continuation-verified critical-state router.
  hetero_router/            R2R heterogeneous edge-cloud router.
  object_router/            REVERIE/SOON object-navigation router variants.
  utils/                    Router features, latency simulation, input helpers.
```

Large datasets, checkpoints, local build outputs, W&B logs, and
machine-specific artifacts are intentionally excluded.

## Expected External Assets

Prepare the original VLN data and simulator assets outside git, typically under:

```text
data/
  connectivity/
  R2R/
  REVERIE/
  SOON/
  eva_features/
  obj_features/
  models/

build/
  nav_ckpts/
  distill_training_stage2/checkpoints/
  router_data/checkpoints/
```

The scripts use relative paths by default where possible. For shell scripts,
override environment variables such as `ROOT`, `PYTHON`, `MATTERSIM_PYTHONPATH`,
`DATA_DIR`, `VICUNA_DIR`, `QWEN_VL_MODEL_DIR`, and `OFFICIAL_ROOT` when your
layout differs.

## Quick Smoke Commands

Distillation training:

```bash
python distill_code/train_distill.py \
  --cfg_file configs/multi.yaml \
  --pretrained_model_name_or_path data/models/Qwen3-1.7B \
  --distill_jsonl_paths build/distill_logs/r2r_train.jsonl \
  --output_dir build/distill_training
```

Edge-cloud R2R evaluation:

```bash
python edgecloud_experiments/eval_edgecloud.py \
  --task R2R \
  --split val_unseen \
  --cfg_file configs/multi.yaml \
  --data_dir data \
  --teacher_ckpt build/nav_ckpts/navillm_teacher.pt \
  --student_ckpt build/distill_training_stage2/checkpoints/student_best.pt \
  --router_ckpt build/router_data/checkpoints/router_best.pt \
  --pretrained_model_name_or_path data/models/Qwen3-1.7B \
  --mode edgecloud \
  --router_type offcourse \
  --student_type distill \
  --router_tau 0.5 \
  --output_dir build/edgecloud_results
```

Continuation-verified router workflow:

```bash
bash edgecloud_experiments/continuation_router/run_cv_smoke_r2r_v1.sh
```

## Anonymization Notes

- Fresh repository history should be created from this directory only.
- Commit authors should be anonymous, for example:
  `git -c user.name="Anonymous Authors" -c user.email="anonymous@example.com" commit`.
- Do not commit `data/`, `build/`, checkpoints, logs, local notes, or W&B run
  directories.
- W&B login keys are not stored in code; set `WANDB_API_KEY` only in your local
  environment if logging is needed.
- The optional Meteor paraphrase resource is omitted because it exceeds the
  normal GitHub single-file limit. It is not required for the edge-cloud routing
  workflows.
