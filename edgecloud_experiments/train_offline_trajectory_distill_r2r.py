
"""Offline teacher-trajectory distillation for R2R.

This script deliberately separates the environment-sensitive NaviLLM teacher
from the Qwen student. Teacher decisions are collected with the official
NaviLLM environment (transformers==4.28.0) and saved as JSON. The student is
then trained in the modern Qwen environment by replaying those teacher decision
labels through the real navigation runtime path.
"""
import os
import sys
import copy
import json
import math
import logging
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple

import yaml
import torch
import numpy as np
import torch.nn as nn
from easydict import EasyDict
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Force local project `models` package to win over any installed module.
import importlib.util
models_init = project_root / 'models' / '__init__.py'
models_spec = importlib.util.spec_from_file_location('models', models_init, submodule_search_locations=[str(models_init.parent)])
models_pkg = importlib.util.module_from_spec(models_spec)
sys.modules['models'] = models_pkg
models_spec.loader.exec_module(models_pkg)
import models.ops  # noqa: F401

from tools.distributed import init_distributed_device
from tools.parser import random_seed
from tasks.feature_db import create_feature_db
from tasks.datasets import load_dataset
from tasks.loaders import build_dataloader, PrefetchLoader
from tasks.agents import load_agent
from models.graph_utils import GraphMap
from distill_code.models.nav_qwen3 import NavQwen3
from distill_code.train_distill import setup_freeze_strategy
from distill_code.train_runtime_distill_r2r_v2 import (
    get_autocast_context,
    get_cls_token,
    make_nav_inputs,
    update_graph_embeddings,
    save_runtime_checkpoint,
    build_optimizer,
    build_scheduler,
)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Offline teacher-decision R2R distillation for NavQwen3')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--teacher_decision_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--student_pretrained_model_name_or_path', type=str, required=True)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--distill_stage', choices=['stage1', 'stage2', 'stage3'], default='stage1')
    parser.add_argument('--validation_split', type=str, default='train')
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_step', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lr_head', type=float, default=1e-4)
    parser.add_argument('--lr_visual', type=float, default=5e-5)
    parser.add_argument('--lr_llm', type=float, default=1e-5)
    parser.add_argument('--lr_token', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=200)
    parser.add_argument('--precision', choices=['amp_bf16', 'amp_bfloat16', 'bf16', 'fp16', 'fp32'], default='amp_bf16')
    parser.add_argument('--feat_dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_every_epochs', type=int, default=1)
    parser.add_argument('--max_train_batches', type=int, default=-1)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--num_unfreeze_layers', type=int, default=4)
    parser.add_argument('--ignoreid', type=int, default=-100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--disable_nav', action='store_true')
    parser.add_argument('--path_type', type=str, default='trusted_path')
    parser.add_argument('--multi_endpoints', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--few_shot', type=int, default=None)
    parser.add_argument('--max_datapoints', type=int, default=None)
    parser.add_argument('--enable_og', action='store_true')
    parser.add_argument('--fuse_obj', action='store_true')
    parser.add_argument('--freeze_llama', action='store_true')
    parser.add_argument('--tune_token_emb', action='store_true')
    parser.add_argument('--no_loc_fts', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--tour3d_nav_head', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target', type=str, nargs='+', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--horovod', action='store_true', default=False)
    parser.add_argument('--no-set-device-rank', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--stage', type=str, default='multi')
    parser.add_argument('--num_steps_per_epoch', type=int, default=-1)
    parser.add_argument('--allow_oom_skip', action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)
    parser.add_argument('--clip_grad_norm_head', type=float, default=5.0)
    parser.add_argument('--clip_grad_norm_visual', type=float, default=1.0)
    parser.add_argument('--clip_grad_norm_llm', type=float, default=1.0)
    parser.add_argument('--clip_grad_norm_token', type=float, default=1.0)
    parser.add_argument('--grad_log_every', type=int, default=500)
    return parser.parse_args()


def resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str((project_root / p).resolve())


def setup_logging(output_dir: Path, rank: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('offline_traj_distill_r2r')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0:
        fh = logging.FileHandler(output_dir / 'log.txt')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_cfg(args):
    cfg_path = Path(resolve_path(args.cfg_file))
    with open(cfg_path, 'r', encoding='utf-8') as f:
        global_cfg = EasyDict(yaml.safe_load(f))
    args.cfg_file = str(cfg_path)
    args.data_dir = Path(resolve_path(args.data_dir))
    args.student_pretrained_model_name_or_path = resolve_path(args.student_pretrained_model_name_or_path)
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = resolve_path(args.resume_from_checkpoint)
    args.teacher_decision_json = resolve_path(args.teacher_decision_json)
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = global_cfg.Model.enc_full_graph
    args.expert_policy = global_cfg.Model.expert_policy
    args.num_pano_layers = global_cfg.Model.num_pano_layers
    return global_cfg


def build_dataset_cfg(global_cfg):
    dataset_cfg = copy.deepcopy(global_cfg.Dataset)
    dataset_cfg.update(copy.deepcopy(global_cfg.Multi))
    dataset_cfg.update(copy.deepcopy(global_cfg.Feature))
    dataset_cfg.SOURCE = ['R2R']
    dataset_cfg.Ratio = [1]
    return dataset_cfg


def build_r2r_loader(args, global_cfg, logger, device):
    feat_db = create_feature_db(global_cfg.Feature.feature_database, global_cfg.Feature.image_feat_size, args)
    dataset_cfg = build_dataset_cfg(global_cfg)
    # Train collection uses the official train split. For smoke tests we also
    # allow val_seen/val_unseen decision files and must load the matching split.
    is_train_split = args.validation_split == 'train'
    dataset = load_dataset('r2r', args, dataset_cfg, training=is_train_split, logger=logger, source='R2R')
    dataset.init_feat_db(feat_db=feat_db['mp3d'], obj_feat_db=None)
    loader, _ = build_dataloader(dataset, distributed=args.distributed, training=is_train_split, batch_size=args.batch_size, num_workers=args.workers)
    loader = PrefetchLoader(loader, device=device)
    agent = load_agent('r2r', args, getattr(dataset, 'shortest_distances', None), getattr(dataset, 'shortest_paths', None))
    return dataset, loader, agent


def load_model_checkpoint(model, ckpt_path, logger, strict=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=strict)
    logger.info(f'Loaded student checkpoint {ckpt_path}: {msg}')
    return ckpt


def build_student_model(args, global_cfg, logger, device):
    student_args = copy.deepcopy(args)
    student_args.pretrained_model_name_or_path = args.student_pretrained_model_name_or_path
    student_args.from_scratch = False
    student_model = NavQwen3(student_args, logger, global_cfg)
    if args.resume_from_checkpoint:
        load_model_checkpoint(student_model, args.resume_from_checkpoint, logger, strict=False)
    setup_freeze_strategy(student_model, args, stage=args.distill_stage)
    try:
        if hasattr(student_model.lang_model, 'gradient_checkpointing_enable'):
            student_model.lang_model.gradient_checkpointing_enable()
        elif hasattr(student_model.lang_model.model, 'gradient_checkpointing_enable'):
            student_model.lang_model.model.gradient_checkpointing_enable()
    except Exception as e:
        logger.warning(f'Could not enable gradient checkpointing: {e}')
    if hasattr(student_model.lang_model, 'config'):
        student_model.lang_model.config.use_cache = False
    if hasattr(student_model.lang_model, 'generation_config'):
        student_model.lang_model.generation_config.use_cache = False
    student_model.to(device)
    if args.distributed:
        student_model = DDP(student_model, device_ids=[args.local_rank], find_unused_parameters=True)
    return student_model


def load_teacher_decisions(path: str) -> Dict[str, List[Dict[str, Any]]]:
    data = json.load(open(path, 'r', encoding='utf-8'))
    out = {}
    bad = 0
    for item in data:
        instr_id = str(item.get('instr_id'))
        decisions = item.get('decisions', [])
        if not decisions:
            bad += 1
            continue
        out[instr_id] = decisions
        if not instr_id.startswith('r2r_'):
            out[f'r2r_{instr_id}'] = decisions
    if not out:
        raise ValueError(f'No decision records found in {path}; did you collect with the decision logger patch?')
    return out


def select_teacher_targets(
    obs,
    nav_inputs,
    teacher_decisions: Dict[str, List[Dict[str, Any]]],
    decision_pos: List[int],
    ignoreid: int,
    device,
):
    batch_size = len(obs)
    targets = torch.full((batch_size,), ignoreid, dtype=torch.long, device=device)
    selected_decisions: List[Optional[Dict[str, Any]]] = [None] * batch_size
    stats = {'missing_episode': 0, 'missing_state': 0, 'target_not_in_graph': 0, 'valid': 0, 'stop': 0}

    cand_masks = nav_inputs['gmap_masks'] & nav_inputs['gmap_visited_masks'].logical_not()
    for i, ob in enumerate(obs):
        instr_id = str(ob['instr_id'])
        decisions = teacher_decisions.get(instr_id)
        if not decisions:
            stats['missing_episode'] += 1
            continue
        cur_vp = ob['viewpoint']
        start = min(decision_pos[i], len(decisions) - 1)
        match_idx = None
        for j in range(start, len(decisions)):
            if decisions[j].get('viewpoint') == cur_vp:
                match_idx = j
                break
        if match_idx is None:
            for j in range(0, start):
                if decisions[j].get('viewpoint') == cur_vp:
                    match_idx = j
                    break
        if match_idx is None:
            stats['missing_state'] += 1
            continue
        decision_pos[i] = match_idx + 1
        dec = decisions[match_idx]
        selected_decisions[i] = dec
        action_vpid = dec.get('action_vpid')
        if dec.get('is_stop') or action_vpid is None:
            if cand_masks[i, 0]:
                targets[i] = 0
                stats['valid'] += 1
                stats['stop'] += 1
            else:
                stats['target_not_in_graph'] += 1
            continue
        vpids = nav_inputs['gmap_vpids'][i]
        target_idx = None
        for k, vpid in enumerate(vpids):
            if vpid == action_vpid:
                target_idx = k
                break
        if target_idx is None or target_idx >= cand_masks.size(1) or not bool(cand_masks[i, target_idx].item()):
            stats['target_not_in_graph'] += 1
            continue
        targets[i] = int(target_idx)
        stats['valid'] += 1
    return targets, selected_decisions, stats


def rollout_offline_distill(args, agent, batch_dict, dataset, student_model, teacher_decisions, criterion):
    obs = batch_dict['observations']
    envs = batch_dict['env']
    data_type = batch_dict['data_type']
    batch_size = len(obs)
    agent.update_scanvp_cands(obs)

    gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
    for i, ob in enumerate(obs):
        gmaps[i].update_graph(ob)

    traj = [{'instr_id': ob['instr_id'], 'path': [[ob['viewpoint']]], 'details': {}} for ob in obs]
    ended = np.array([False] * batch_size)
    decision_pos = [0] * batch_size
    instructions = [ob['instruction'] for ob in obs]
    history_tokens = [[] for _ in range(batch_size)]
    hist_vis = [[] for _ in range(batch_size)]

    total_loss = None
    total_acc = 0.0
    total_steps = 0
    action_steps = 0
    stat_totals = {'missing_episode': 0, 'missing_state': 0, 'target_not_in_graph': 0, 'valid': 0, 'stop': 0}

    student_core = student_model.module if hasattr(student_model, 'module') else student_model
    cls_token = get_cls_token(student_model)
    autocast_ctx = get_autocast_context(args)

    for t in range(args.max_action_len):
        if ended.all():
            break
        for i, gmap in enumerate(gmaps):
            if not ended[i]:
                gmap.node_step_ids[obs[i]['viewpoint']] = t + 1

        with autocast_ctx:
            pano_inputs = agent.panorama_feature_variable_object(obs)
            panorama_out = student_core('panorama', pano_inputs)
            pano_embeds = panorama_out['pano_embeds']
            pano_masks = panorama_out['pano_masks']
            avg = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / torch.sum(pano_masks, 1, keepdim=True)
        update_graph_embeddings(gmaps, obs, pano_inputs, pano_embeds, pano_masks, avg, ended)

        with autocast_ctx:
            nav_inputs = make_nav_inputs(agent, obs, gmaps, pano_inputs, pano_embeds, pano_masks, instructions, history_tokens, hist_vis, data_type, cls_token)
            nav_outs = student_core('navigation', nav_inputs, disable_cand_permute=True)
            student_logits = nav_outs['fuse_logits']
            targets, selected_decisions, step_stats = select_teacher_targets(obs, nav_inputs, teacher_decisions, decision_pos, args.ignoreid, student_logits.device)
            for k in stat_totals:
                stat_totals[k] += step_stats[k]
            valid_mask = targets != args.ignoreid
            if valid_mask.any():
                step_loss = criterion(student_logits, targets)
                total_loss = step_loss if total_loss is None else total_loss + step_loss
                preds = student_logits.argmax(dim=1)
                total_acc += (preds[valid_mask] == targets[valid_mask]).float().mean().item()
                total_steps += 1
                action_steps += int(valid_mask.sum().item())

        # Follow the teacher decision labels. Samples with missing labels are ended
        # immediately so they cannot drift into off-policy states.
        cpu_a_t = []
        for i in range(batch_size):
            target = int(targets[i].detach().cpu().item())
            if target == args.ignoreid or ended[i] or nav_inputs['no_vp_left'][i] or target == 0 or (t == args.max_action_len - 1):
                cpu_a_t.append(None)
            else:
                cpu_a_t.append(nav_inputs['gmap_vpids'][i][target])
                history_tokens[i].append('<hist>')
                hist_vis[i].append(nav_outs['fuse_embeds'][i][target].detach())

        agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=envs)
        new_obs = []
        for b_i in range(batch_size):
            new_obs.append(dataset.get_obs(items=[batch_dict['item'][b_i]], env=envs[b_i], data_type=data_type[b_i])[0])
        obs = new_obs
        agent.update_scanvp_cands(obs)
        for i, ob in enumerate(obs):
            if not ended[i]:
                gmaps[i].update_graph(ob)
        ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

    if total_loss is None:
        # Keep graph connected for DDP even if this rare batch has no valid labels.
        dummy = sum((p.sum() * 0.0) for p in student_core.parameters() if p.requires_grad)
        total_loss = dummy
    else:
        total_loss = total_loss / max(total_steps, 1)

    metrics = {
        'acc': total_acc / max(total_steps, 1),
        'loss_steps': total_steps,
        'action_steps': action_steps,
        **{f'label_{k}': v for k, v in stat_totals.items()},
    }
    return total_loss, metrics


def clip_optimizer_gradients(optimizer, args) -> Dict[str, float]:
    """Clip gradients per optimizer group.

    A single global clip can be dominated by very large LoRA/visual gradients,
    which effectively zeroes the navigation head update. Group-wise clipping
    keeps each trainable subsystem numerically sane without letting one group
    suppress the others.
    """
    max_norm_by_name = {
        'head': args.clip_grad_norm_head,
        'visual': args.clip_grad_norm_visual,
        'llm': args.clip_grad_norm_llm,
        'token_emb': args.clip_grad_norm_token,
    }
    grad_norms = {}
    for group in optimizer.param_groups:
        name = group.get('name', 'default')
        params = [p for p in group['params'] if p.grad is not None]
        if not params:
            grad_norms[name] = 0.0
            continue
        max_norm = max_norm_by_name.get(name, args.clip_grad_norm)
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        grad_norms[name] = float(norm.detach().cpu().item() if torch.is_tensor(norm) else norm)
    return grad_norms


def main():
    args = parse_args()
    device = init_distributed_device(args)
    random_seed(args.seed, args.rank if hasattr(args, 'rank') else 0)
    global_cfg = load_cfg(args)
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir, args.rank)
    logger.info('=== Offline teacher-decision R2R distillation ===')
    logger.info(json.dumps({
        'teacher_decision_json': args.teacher_decision_json,
        'student_base': args.student_pretrained_model_name_or_path,
        'stage': args.distill_stage,
        'use_lora': args.use_lora,
        'tune_token_emb': args.tune_token_emb,
        'precision': args.precision,
    }, indent=2))

    teacher_decisions = load_teacher_decisions(args.teacher_decision_json)
    logger.info(f'Loaded teacher decisions for {len(teacher_decisions)} instructions')
    dataset, loader, agent = build_r2r_loader(args, global_cfg, logger, device)
    student_model = build_student_model(args, global_cfg, logger, device)
    optimizer = build_optimizer(student_model, args, logger)
    total_batches = loader.num_batches if args.max_train_batches < 0 else min(loader.num_batches, args.max_train_batches)
    total_steps = max(total_batches * args.num_epochs // max(args.gradient_accumulation_step, 1), 1)
    args.lr_scheduler = 'cosine'
    scheduler = build_scheduler(optimizer, args, total_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='mean')

    metrics_history = []
    best_acc = -1.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_epochs):
        if hasattr(loader, 'loader') and hasattr(loader.loader, 'sampler') and hasattr(loader.loader.sampler, 'set_epoch'):
            loader.loader.sampler.set_epoch(epoch)
        student_model.train()
        pbar = tqdm(total=total_batches, disable=args.rank != 0, desc=f'epoch {epoch+1}/{args.num_epochs} [{args.distill_stage}]')
        agg = {'loss': 0.0, 'acc': 0.0, 'count': 0, 'oom': 0, 'loss_steps': 0, 'action_steps': 0,
               'label_missing_episode': 0, 'label_missing_state': 0, 'label_target_not_in_graph': 0, 'label_valid': 0, 'label_stop': 0}
        recent_losses, recent_accs = [], []
        iterator = iter(loader)
        for step in range(total_batches):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            sync_ctx = student_model.no_sync if hasattr(student_model, 'no_sync') and ((step + 1) % args.gradient_accumulation_step != 0) else nullcontext
            try:
                with sync_ctx():
                    loss, metrics = rollout_offline_distill(args, agent, batch, dataset, student_model, teacher_decisions, criterion)
                    scaled_loss = loss / max(args.gradient_accumulation_step, 1)
                    scaled_loss.backward()
                if (step + 1) % args.gradient_accumulation_step == 0:
                    grad_norms = clip_optimizer_gradients(optimizer, args)
                    if args.rank == 0 and (step < args.gradient_accumulation_step or ((step + 1) // args.gradient_accumulation_step) % max(args.grad_log_every, 1) == 0):
                        logger.info(f"preclip grad norms at epoch={epoch+1} step={step+1}: {grad_norms}")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    torch.cuda.empty_cache()
                agg['loss'] += float(loss.detach().item())
                agg['acc'] += metrics['acc']
                agg['count'] += 1
                recent_losses.append(float(loss.detach().item()))
                recent_accs.append(float(metrics['acc']))
                if len(recent_losses) > 200:
                    recent_losses.pop(0)
                    recent_accs.pop(0)
                for k in ['loss_steps', 'action_steps', 'label_missing_episode', 'label_missing_state', 'label_target_not_in_graph', 'label_valid', 'label_stop']:
                    agg[k] += int(metrics.get(k, 0))
                if args.rank == 0:
                    pbar.set_postfix({
                        'loss': agg['loss'] / max(agg['count'], 1),
                        'acc': agg['acc'] / max(agg['count'], 1),
                        'w_loss': sum(recent_losses) / max(len(recent_losses), 1),
                        'w_acc': sum(recent_accs) / max(len(recent_accs), 1),
                        'valid': agg['label_valid'],
                        'miss': agg['label_missing_state'] + agg['label_target_not_in_graph'],
                        'lr': scheduler.get_last_lr()[0],
                        'oom': agg['oom'],
                    })
                    pbar.update(1)
            except torch.cuda.OutOfMemoryError:
                agg['oom'] += 1
                logger.exception(f'OOM at epoch={epoch} step={step}')
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if not args.allow_oom_skip:
                    raise
                continue
            except Exception as e:
                logger.exception(f'Error at epoch={epoch} step={step}: {e}')
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                raise
        if args.rank == 0:
            pbar.close()
            epoch_metrics = {
                'epoch': epoch + 1,
                'stage': args.distill_stage,
                'loss': agg['loss'] / max(agg['count'], 1),
                'acc': agg['acc'] / max(agg['count'], 1),
                'oom_batches': agg['oom'],
                'lr': scheduler.get_last_lr()[0],
                'batches': agg['count'],
                'loss_steps': agg['loss_steps'],
                'action_steps': agg['action_steps'],
                'label_valid': agg['label_valid'],
                'label_stop': agg['label_stop'],
                'label_missing_episode': agg['label_missing_episode'],
                'label_missing_state': agg['label_missing_state'],
                'label_target_not_in_graph': agg['label_target_not_in_graph'],
            }
            metrics_history.append(epoch_metrics)
            (output_dir / 'training_history.json').write_text(json.dumps(metrics_history, indent=2), encoding='utf-8')
            logger.info(f'Epoch {epoch+1} metrics: {epoch_metrics}')
            save_runtime_checkpoint(student_model, optimizer, epoch, output_dir, args.distill_stage, tag=f'epoch_{epoch+1}', extra={'metrics': epoch_metrics})
            save_runtime_checkpoint(student_model, optimizer, epoch, output_dir, args.distill_stage, tag='latest', extra={'metrics': epoch_metrics})
            if epoch_metrics['acc'] > best_acc:
                best_acc = epoch_metrics['acc']
                save_runtime_checkpoint(student_model, optimizer, epoch, output_dir, args.distill_stage, tag='best', extra={'metrics': epoch_metrics})
    if args.rank == 0:
        logger.info('Offline trajectory distillation finished.')


if __name__ == '__main__':
    main()
