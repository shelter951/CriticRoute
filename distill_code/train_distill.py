"""
Training script for knowledge distillation from Teacher (Vicuna-7B) to Student (Qwen3-1.7B)
"""
import os
import json
import torch
import random
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
from pathlib import Path

# 添加项目根目录到Python路径（必须在其他导入之前）
# 这样无论从哪个目录运行脚本都能正确导入
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.common_utils import all_gather
from tools.parser import random_seed
from distill_code.parse_distill_args import read_distill_args

# 现在可以安全导入项目模块了
from tasks.feature_db import create_feature_db
from distill_code.models.nav_qwen3 import NavQwen3
from models.image_embedding import ImageEmbeddings
from distill_code.datasets.distill_dataset import DistillDataset
from tools.optims import save_checkpoint
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup as get_cosine_scheduler
import logging
import yaml
from easydict import EasyDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self):
        self.num = 0
        self.total = 0

    def accumulate(self, x):
        self.num += 1
        self.total += x

    @property
    def average(self):
        if self.num == 0:
            return 0
        return self.total / self.num

    def reset(self):
        self.num = 0
        self.total = 0


def setup_freeze_strategy(model, args, stage='stage1'):
    """
    Setup parameter freezing strategy for staged training
    
    Args:
        model: NavQwen3 model
        args: Training arguments
        stage: 'stage1' (conservative) or 'stage2' (more layers) or 'stage3' (all)
    """
    # Unfreeze all first
    for param in model.parameters():
        param.requires_grad = True
    
    if stage == 'stage1':
        # Stage 1: Only train classification head and visual projection layers
        logger.info("Stage 1: Freezing most layers, only training head and visual layers")
        
        # Freeze LLM
        for name, param in model.lang_model.model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze token embeddings if needed
        if args.tune_token_emb:
            for name, param in model.lang_model.get_input_embeddings().named_parameters():
                param.requires_grad = True
        
        # Unfreeze classification head
        for param in model.out_head.parameters():
            param.requires_grad = True
        
        # Unfreeze visual embeddings
        for param in model.img_embeddings.parameters():
            param.requires_grad = True
        
        # Unfreeze other projection layers
        for param in model.gmap_pos_embeddings.parameters():
            param.requires_grad = True
        for param in model.vp_pos_embeddings.parameters():
            param.requires_grad = True
        for param in model.token_type_embeddings.parameters():
            param.requires_grad = True
        
    elif stage == 'stage2':
        # Stage 2: Unfreeze last N layers of LLM
        logger.info("Stage 2: Unfreezing last layers of LLM")
        
        # Freeze most LLM layers
        for name, param in model.lang_model.model.named_parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers (default: 6)
        num_unfreeze_layers = getattr(args, 'num_unfreeze_layers', 6)
        total_layers = len(model.lang_model.model.layers)
        for i in range(total_layers - num_unfreeze_layers, total_layers):
            for param in model.lang_model.model.layers[i].parameters():
                param.requires_grad = True
        
        # Unfreeze all other components
        if args.tune_token_emb:
            for param in model.lang_model.get_input_embeddings().parameters():
                param.requires_grad = True
        
        for param in model.out_head.parameters():
            param.requires_grad = True
        for param in model.img_embeddings.parameters():
            param.requires_grad = True
        for param in model.gmap_pos_embeddings.parameters():
            param.requires_grad = True
        for param in model.vp_pos_embeddings.parameters():
            param.requires_grad = True
        for param in model.token_type_embeddings.parameters():
            param.requires_grad = True
        
    elif stage == 'stage3':
        # Stage 3: Unfreeze all layers (full fine-tuning)
        logger.info("Stage 3: Full fine-tuning (all layers trainable)")
        # All parameters are already unfrozen
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*trainable_params/total_params:.1f}%)")


def create_optimizer(model, args, stage='stage1'):
    """
    Create optimizer with different learning rates for different components
    """
    # Group parameters by component
    param_groups = []
    
    # Classification head: higher LR
    head_params = list(model.out_head.parameters())
    if head_params:
        head_lr = args.lr_head if (hasattr(args, 'lr_head') and args.lr_head is not None) else args.lr * 10
        param_groups.append({
            'params': head_params,
            'lr': head_lr,  # 10x for head
            'name': 'head'
        })
    
    # Visual embeddings: medium LR
    visual_params = list(model.img_embeddings.parameters()) + \
                   list(model.gmap_pos_embeddings.parameters()) + \
                   list(model.vp_pos_embeddings.parameters()) + \
                   list(model.token_type_embeddings.parameters())
    if visual_params:
        visual_lr = args.lr_visual if (hasattr(args, 'lr_visual') and args.lr_visual is not None) else args.lr
        param_groups.append({
            'params': visual_params,
            'lr': visual_lr,
            'name': 'visual'
        })
    
    # LLM layers: lower LR
    llm_params = [p for n, p in model.lang_model.model.named_parameters() if p.requires_grad]
    if llm_params:
        llm_lr = args.lr_llm if (hasattr(args, 'lr_llm') and args.lr_llm is not None) else args.lr * 0.1  # 0.1x for LLM
        param_groups.append({
            'params': llm_params,
            'lr': llm_lr,
            'name': 'llm'
        })
    
    # Token embeddings
    if args.tune_token_emb:
        token_params = list(model.lang_model.get_input_embeddings().parameters())
        if token_params:
            token_lr = args.lr_token if (hasattr(args, 'lr_token') and args.lr_token is not None) else args.lr
            param_groups.append({
                'params': token_params,
                'lr': token_lr,
                'name': 'token_emb'
            })
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01)
    
    logger.info(f"Optimizer created with {len(param_groups)} parameter groups:")
    for group in param_groups:
        logger.info(f"  {group['name']}: {len(group['params'])} params, LR={group['lr']}")
    
    return optimizer


# Removed prepare_nav_batch - using forward_distill instead


def train_one_epoch(
    args,
    model,
    optimizer,
    lr_scheduler,
    dataloader,
    epoch,
    stage='stage1',
):
    """Train for one epoch"""
    model.train()
    loss_metric = Metrics()
    acc_metric = Metrics()
    
    pbar = tqdm(dataloader, disable=args.rank != 0, desc=f"Epoch {epoch} [{stage}]")
    
    criterion = nn.CrossEntropyLoss()
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass using simplified distill method
        try:
            # Skip batch if it was marked as invalid (from label check)
            if batch is None:
                continue
            # Get model (handle DDP wrapper)
            model_to_use = model.module if hasattr(model, 'module') else model
            
            # Use forward_distill for simplified training
            # Note: batch tensors are already on device, no need to .to(device) again
            nav_outs = model_to_use.forward_distill(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                cand_feats=batch['cand_feats'],
                cand_masks=batch['cand_masks'],
            )
            
            logits = nav_outs['fuse_logits']  # [B, max_num_cands+1] (first is stop)
            
            # Get labels
            # In DistillDataset, label is teacher_action_idx (0=stop, 1+=candidates)
            # In forward_distill output, position 0 is stop, positions 1+ are candidates
            # So labels are already correct (0 for stop, 1+ for candidates)
            labels = batch['labels'].to(device)  # [B] (0=stop, 1+=candidates)
            
            # Check labels are within valid range (don't silently clamp)
            max_label = logits.shape[1] - 1
            invalid_mask = labels > max_label
            if invalid_mask.any():
                # Log invalid labels for debugging
                invalid_labels = labels[invalid_mask]
                invalid_indices = invalid_mask.nonzero(as_tuple=True)[0]
                logger.warning(f"Found {invalid_mask.sum().item()} invalid labels: {invalid_labels.tolist()} "
                             f"at batch indices {invalid_indices.tolist()}, max_label={max_label}")
                # Skip this batch to avoid training on corrupted data
                continue
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Calculate accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            
            # Backward pass
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            
            loss.backward()
            
            if (step + 1) % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                # Clear cache after optimizer step to free memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            # Update metrics
            loss_metric.accumulate(loss.item() * args.gradient_accumulation_step)
            acc_metric.accumulate(acc)
            
            # Update progress bar
            if args.rank == 0:
                pbar.set_postfix({
                    'loss': loss_metric.average,
                    'acc': acc_metric.average,
                    'lr': lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                })
                
                # Log to wandb
                if step % 100 == 0:
                    wandb.log({
                        'epoch': epoch,
                        'step': step + epoch * len(dataloader),
                        'loss': loss_metric.average,
                        'acc': acc_metric.average,
                        'lr': lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    })
        
        except Exception as e:
            logger.error(f"Error in forward/backward: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return loss_metric.average, acc_metric.average


def main():
    # 使用独立的参数解析函数，避免修改原有代码
    args, global_cfg = read_distill_args()
    
    random_seed(args.seed)
    
    # Setup distributed training
    if args.distributed:
        dist.init_process_group(backend='nccl')
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = Path(args.output_dir) / 'log.txt'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    if args.rank == 0:
        logger.info("=" * 50)
        logger.info("Starting Knowledge Distillation Training")
        logger.info("=" * 50)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Config file: {args.cfg_file}")
        logger.info(f"Student model: {args.pretrained_model_name_or_path}")
        logger.info(f"Distill stage: {args.distill_stage}")
    
    # 路径和配置已经在 read_distill_args() 中解析好了
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    # Create feature database
    feat_db = create_feature_db(
        global_cfg.Feature.feature_database,
        global_cfg.Feature.image_feat_size,
        args
    )
    
    # Initialize model
    logger.info("Initializing NavQwen3 model...")
    model = NavQwen3(args, logger, global_cfg)
    
    # Load checkpoint if specified (for continuing from Stage1 to Stage2)
    resume_from_epoch = 0
    checkpoint = None
    checkpoint_path = None
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.is_absolute():
            # Try relative to project root
            checkpoint_path = project_root / checkpoint_path
            if not checkpoint_path.exists():
                # Try as-is
                checkpoint_path = Path(args.resume_from_checkpoint)
        
        if checkpoint_path.exists():
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
            
            # Load model state dict
            model_state_dict = model.state_dict()
            if 'model_state_dict' in checkpoint:
                state_disk = checkpoint['model_state_dict']
            else:
                state_disk = checkpoint
            
            # Handle DDP prefix (module.)
            state_disk = {k.replace('module.', ''): v for k, v in state_disk.items()}
            
            # Only load matching parameters
            update_model_state = {}
            for key, val in state_disk.items():
                if key in model_state_dict and model_state_dict[key].shape == val.shape:
                    update_model_state[key] = val
                else:
                    if args.rank == 0:
                        logger.warning(f"Ignore weight {key}: shape mismatch or not found in model")
            
            msg = model.load_state_dict(update_model_state, strict=False)
            if args.rank == 0:
                logger.info(f"Loaded checkpoint: {msg}")
            
            # Get epoch if available
            if 'epoch' in checkpoint:
                resume_from_epoch = checkpoint['epoch'] + 1
                if args.rank == 0:
                    logger.info(f"Resuming from epoch {resume_from_epoch}")
        else:
            if args.rank == 0:
                logger.warning(f"Checkpoint file not found: {checkpoint_path}, starting from scratch")
    
    # Move to device
    device = torch.device(f'cuda:{args.local_rank}')
    model = model.to(device)
    
    # Setup distributed model
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # Get tokenizer from model
    if hasattr(model, 'module'):
        tokenizer = model.module.lang_model.tokenizer
    else:
        tokenizer = model.lang_model.tokenizer
    
    # Create dataset
    logger.info("Creating DistillDataset...")
    jsonl_paths = getattr(args, 'distill_jsonl_paths', [])
    if not jsonl_paths:
        # Default paths
        distill_log_dir = getattr(args, 'distill_output_dir', os.path.join(args.output_dir, 'distill_logs'))
        jsonl_paths = [
            os.path.join(distill_log_dir, f'distill_{task.lower()}_train.jsonl')
            for task in getattr(args, 'distill_tasks', ['CVDN', 'R2R', 'REVERIE', 'SOON'])
        ]
        jsonl_paths = [p for p in jsonl_paths if os.path.exists(p)]
    
    if not jsonl_paths:
        raise ValueError(f"No JSONL files found. Please specify --distill_jsonl_paths or ensure files exist in {distill_log_dir}")
    
    # Resolve JSONL paths relative to project root
    resolved_jsonl_paths = []
    for path in jsonl_paths:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            # Try relative to project root
            resolved_path = project_root / path
            if resolved_path.exists():
                resolved_jsonl_paths.append(str(resolved_path))
            else:
                # Try as-is (relative to current working directory)
                resolved_jsonl_paths.append(path)
        else:
            resolved_jsonl_paths.append(path)
    
    logger.info(f"Using JSONL files: {resolved_jsonl_paths}")
    
    # Load original datasets for scan lookup (optional)
    original_datasets = {}
    try:
        from tasks.loaders import create_dataloaders as create_original_dataloaders
        # This is optional - only needed if scan is not in JSONL
        pass
    except:
        pass
    
    dataset = DistillDataset(
        jsonl_paths=resolved_jsonl_paths,
        args=args,
        config=global_cfg.Feature,
        tokenizer=tokenizer,
        feat_db=feat_db,
        original_datasets=original_datasets,
        task_filter=getattr(args, 'distill_tasks', None),
        filter_success_only=getattr(args, 'filter_success_only', False),
        filter_min_spl=getattr(args, 'filter_min_spl', 0.0),
    )
    
    # Create dataloader
    # For distributed training, use DistributedSampler
    if args.distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
        collate_fn=DistillDataset.collate_fn,
        pin_memory=True,
    )
    
    # Determine training stage
    training_stage = getattr(args, 'distill_stage', 'stage1')
    
    # Setup freezing strategy
    if hasattr(model, 'module'):
        setup_freeze_strategy(model.module, args, training_stage)
    else:
        setup_freeze_strategy(model, args, training_stage)
    
    # Create optimizer
    if hasattr(model, 'module'):
        optimizer = create_optimizer(model.module, args, training_stage)
    else:
        optimizer = create_optimizer(model, args, training_stage)
    
    # Load optimizer state if resuming from checkpoint
    if checkpoint is not None and checkpoint_path is not None and checkpoint_path.exists():
        if 'optimizer' in checkpoint:
            optimizer_state = checkpoint['optimizer']
            # Move optimizer state to correct device
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            try:
                optimizer.load_state_dict(optimizer_state)
                if args.rank == 0:
                    logger.info("Loaded optimizer state from checkpoint")
            except Exception as e:
                if args.rank == 0:
                    logger.warning(f"Failed to load optimizer state: {e}, using fresh optimizer")
    
    # Create learning rate scheduler
    num_training_steps = len(dataloader) * args.num_epochs
    num_warmup_steps = getattr(args, 'num_warmup_steps', int(0.1 * num_training_steps))
    
    lr_scheduler_type = getattr(args, 'lr_scheduler', 'constant')
    if lr_scheduler_type == 'cosine':
        lr_scheduler = get_cosine_scheduler(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    
    # Restore scheduler state if resuming
    if checkpoint is not None and checkpoint_path is not None and checkpoint_path.exists():
        if 'lr_scheduler' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if args.rank == 0:
                    logger.info("Loaded learning rate scheduler state from checkpoint")
            except Exception as e:
                if args.rank == 0:
                    logger.warning(f"Failed to load lr_scheduler state: {e}, using fresh scheduler")
    
    # Initialize wandb
    if args.rank == 0 and not getattr(args, 'no_wandb', False):
        wandb.init(
            project=getattr(args, 'wandb_project', 'navillm_distill'),
            name=getattr(args, 'wandb_name', f'navqwen3_{training_stage}'),
            config=vars(args)
        )
    
    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    if args.distributed:
        logger.info(f"Distributed training: world_size={args.world_size}, rank={args.rank}, local_rank={args.local_rank}")
    
    # Resume from epoch if loading checkpoint
    start_epoch = resume_from_epoch if args.resume_from_checkpoint is not None else 0
    if start_epoch > 0 and args.rank == 0:
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    best_acc = 0.0
    
    # Training history for plotting (saved to JSON)
    training_history = {
        'epochs': [],
        'loss': [],
        'acc': [],
        'lr': []
    }
    history_file = Path(args.output_dir) / 'training_history.json'
    
    for epoch in range(start_epoch, args.num_epochs):
        # Set epoch for DistributedSampler (ensures different shuffling each epoch)
        if args.distributed and hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        if args.rank == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
            logger.info(f"{'='*50}")
        
        # Train
        avg_loss, avg_acc = train_one_epoch(
            args, model, optimizer, lr_scheduler, dataloader, epoch, training_stage
        )
        
        if args.rank == 0:
            logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
            
            # Get current learning rate
            current_lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            
            # Save training history
            training_history['epochs'].append(epoch + 1)
            training_history['loss'].append(float(avg_loss))
            training_history['acc'].append(float(avg_acc))
            training_history['lr'].append(float(current_lr))
            
            # Save history to JSON file (for plotting)
            with open(history_file, 'w') as f:
                json.dump(training_history, f, indent=2)
            
            # Log to wandb (epoch-level metrics)
            if not getattr(args, 'no_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'epoch_loss': avg_loss,
                    'epoch_acc': avg_acc,
                    'epoch_lr': current_lr,
                }, step=(epoch + 1) * len(dataloader))
            
            # Save checkpoint
            checkpoint_dir = Path(args.output_dir) / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f'navqwen3_{training_stage}_epoch_{epoch+1}.pt'
            save_checkpoint(model, str(checkpoint_path), optimizer, epoch, save_states=True)
            
            # Save best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_path = checkpoint_dir / f'navqwen3_{training_stage}_best.pt'
                save_checkpoint(model, str(best_path), optimizer, epoch, save_states=False)
                logger.info(f"Saved best model (acc: {best_acc:.4f}) to {best_path}")
    
    if args.rank == 0:
        logger.info("Training completed!")
        if not getattr(args, 'no_wandb', False):
            wandb.finish()


if __name__ == '__main__':
    main()

