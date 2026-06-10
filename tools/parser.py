import argparse
import random
import numpy as np
import torch
import os
import datetime
import yaml
from easydict import EasyDict
from .distributed import world_info_from_env, init_distributed_device
from .common_utils import create_logger, log_config_to_file
from pathlib import Path


def random_seed(seed=0, rank=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, default=None, help='dataset configs', required=True)
    parser.add_argument('--pretrained_model_name_or_path', default=None, type=str, required=True, help="path to tokenizer")

    # local fusion
    parser.add_argument('--off_batch_task', action='store_true', default=False, help="whether all process is training same task")
    parser.add_argument('--debug', action="store_true", help="debug mode")
    parser.add_argument('--few_shot', type=int, default=None, help='sample number for few shot')
    parser.add_argument('--tour3d_nav_head', action="store_true", help="whether use seperate nav head for tour3d")
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to ckpt to resume from")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--feat_dropout", type=float, default=0.4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps_per_epoch", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    parser.add_argument("--workers", type=int, default=0)

    # distributed training args
    parser.add_argument('--gpu', type=int, default=0, help='current gpu id, local rank')
    parser.add_argument('--world_size', type=int, default=0, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # Save checkpoints
    parser.add_argument('--output_dir', type=str, default=None, required=True, help="output logs and ckpts")
    parser.add_argument("--max_saved_checkpoints", type=int, default=0)
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    parser.add_argument("--save_latest_states", action='store_true')
    parser.add_argument("--save_pred_results", action="store_true")
    parser.add_argument("--save_detail_results", action="store_true")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument("--stage", type=str, default=None, choices=["pretrain", "multi"], 
                        help="Training stage. Required for train.py, optional for distill training.")
    parser.add_argument('--ignoreid', default=-100, type=int, help="criterion: ignore label")
    parser.add_argument('--enable_og', action='store_true', default=False, help="object grounding task")
    parser.add_argument("--enable_summarize", action="store_true", help="perform EQA or generate instructions")
    parser.add_argument("--enable_fgr2r", action="store_true", help="perform fgr2r for R2R")
    parser.add_argument("--disable_nav", action="store_true", help="disable nav loss")
    parser.add_argument("--gen_loss_coef", type=float, default=1.)
    parser.add_argument("--obj_loss_coef", type=float, default=1.)
    parser.add_argument("--teacher_forcing_coef", type=float, default=1.)
    parser.add_argument("--fuse_obj", action="store_true", help="whether fuse object features for REVERIE and SOON")
    parser.add_argument("--use_lora", action="store_true", help="whether using lora")
    parser.add_argument("--lora_rank", type=int, default=8, help="lora rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha, usually starting from two times of rank")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora dropout")
    parser.add_argument('--lora_target', type=str, default=None, nargs='+')
    parser.add_argument("--freeze_llama", action="store_true", help="whether freezing llama")
    parser.add_argument("--tune_token_emb", action="store_true", help="whether tuning token embedding")


    # room tour
    parser.add_argument("--no_loc_fts", action="store_true", help="no loc fts during nav")

    # datasets
    parser.add_argument("--multi_endpoints", type=int, default=1)
    parser.add_argument("--path_type", type=str, default="trusted_path", choices=["planner_path", "trusted_path"])

    # evaluation
    parser.add_argument('--test_datasets', type=str, default=None, nargs='+')
    parser.add_argument('--validation_split', type=str, default="val_unseen", help="validation split: val_seen, val_unseen, test")
    parser.add_argument("--do_sample", action="store_true", help="do_sample in evaluation")
    parser.add_argument("--temperature", type=float, default=1.)
    
    # distillation logging
    parser.add_argument("--enable_distill_log", action="store_true", help="Enable distillation data logging for teacher model")
    parser.add_argument("--distill_output_dir", type=str, default=None, help="Output directory for distillation logs (default: output_dir/distill_logs)")
    
    # distillation training
    parser.add_argument("--distill_jsonl_paths", type=str, nargs='+', default=None, help="Paths to JSONL files for distillation")
    parser.add_argument("--distill_tasks", type=str, nargs='+', default=None, help="Tasks to include in distillation (CVDN, R2R, etc.)")
    parser.add_argument("--distill_stage", type=str, default="stage1", choices=["stage1", "stage2", "stage3"], help="Distillation training stage")
    parser.add_argument("--num_unfreeze_layers", type=int, default=6, help="Number of LLM layers to unfreeze in stage2")
    parser.add_argument("--lr_head", type=float, default=None, help="Learning rate for classification head (default: lr * 10)")
    parser.add_argument("--lr_visual", type=float, default=None, help="Learning rate for visual layers (default: lr)")
    parser.add_argument("--lr_llm", type=float, default=None, help="Learning rate for LLM layers (default: lr * 0.1)")
    parser.add_argument("--lr_token", type=float, default=None, help="Learning rate for token embeddings (default: lr)")
    parser.add_argument("--filter_success_only", action="store_true", help="Only use successful episodes for distillation")
    parser.add_argument("--filter_min_spl", type=float, default=0.0, help="Minimum SPL threshold for filtering episodes")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["constant", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="navillm_distill", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")


    # others
    parser.add_argument(
        "--max_datapoints",
        default=None,
        type=int,
        help="The number of datapoints used for debug."
    )

    args = parser.parse_args()

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    ###################### configurations #########################
    # single-gpu or multi-gpu
    device_id = init_distributed_device(args)
    
    # Resolve config file path
    # Handle relative paths, especially those starting with ../
    cfg_path = Path(args.cfg_file)
    if not cfg_path.is_absolute():
        # If path starts with ../, we need to resolve it properly
        if args.cfg_file.startswith('../'):
            # Find project root by looking for configs/ directory
            current_dir = Path.cwd()
            project_root = None
            # Check current directory and parent directories
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "configs").exists():
                    project_root = parent
                    break
            
            if project_root:
                # Remove ../ prefix and construct path from project root
                rel_path = args.cfg_file[3:]  # Remove '../'
                cfg_path = project_root / rel_path
            else:
                # Fallback: resolve relative to current directory
                cfg_path = cfg_path.resolve()
        else:
            # Regular relative path - resolve from current directory
            cfg_path = cfg_path.resolve()
            
            # If doesn't exist, try to find project root
            if not cfg_path.exists():
                current_dir = Path.cwd()
                for parent in [current_dir] + list(current_dir.parents):
                    test_path = parent / args.cfg_file
                    if test_path.exists():
                        cfg_path = test_path
                        break
    else:
        cfg_path = cfg_path.resolve()
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path} (resolved from {args.cfg_file}, cwd={Path.cwd()})")
    
    global_cfg = EasyDict(yaml.safe_load(open(str(cfg_path))))

    # Resolve data_dir - similar logic
    data_dir_path = Path(args.data_dir)
    if not data_dir_path.is_absolute():
        if args.data_dir.startswith('../'):
            # Use same project_root found above
            if 'project_root' in locals() and project_root:
                rel_path = args.data_dir[3:]
                data_dir_path = project_root / rel_path
            else:
                # Find project root
                current_dir = Path.cwd()
                for parent in [current_dir] + list(current_dir.parents):
                    if (parent / "data").exists():
                        rel_path = args.data_dir[3:]
                        data_dir_path = parent / rel_path
                        break
                else:
                    data_dir_path = data_dir_path.resolve()
        else:
            data_dir_path = data_dir_path.resolve()
    else:
        data_dir_path = data_dir_path.resolve()
    args.data_dir = data_dir_path

    # off-line image features from Matterport3D
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size

    ############# Configurations ###############
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = global_cfg.Model.enc_full_graph
    args.expert_policy = global_cfg.Model.expert_policy
    args.num_pano_layers = global_cfg.Model.num_pano_layers

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = Path(args.output_dir) / 'log.txt'

    logger = create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(global_cfg, logger=logger)

    print(" + rank: {}, + device_id: {}".format(args.local_rank, device_id))
    print(f"Start running training on rank {args.rank}.")

    if os.path.exists(os.path.join(args.output_dir, "latest_states.pt")):
        state_path = os.path.join(args.output_dir, "latest_states.pt")
        logger.info("Resume checkponit from {}".format(state_path))
        args.resume_from_checkpoint = state_path

    return args, global_cfg, logger, device_id
