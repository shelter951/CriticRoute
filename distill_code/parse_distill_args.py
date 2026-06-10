"""
独立的参数解析函数，用于蒸馏训练
避免修改原有的 read_args() 函数，保持代码解耦
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import os
import yaml
from easydict import EasyDict
import torch


def read_distill_args():
    """
    为蒸馏训练创建独立的参数解析器
    返回: args, global_cfg
    """
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    
    # 基础参数
    parser.add_argument('--data_dir', type=str, default='data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset configs')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, 
                        help="path to student model (Qwen3-1.7B)")
    
    # 训练参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="path to ckpt to resume from")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--feat_dropout", type=float, default=0.4)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--precision", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
                        default="fp32", help="Floating point precision.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # 分布式训练参数
    parser.add_argument('--gpu', type=int, default=0, help='current gpu id, local rank')
    parser.add_argument('--world_size', type=int, default=0, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--dist-url", default="env://", type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--no-set-device-rank", action="store_true",
                        help="Don't set device index from local rank")
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, required=True, help="output logs and ckpts")
    parser.add_argument("--max_saved_checkpoints", type=int, default=0)
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    
    # 模型参数
    parser.add_argument("--freeze_llama", action="store_true", help="whether freezing LLM")
    parser.add_argument("--tune_token_emb", action="store_true", help="whether tuning token embedding")
    parser.add_argument("--no_loc_fts", action="store_true", help="no loc fts during nav")
    parser.add_argument("--enable_og", action="store_true", help="enable object grounding")
    parser.add_argument("--fuse_obj", action="store_true", help="fuse object features")
    
    # 蒸馏训练专用参数
    parser.add_argument("--distill_jsonl_paths", type=str, nargs='+', default=None,
                        help="Paths to JSONL files for distillation")
    parser.add_argument("--distill_tasks", type=str, nargs='+', default=None,
                        help="Tasks to include in distillation (CVDN, R2R, etc.)")
    parser.add_argument("--distill_stage", type=str, default="stage1",
                        choices=["stage1", "stage2", "stage3"], help="Distillation training stage")
    parser.add_argument("--num_unfreeze_layers", type=int, default=6,
                        help="Number of LLM layers to unfreeze in stage2")
    parser.add_argument("--lr_head", type=float, default=None,
                        help="Learning rate for classification head (default: lr * 10)")
    parser.add_argument("--lr_visual", type=float, default=None,
                        help="Learning rate for visual layers (default: lr)")
    parser.add_argument("--lr_llm", type=float, default=None,
                        help="Learning rate for LLM layers (default: lr * 0.1)")
    parser.add_argument("--lr_token", type=float, default=None,
                        help="Learning rate for token embeddings (default: lr)")
    parser.add_argument("--filter_success_only", action="store_true",
                        help="Only use successful episodes for distillation")
    parser.add_argument("--filter_min_spl", type=float, default=0.0,
                        help="Minimum SPL threshold for filtering episodes")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="navillm_distill", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    
    # 兼容性参数（为了兼容原有代码，但不使用）
    parser.add_argument("--stage", type=str, default="multi", choices=["pretrain", "multi"],
                        help="Not used in distill training, kept for compatibility")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Not used in distill training, kept for compatibility")
    
    args = parser.parse_args()
    
    # 设置分布式相关属性
    args.rank = 0
    args.world_size = 1
    args.local_rank = args.gpu if args.local_rank == -1 else args.local_rank
    args.distributed = False  # 默认单GPU，可以通过环境变量启用分布式
    
    # 检查是否启用分布式训练
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    
    # 解析配置文件路径
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.absolute()
    
    cfg_path = Path(args.cfg_file)
    if not cfg_path.is_absolute():
        if args.cfg_file.startswith('../'):
            rel_path = args.cfg_file[3:]
            cfg_path = project_root / rel_path
        else:
            cfg_path = project_root / args.cfg_file
    else:
        cfg_path = Path(args.cfg_file)
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path} (resolved from {args.cfg_file})")
    
    # 加载配置
    with open(cfg_path, 'r') as f:
        global_cfg = EasyDict(yaml.safe_load(f))
    
    # 解析其他路径
    if not Path(args.data_dir).is_absolute():
        if args.data_dir.startswith('../'):
            args.data_dir = str(project_root / args.data_dir[3:])
        else:
            args.data_dir = str(project_root / args.data_dir)
    
    if not Path(args.pretrained_model_name_or_path).is_absolute():
        if args.pretrained_model_name_or_path.startswith('../'):
            args.pretrained_model_name_or_path = str(project_root / args.pretrained_model_name_or_path[3:])
        else:
            args.pretrained_model_name_or_path = str(project_root / args.pretrained_model_name_or_path)
    
    if not Path(args.output_dir).is_absolute():
        if args.output_dir.startswith('../'):
            args.output_dir = str(project_root / args.output_dir[3:])
        else:
            args.output_dir = str(project_root / args.output_dir)
    
    # 设置配置相关属性
    args.image_feat_size = global_cfg.Feature.image_feat_size
    args.obj_feat_size = global_cfg.Feature.obj_feat_size
    args.angle_feat_size = global_cfg.Feature.angle_feat_size
    args.enc_full_graph = getattr(global_cfg.Model, 'enc_full_graph', False)
    args.expert_policy = getattr(global_cfg.Model, 'expert_policy', False)
    args.num_pano_layers = getattr(global_cfg.Model, 'num_pano_layers', 0)
    
    return args, global_cfg

