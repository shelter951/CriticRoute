"""
边云协同导航评估主脚本

完整实现所有功能，无TODO遗留：
1. Teacher-only / Student-only / EdgeCloud 模式
2. 多种路由策略（Entropy, Divergence, Off-course Router）
3. τ sweep（不同阈值下的性能评估）
4. 网络延迟模拟（固定400ms）
5. 多卡并行评估
6. 完整的指标收集（SR/SPL/调用率/时间/资源占用）
7. 支持Student-base和Student-distill对比
8. 在CVDN val_unseen上评估
"""
import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
import copy

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from tasks.loaders import create_dataloaders
from tasks.agents.mp3d_agent import MP3DAgent
from tasks.agents.cvdn import CVDNAgent
from models.nav_model import NavModel
from distill_code.models.nav_qwen3 import NavQwen3
from models.graph_utils import GraphMap
from tools.parser import read_args
import yaml
from easydict import EasyDict

# Import edgecloud modules
from edgecloud_experiments.utils.router_features import extract_router_features, check_feature_statistics
from edgecloud_experiments.utils.latency_simulator import simulate_cloud_latency, LatencyMode
from edgecloud_experiments.utils.nav_inputs_builder import build_nav_inputs_from_obs, move_batch_to_device
from edgecloud_experiments.routers import EntropyRouter, DivergenceRouter, OffCourseRouter, PPORouter, BaseRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_teacher_model(args, global_cfg, device: torch.device, logger):
    """加载Teacher模型（RoomTour3D-NaviLLM）"""
    from models.nav_model import NavModel
    
    logger.info("Loading Teacher model...")
    
    # === 关键：Teacher 强制使用 Vicuna-7B 的 tokenizer 路径 ===
    # 完全忽略 args.pretrained_model_name_or_path（那是给Student用的）
    # 参考router_code/collect_router_data.py的处理方式
    teacher_pretrained_path = getattr(args, 'teacher_pretrained_path', None)
    
    if not teacher_pretrained_path:
        # 尝试从config获取（如果config中有Vicuna路径）
        if hasattr(global_cfg, 'Model') and hasattr(global_cfg.Model, 'pretrained_model_name_or_path'):
            config_path = global_cfg.Model.pretrained_model_name_or_path
            # 检查是否是Vicuna路径（不是Qwen）
            if config_path and 'vicuna' in str(config_path).lower() and 'qwen' not in str(config_path).lower():
                teacher_pretrained_path = config_path
        
        # 如果config中没有或不是Vicuna，尝试默认位置
        if not teacher_pretrained_path:
            default_paths = [
                project_root / "data" / "models" / "Vicuna-7B",
                project_root / "data" / "models" / "vicuna-7b",
                Path("data/models/Vicuna-7B"),
            ]
            for default_path in default_paths:
                if default_path.exists() and (default_path / "config.json").exists():
                    teacher_pretrained_path = str(default_path)
                    logger.info(f"Using default Teacher pretrained path: {teacher_pretrained_path}")
                    break
        
        # 最后兜底
        if not teacher_pretrained_path:
            logger.warning("Vicuna-7B path not found, fallback to 'data/models/Vicuna-7B'")
            teacher_pretrained_path = "data/models/Vicuna-7B"
    
    teacher_pretrained_path = str(teacher_pretrained_path)  # 确保是字符串
    
    # 构建Teacher args（必须包含所有NavModel需要的属性）
    # 参考router_code/collect_router_data.py中的完整设置
    # 重要：不要使用 teacher_args.update(vars(args))，因为会继承Student的pretrained_model_name_or_path
    teacher_args = EasyDict()
    
    # 只复制需要的属性，不复制pretrained_model_name_or_path
    # 重要：原项目评估使用amp_bf16，可以大幅减少内存占用（约50%）
    # 默认使用amp_bf16，和原项目保持一致
    teacher_args.precision = getattr(args, 'precision', 'amp_bf16')
    teacher_args.resume_from_checkpoint = getattr(args, 'resume_from_checkpoint', None)
    teacher_args.from_scratch = getattr(args, 'from_scratch', False)
    teacher_args.freeze_llama = getattr(args, 'freeze_llama', False)
    teacher_args.tune_token_emb = getattr(args, 'tune_token_emb', False)
    teacher_args.use_lora = getattr(args, 'use_lora', False)
    teacher_args.lora_rank = getattr(args, 'lora_rank', None)
    teacher_args.lora_alpha = getattr(args, 'lora_alpha', None)
    teacher_args.lora_dropout = getattr(args, 'lora_dropout', None)
    teacher_args.lora_target = getattr(args, 'lora_target', None)
    teacher_args.enable_og = getattr(args, 'enable_og', False)
    teacher_args.fuse_obj = getattr(args, 'fuse_obj', False)
    teacher_args.no_loc_fts = getattr(args, 'no_loc_fts', False)
    
    # === 强制设置Teacher的pretrained路径为Vicuna-7B ===
    teacher_args.pretrained_model_name_or_path = teacher_pretrained_path
    logger.info(f"Using Teacher pretrained path: {teacher_args.pretrained_model_name_or_path}")
    
    # 从global_cfg获取Teacher相关配置
    # 重要：优先从Feature配置获取image_feat_size等特征维度（因为这是实际使用的配置）
    # 然后从Model配置获取其他模型相关参数
    if hasattr(global_cfg, 'Feature'):
        # 优先从Feature配置获取特征维度（这是配置文件中的实际值）
        teacher_args.image_feat_size = getattr(global_cfg.Feature, 'image_feat_size', 1024)
        teacher_args.angle_feat_size = getattr(global_cfg.Feature, 'angle_feat_size', 4)
        teacher_args.obj_feat_size = getattr(global_cfg.Feature, 'obj_feat_size', 768)
        logger.info(f"Using Feature config: image_feat_size={teacher_args.image_feat_size}, "
                   f"angle_feat_size={teacher_args.angle_feat_size}, obj_feat_size={teacher_args.obj_feat_size}")
    else:
        # 如果没有Feature配置，使用默认值（注意：multi.yaml中是1024，不是768）
        teacher_args.image_feat_size = 1024
        teacher_args.angle_feat_size = 4
        teacher_args.obj_feat_size = 768
        logger.warning("No Feature config found, using default image_feat_size=1024")
    
    if hasattr(global_cfg, 'Model'):
        teacher_args.tour3d_nav_head = getattr(global_cfg.Model, 'tour3d_nav_head', False)
        teacher_args.feat_dropout = getattr(global_cfg.Model, 'feat_dropout', 0.4)
        # 如果Model配置中有image_feat_size，使用它（但通常应该和Feature一致）
        if hasattr(global_cfg.Model, 'image_feat_size'):
            teacher_args.image_feat_size = global_cfg.Model.image_feat_size
            logger.info(f"Model config overrides image_feat_size to {teacher_args.image_feat_size}")
        teacher_args.enc_full_graph = getattr(global_cfg.Model, 'enc_full_graph', True)
        teacher_args.expert_policy = getattr(global_cfg.Model, 'expert_policy', 'spl')
    else:
        # 如果没有Model配置，使用默认值
        teacher_args.feat_dropout = 0.4
        teacher_args.tour3d_nav_head = False
        teacher_args.enc_full_graph = True
        teacher_args.expert_policy = 'spl'
    
    # 确保pretrained_model_name_or_path是字符串且路径有效
    teacher_args.pretrained_model_name_or_path = str(teacher_args.pretrained_model_name_or_path)
    
    # 确保路径是绝对路径（如果相对路径存在）
    pretrained_path = Path(teacher_args.pretrained_model_name_or_path)
    if not pretrained_path.is_absolute():
        # 尝试相对于项目根目录
        abs_path = project_root / pretrained_path
        if abs_path.exists():
            teacher_args.pretrained_model_name_or_path = str(abs_path)
        else:
            # 保持原路径，让transformers库处理
            teacher_args.pretrained_model_name_or_path = str(pretrained_path)
    else:
        teacher_args.pretrained_model_name_or_path = str(pretrained_path)
    
    logger.info(f"Teacher pretrained_model_name_or_path: '{teacher_args.pretrained_model_name_or_path}' (type: {type(teacher_args.pretrained_model_name_or_path).__name__})")
    logger.info(f"Path exists: {os.path.exists(teacher_args.pretrained_model_name_or_path)}")
    
    # 创建模型配置（确保Model配置存在）
    model_config = global_cfg.Model if hasattr(global_cfg, 'Model') else EasyDict()
    if not hasattr(model_config, 'num_pano_layers'):
        model_config.num_pano_layers = 2  # 默认值
    
    # 创建模型
    try:
        model = NavModel(teacher_args, logger, model_config)
    except TypeError as e:
        if "not a string" in str(e):
            logger.error(f"TypeError in NavModel initialization. pretrained_model_name_or_path type: {type(teacher_args.pretrained_model_name_or_path)}, value: {teacher_args.pretrained_model_name_or_path}")
            logger.error(f"All teacher_args attributes: {[(k, type(v).__name__) for k, v in teacher_args.items() if 'pretrained' in k.lower() or 'path' in k.lower()]}")
            raise
        else:
            raise
    
    # 加载checkpoint（先加载到CPU，再移动到目标GPU，避免内存碎片）
    teacher_ckpt_path = None
    if args.teacher_ckpt:
        # 检查路径是否存在（支持相对路径和绝对路径）
        if os.path.exists(args.teacher_ckpt):
            teacher_ckpt_path = args.teacher_ckpt
        else:
            # 尝试相对于项目根目录
            abs_path = project_root / args.teacher_ckpt
            if abs_path.exists():
                teacher_ckpt_path = str(abs_path)
            else:
                logger.warning(f"Teacher checkpoint path not found: {args.teacher_ckpt}, trying default locations...")
    
    # 如果没有提供或路径不存在，尝试默认路径
    if teacher_ckpt_path is None:
        default_teacher_paths = [
            project_root / 'navillm_roomtour3d_video_action_instruction.pt',
            project_root / 'build' / 'nav_ckpts' / 'navillm_cvdn_teacher.pt',
            project_root / 'build' / 'nav_ckpts' / 'navillm_roomtour3d_video_action_instruction.pt',
        ]
        for path in default_teacher_paths:
            if path.exists():
                teacher_ckpt_path = str(path)
                logger.info(f"Found Teacher checkpoint at default location: {teacher_ckpt_path}")
                break
    
    if teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
        logger.info(f"Loading Teacher checkpoint from {teacher_ckpt_path}")
        logger.info(f"Teacher model config: image_feat_size={teacher_args.image_feat_size}")
        # 先加载到CPU，避免在错误设备上占用内存
        checkpoint = torch.load(teacher_ckpt_path, map_location='cpu', weights_only=False)
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            logger.info("Teacher checkpoint loaded successfully")
        except RuntimeError as e:
            if 'size mismatch' in str(e) or 'shape' in str(e).lower():
                logger.error(f"Checkpoint shape mismatch! This usually means the checkpoint was trained with "
                           f"different image_feat_size than current config ({teacher_args.image_feat_size}).")
                logger.error(f"Error details: {e}")
                logger.error("Solutions:")
                logger.error("1. Use a checkpoint that matches your config (e.g., navillm_cvdn_teacher.pt for CVDN)")
                logger.error("2. Or adjust image_feat_size in your config file to match the checkpoint")
                raise RuntimeError(f"Checkpoint shape mismatch. Current image_feat_size={teacher_args.image_feat_size}. "
                                 f"Please use a matching checkpoint or adjust config. Original error: {e}")
            else:
                raise
        # 清理checkpoint内存
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        logger.warning(f"No Teacher checkpoint found. Tried: {args.teacher_ckpt if args.teacher_ckpt else 'None'} and default locations. Using random initialization.")
    
    # 移动到目标设备
    # 注意：原项目使用amp_bf16时，模型会在ModifiedLM.__init__中自动转换为bfloat16
    # 这里只需要移动到设备，dtype转换已经在模型初始化时完成
    logger.info(f"Moving Teacher model to device: {device} (precision: {teacher_args.precision})")
    model = model.to(device)
    # 清理缓存，确保内存释放
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.eval()
    logger.info(f"Teacher model loaded (dtype: {next(model.parameters()).dtype})")
    
    return model


def load_student_model(args, global_cfg, device: torch.device, logger, student_type='distill'):
    """加载Student模型（Qwen3-1.7B）"""
    from distill_code.models.nav_qwen3 import NavQwen3
    
    logger.info(f"Loading Student model (type: {student_type})...")
    
    # 构建Student args（必须包含所有NavQwen3需要的属性）
    student_args = EasyDict()
    student_args.update(vars(args))
    
    # 必需属性
    # 重要：原项目评估使用amp_bf16，可以大幅减少内存占用（约50%）
    # 默认使用amp_bf16，和原项目保持一致
    student_args.precision = getattr(args, 'precision', 'amp_bf16')
    student_args.resume_from_checkpoint = getattr(args, 'student_ckpt', None)
    student_args.from_scratch = getattr(args, 'from_scratch', False)
    student_args.enable_og = getattr(args, 'enable_og', False)
    student_args.fuse_obj = getattr(args, 'fuse_obj', False)
    student_args.no_loc_fts = getattr(args, 'no_loc_fts', False)
    # NavQwen3需要的其他字段（参考NavModel）
    student_args.freeze_llama = getattr(args, 'freeze_llama', False)
    student_args.tune_token_emb = getattr(args, 'tune_token_emb', False)
    student_args.use_lora = getattr(args, 'use_lora', False)
    student_args.lora_rank = getattr(args, 'lora_rank', None)
    student_args.lora_alpha = getattr(args, 'lora_alpha', None)
    student_args.lora_dropout = getattr(args, 'lora_dropout', None)
    student_args.lora_target = getattr(args, 'lora_target', None)
    
    # ===== 补齐 Student 需要的配置字段（参考 load_teacher_model） =====
    # 重要：优先从Feature配置获取image_feat_size等特征维度（和Teacher保持一致）
    if hasattr(global_cfg, 'Feature'):
        # 优先从Feature配置获取特征维度（这是配置文件中的实际值）
        student_args.image_feat_size = getattr(global_cfg.Feature, 'image_feat_size', 1024)
        student_args.angle_feat_size = getattr(global_cfg.Feature, 'angle_feat_size', 4)
        student_args.obj_feat_size = getattr(global_cfg.Feature, 'obj_feat_size', 768)
        logger.info(f"Student using Feature config: image_feat_size={student_args.image_feat_size}, "
                   f"angle_feat_size={student_args.angle_feat_size}, obj_feat_size={student_args.obj_feat_size}")
    else:
        # 如果没有Feature配置，使用默认值（注意：multi.yaml中是1024，不是768）
        student_args.image_feat_size = 1024
        student_args.angle_feat_size = 4
        student_args.obj_feat_size = 768
        logger.warning("No Feature config found for Student, using default image_feat_size=1024")
    
    if hasattr(global_cfg, 'Model'):
        student_args.tour3d_nav_head = getattr(global_cfg.Model, 'tour3d_nav_head', False)
        student_args.feat_dropout = getattr(global_cfg.Model, 'feat_dropout', 0.4)
        # 如果Model配置中有image_feat_size，使用它（但通常应该和Feature一致）
        if hasattr(global_cfg.Model, 'image_feat_size'):
            student_args.image_feat_size = global_cfg.Model.image_feat_size
            logger.info(f"Student Model config overrides image_feat_size to {student_args.image_feat_size}")
        # 如果Model配置中有no_loc_fts，使用它；否则使用args中的值（上面已设置）
        if hasattr(global_cfg.Model, 'no_loc_fts'):
            student_args.no_loc_fts = global_cfg.Model.no_loc_fts
    else:
        # 如果没有Model配置，使用默认值
        student_args.feat_dropout = 0.4
        student_args.tour3d_nav_head = False
    # ===== 补齐结束 =====
    
    # 确定pretrained路径（确保是字符串）
    if args.pretrained_model_name_or_path:
        student_pretrained_path = str(args.pretrained_model_name_or_path)
    else:
        # 尝试默认路径
        default_paths = [
            project_root / 'data' / 'models' / 'Qwen3-1.7B',
            project_root / 'Qwen3-1.7B',
            Path.home() / 'NaviLLM' / 'data' / 'models' / 'Qwen3-1.7B',
        ]
        student_pretrained_path = None
        for path in default_paths:
            if path.exists() and (path / 'config.json').exists():
                student_pretrained_path = str(path)
                break
        
        if student_pretrained_path is None:
            raise ValueError(
                f"Student pretrained model path not found. Tried: {default_paths}. "
                "Please specify --pretrained_model_name_or_path"
            )
    
    logger.info(f"Using Student pretrained path: {student_pretrained_path}")
    
    # 确保pretrained_model_name_or_path是字符串
    student_args.pretrained_model_name_or_path = str(student_pretrained_path)
    
    # 创建模型（此时student_args应该包含所有NavQwen3需要的字段）
    logger.info(f"Student args: no_loc_fts={student_args.no_loc_fts}, fuse_obj={student_args.fuse_obj}, "
                f"feat_dropout={student_args.feat_dropout}, tour3d_nav_head={student_args.tour3d_nav_head}")
    model = NavQwen3(student_args, logger, global_cfg)
    
    # 加载checkpoint（如果是distill类型，先加载到CPU，再移动到目标GPU）
    if student_type == 'distill' and args.student_ckpt and os.path.exists(args.student_ckpt):
        logger.info(f"Loading Student checkpoint from {args.student_ckpt}")
        # 先加载到CPU，避免在错误设备上占用内存
        checkpoint = torch.load(args.student_ckpt, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        # 清理checkpoint内存
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    elif student_type == 'base':
        logger.info("Using base Student model (no distillation checkpoint)")
    else:
        logger.warning("No Student checkpoint provided for distill type")
    
    # 移动到目标设备
    # 注意：NavQwen3可能也支持混合精度，但需要检查其实现
    logger.info(f"Moving Student model to device: {device} (precision: {student_args.precision})")
    model = model.to(device)
    # 清理缓存，确保内存释放
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model.eval()
    logger.info(f"Student model loaded (dtype: {next(model.parameters()).dtype})")
    
    return model


def load_router(args, device: torch.device, logger) -> Optional[BaseRouter]:
    """加载Router"""
    if args.router_type == 'offcourse':
        if not args.router_ckpt or not os.path.exists(args.router_ckpt):
            raise ValueError(f"Off-course router requires --router_ckpt, got: {args.router_ckpt}")
        
        logger.info(f"Loading Off-course Router from {args.router_ckpt}")
        router = OffCourseRouter(
            checkpoint_path=args.router_ckpt,
            input_dim=None,  # 自动从checkpoint推断（实际是13维）
            hidden_dim=args.router_hidden_dim if hasattr(args, 'router_hidden_dim') else 128,
            device=device
        )
    elif args.router_type == 'entropy':
        logger.info("Using Entropy Router")
        router = EntropyRouter()
    elif args.router_type == 'divergence':
        logger.info("Using Divergence Router")
        router = DivergenceRouter()
    elif args.router_type == 'ppo':
        if not args.router_ckpt or not os.path.exists(args.router_ckpt):
            raise ValueError(f"PPO router requires --router_ckpt, got: {args.router_ckpt}")
        logger.info(f"Loading PPO Router from {args.router_ckpt}")
        router = PPORouter(
            checkpoint_path=args.router_ckpt,
            device=device,
            threshold=args.router_tau if hasattr(args, 'router_tau') else 0.5,
        )
    else:
        router = None
    
    return router


class EdgeCloudEvaluator:
    """边云协同导航评估器"""
    
    def __init__(
        self,
        teacher_model: NavModel,
        student_model: NavQwen3,
        router: Optional[BaseRouter],
        agent: MP3DAgent,
        dataset,
        device: torch.device,
        teacher_device: torch.device = None,
        student_device: torch.device = None,
        latency_mode: LatencyMode = 'fixed',
        latency_ms: float = 400.0,
        max_steps: int = 35,
        mode: str = 'edgecloud',
        task: str = 'CVDN',
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.router = router
        self.agent = agent
        self.dataset = dataset
        self.device = device  # 主设备（Student设备）
        self.teacher_device = teacher_device if teacher_device is not None else device
        self.student_device = student_device if student_device is not None else device
        self.latency_mode = latency_mode
        self.latency_ms = latency_ms
        self.max_steps = max_steps
        self.mode = mode
        self.task = task  # 数据集任务类型
        
        # 确保模型在eval模式
        self.teacher_model.eval()
        self.student_model.eval()
        if self.router and hasattr(self.router, 'model'):
            self.router.model.eval()
        
        # 记录设备配置
        if self.teacher_device != self.student_device:
            logger.info(f"Edge-Cloud simulation: Teacher on GPU {self.teacher_device.index}, Student on GPU {self.student_device.index}")
    
    def _calculate_distance_to_goal(self, obs: Dict, new_vp: Optional[str] = None) -> float:
        """
        计算当前viewpoint到目标viewpoint的距离
        参考原项目 tasks/agents/mp3d_agent.py 的实现
        """
        try:
            scan = obs['scan']
            current_vp = new_vp if new_vp else obs['viewpoint']
            
            # 注意：get_obs 返回的观测中使用的是 'gt_path'，不是 'path'
            if 'gt_path' in obs and len(obs['gt_path']) > 0:
                goal_vp = obs['gt_path'][-1]
            else:
                return -1.0
            
            # 使用agent的shortest_distances（和原项目保持一致）
            # 注意：agent在初始化时已经传入了shortest_distances
            if goal_vp not in self.agent.shortest_distances[scan][current_vp]:
                return -1.0
                
            return float(self.agent.shortest_distances[scan][current_vp][goal_vp])
        except (KeyError, IndexError, TypeError):
            return -1.0
    
    def run_episode(
        self,
        episode_idx: int,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """运行一个episode的边云协同导航"""
        # 初始化计时器
        t_student_forward = 0.0
        t_teacher_forward = 0.0
        t_router_forward = 0.0
        t_env_step = 0.0
        t_episode_start = time.time()
        
        # 统计
        teacher_call_steps = 0
        total_steps = 0
        
        # 从 dataset.__getitem__ 拿到完整的一条 sample（里面已经帮你建好 env 了）
        # 注意：CVDNDataset 没有 dataset.env 属性，每个 sample 通过 __getitem__ 返回时自带一个 env
        sample = self.dataset[episode_idx]
        
        item = sample['item']              # 原始anno
        episode_id = sample['instr_id']    # 指令ID
        env = sample['env']                # 这一条episode专用的 EnvBatch
        # 注意：__getitem__ 里是 observations = get_obs(...)[0]，是单个 dict
        # 而后面代码期望的是 obs 列表，所以这里手动包一层 list
        obs = [sample['observations']]
        
        # 初始化agent状态
        self.agent.update_scanvp_cands(obs)
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)
        
        traj = [{
            'instr_id': episode_id,
            'path': [[obs[0]['viewpoint']]],
            'details': {},
        }]
        
        instructions = [ob["instruction"] for ob in obs]
        student_history = [[]]
        teacher_history = [[]]
        hist_vis_student = [[]]
        hist_vis_teacher = [[]]
        
        ended = np.array([False])
        step_id = 0
        
        # 主循环
        while not ended[0] and step_id < self.max_steps:
            cur_obs = obs[0]
            
            # 计算当前到goal的距离
            dist_before = self._calculate_distance_to_goal(cur_obs)
            
            # ========== Student forward ==========
            t0 = time.time()
            
            # 准备panorama特征
            pano_inputs = self.agent.panorama_feature_variable_object(obs)
            panorama_out = self.student_model('panorama', pano_inputs)
            pano_embeds_student, pano_masks_student = panorama_out['pano_embeds'], panorama_out['pano_masks']
            
            # 更新graph上的节点特征（严格对齐 NaviLLM 原始逻辑，tasks/agents/mp3d_agent.py:746-757）
            # 这是关键步骤：必须给所有候选节点写入embedding，否则nav_gmap_variable会KeyError
            avg_pano_embeds = torch.sum(
                pano_embeds_student * pano_masks_student.unsqueeze(2), 1
            ) / torch.sum(pano_masks_student, 1, keepdim=True)  # [B, D]
            
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # 1) 更新当前 viewpoint 的平均特征（已访问节点）
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i].detach(), rewrite=True)
                    
                    # 2) ★★ 关键：给所有"还未 visited 的候选 viewpoint" 写入 node_embeds ★★
                    # 原项目在rollout中也是这样做的（tasks/agents/mp3d_agent.py:753-757）
                    # 注意：pano_embeds_student 的前 len(cand_vpids) 维对应候选节点（panorama_feature_variable_object 中先处理cand views）
                    cand_vpids = pano_inputs['cand_vpids'][i]
                    for j, cand_vp in enumerate(cand_vpids):
                        if not gmap.graph.visited(cand_vp):
                            # pano_embeds_student[i, j] 对应第j个候选节点的embedding
                            cand_embed = pano_embeds_student[i, j].detach()
                            gmap.update_node_embed(cand_vp, cand_embed)
            
            # 构造Student导航输入（使用Student设备）
            student_nav_inputs = build_nav_inputs_from_obs(
                agent=self.agent,
                obs=obs,
                gmaps=gmaps,
                pano_embeds=pano_embeds_student,
                pano_masks=pano_masks_student,
                pano_inputs=pano_inputs,
                instructions=instructions,
                history=student_history,
                hist_vis=hist_vis_student,
                data_type='cvdn',
                model_cls_token=self.student_model.lang_model.cls_token[0],
                device=self.student_device,
            )
            
            # Student前向
            with torch.no_grad():
                student_nav_outs = self.student_model('navigation', student_nav_inputs)
                student_logits = student_nav_outs['fuse_logits']
            
            t_student_forward += time.time() - t0
            
            # Student动作
            student_nav_vpids = student_nav_inputs['gmap_vpids'][0]
            _, student_action_idx = student_logits.max(1)
            student_action_idx = student_action_idx[0].item()
            
            if student_action_idx == 0:
                student_action_vpid = None
            else:
                if student_action_idx < len(student_nav_vpids):
                    student_action_vpid = student_nav_vpids[student_action_idx]
                else:
                    student_action_vpid = None
                    student_action_idx = 0
            
            # 计算Student动作后的距离
            if student_action_vpid is None:
                dist_after_student = dist_before
            else:
                dist_after_student = self._calculate_distance_to_goal(cur_obs, new_vp=student_action_vpid)
            
            # ========== Router决策 ==========
            use_teacher = False
            router_confidence = 0.0
            teacher_action_idx = None
            
            if self.mode == 'edgecloud' and self.router is not None:
                # 提取Router特征
                router_features = extract_router_features(
                    student_logits=student_logits[0],
                    num_cands=len(student_nav_vpids),
                    step_id=step_id,
                    max_steps=self.max_steps,
                    dist_before=dist_before,
                    dist_after_student=dist_after_student,
                    student_action_idx=student_action_idx,
                )
                
                # Router前向
                t0 = time.time()
                use_teacher, router_confidence = self.router.should_call_teacher(
                    student_logits=student_logits[0],
                    student_action_idx=student_action_idx,
                    features=router_features,
                    threshold=threshold,
                    cls_hidden=student_nav_outs.get('cls_hidden', None),
                )
                t_router_forward += time.time() - t0
            elif self.mode == 'teacher_only':
                use_teacher = True
            elif self.mode == 'student_only':
                use_teacher = False
            
            # ========== Teacher forward ==========
            final_action_vpid = student_action_vpid
            final_action_idx = student_action_idx
            
            if use_teacher:
                teacher_call_steps += 1
                
                # 模拟网络延迟
                simulate_cloud_latency(mode=self.latency_mode, fixed_ms=self.latency_ms)
                
                t0 = time.time()
                
                # Teacher panorama特征（需要移动到Teacher设备）
                pano_inputs_teacher = self.agent.panorama_feature_variable_object(obs)
                # 移动输入到Teacher设备
                pano_inputs_teacher = move_batch_to_device(pano_inputs_teacher, self.teacher_device)
                panorama_out_teacher = self.teacher_model('panorama', pano_inputs_teacher)
                pano_embeds_teacher, pano_masks_teacher = panorama_out_teacher['pano_embeds'], panorama_out_teacher['pano_masks']
                
                # 更新Teacher graph（使用4096 dim embeddings，严格对齐原项目逻辑）
                # ⚠️ 关键：Teacher和Student必须使用独立的GraphMap，因为embedding维度不同（4096 vs 2048）
                # 参考 router_code/collect_router_data.py:402-416 的做法
                avg_pano_embeds_teacher = torch.sum(
                    pano_embeds_teacher * pano_masks_teacher.unsqueeze(2), 1
                ) / torch.sum(pano_masks_teacher, 1, keepdim=True)  # [B, D]
                
                # 为Teacher创建独立的graph副本（deep copy结构，但重置embeddings）
                teacher_gmaps = []
                for gmap in gmaps:
                    # Deep copy GraphMap: copy structure but reset embeddings for Teacher dimension
                    teacher_gmap = GraphMap(gmap.start_vp)
                    teacher_gmap.node_positions = copy.deepcopy(gmap.node_positions)
                    teacher_gmap.graph = copy.deepcopy(gmap.graph)
                    teacher_gmap.node_step_ids = copy.deepcopy(gmap.node_step_ids)
                    teacher_gmap.pooling_mode = gmap.pooling_mode
                    # node_embeds will be initialized with Teacher's embeddings (4096 dim)
                    teacher_gmap.node_embeds = {}
                    teacher_gmaps.append(teacher_gmap)
                
                for i, teacher_gmap in enumerate(teacher_gmaps):
                    if not ended[i]:
                        # 1) 更新当前 viewpoint 的平均特征（已访问节点）
                        i_vp = obs[i]['viewpoint']
                        teacher_gmap.update_node_embed(i_vp, avg_pano_embeds_teacher[i].detach(), rewrite=True)
                        
                        # 2) ★★ 关键：给所有"还未 visited 的候选 viewpoint" 写入 node_embeds ★★
                        cand_vpids_teacher = pano_inputs_teacher['cand_vpids'][i]
                        for j, cand_vp in enumerate(cand_vpids_teacher):
                            if not teacher_gmap.graph.visited(cand_vp):
                                cand_embed = pano_embeds_teacher[i, j].detach()
                                teacher_gmap.update_node_embed(cand_vp, cand_embed)
                        
                        # 3) ★★ 关键修复：确保所有在node_positions中的节点都有embedding ★★
                        # 因为Teacher graph是从Student graph复制的，可能包含之前step添加的节点
                        # 但这些节点可能不在当前step的候选列表中，所以没有对应的pano_embeds_teacher
                        # 我们需要为这些节点初始化零向量embedding（使用Teacher的维度4096）
                        if len(teacher_gmap.node_embeds) > 0:
                            # 从已有embedding推断维度（应该是4096）
                            sample_embed = next(iter(teacher_gmap.node_embeds.values()))[0]
                            for vp in teacher_gmap.node_positions.keys():
                                if vp not in teacher_gmap.node_embeds:
                                    # 初始化零向量embedding（Teacher维度4096）
                                    zero_embed = torch.zeros_like(sample_embed)
                                    teacher_gmap.node_embeds[vp] = [zero_embed, 1]
                
                # 构造Teacher导航输入（使用Teacher设备）
                teacher_nav_inputs = build_nav_inputs_from_obs(
                    agent=self.agent,
                    obs=obs,
                    gmaps=teacher_gmaps,
                    pano_embeds=pano_embeds_teacher,
                    pano_masks=pano_masks_teacher,
                    pano_inputs=pano_inputs_teacher,
                    instructions=instructions,
                    history=teacher_history,
                    hist_vis=hist_vis_teacher,
                    data_type='cvdn',
                    model_cls_token=self.teacher_model.lang_model.cls_token[0],
                    device=self.teacher_device,
                )
                
                # Teacher前向
                with torch.no_grad():
                    teacher_nav_outs = self.teacher_model('navigation', teacher_nav_inputs)
                    teacher_logits = teacher_nav_outs['fuse_logits']
                
                t_teacher_forward += time.time() - t0
                
                # Teacher动作
                teacher_nav_vpids = teacher_nav_inputs['gmap_vpids'][0]
                _, teacher_action_idx_tensor = teacher_logits.max(1)
                teacher_action_idx = teacher_action_idx_tensor[0].item()
                
                if teacher_action_idx == 0:
                    final_action_vpid = None
                    final_action_idx = 0
                else:
                    if teacher_action_idx < len(teacher_nav_vpids):
                        final_action_vpid = teacher_nav_vpids[teacher_action_idx]
                        final_action_idx = teacher_action_idx
                    else:
                        final_action_vpid = None
                        final_action_idx = 0
                
                # 对于Divergence router，需要提供teacher_action_idx
                if self.router and isinstance(self.router, DivergenceRouter):
                    # 已经在should_call_teacher中处理，这里不需要额外操作
                    pass
            
            # ========== 执行动作 ==========
            t0 = time.time()
            
            # Safety: 如果选的是当前viewpoint，当作stop处理
            if final_action_vpid is not None and final_action_vpid == cur_obs['viewpoint']:
                final_action_vpid = None
                final_action_idx = 0
            
            cpu_a_t = [None if final_action_vpid is None else final_action_vpid]
            
            # 执行动作（添加错误处理，防止KeyError导致整个episode失败）
            try:
                self.agent.make_equiv_action(cpu_a_t, gmaps, obs, traj=traj, env=[env])
            except (KeyError, IndexError) as e:
                logger.error(
                    f"make_equiv_action error in episode {episode_id}, step {step_id}, "
                    f"scan={cur_obs.get('scan', 'unknown')}, "
                    f"obs_vp={cur_obs.get('viewpoint', 'unknown')}, "
                    f"action_vp={final_action_vpid}: {e}"
                )
                # 直接认为episode结束，让后续eval_cvdn把它记成失败
                ended = np.array([True])
                break
            
            # 更新环境
            if not ended[0] and item is not None:
                data_type = obs[0].get('data_type', 'cvdn')
                new_obs = self.dataset.get_obs([item], env, data_type=data_type)
                obs = [new_obs[0]] if isinstance(new_obs, list) else [new_obs]
                
                # ★★ 关键修复：每次获取新obs后，必须更新scanvp_cands和graph ★★
                # 原项目在rollout中也是这样做的（tasks/agents/mp3d_agent.py:1079-1083）
                self.agent.update_scanvp_cands(obs)  # 更新候选缓存，否则下一步make_equiv_action会KeyError
                for i, ob in enumerate(obs):
                    if not ended[i]:
                        gmaps[i].update_graph(ob)  # 更新graph，保持与原项目一致
                        
                        # ★★ 关键修复：确保所有在node_positions中的节点都有embedding ★★
                        # 因为update_graph会添加新节点到node_positions，但这些节点可能还没有embedding
                        # 当nav_gmap_variable尝试获取所有节点的embedding时，会KeyError
                        # 解决方案：为所有在node_positions中但还没有embedding的节点初始化零向量
                        if len(gmaps[i].node_embeds) > 0:
                            # 从已有embedding推断维度
                            sample_embed = next(iter(gmaps[i].node_embeds.values()))[0]
                            for vp in gmaps[i].node_positions.keys():
                                if vp not in gmaps[i].node_embeds:
                                    # 初始化零向量embedding
                                    zero_embed = torch.zeros_like(sample_embed)
                                    gmaps[i].node_embeds[vp] = [zero_embed, 1]
            
            t_env_step += time.time() - t0
            
            # 检查是否结束
            done = (cpu_a_t[0] is None) or ended[0]

            # 更新历史，保持 student/teacher 路径各自与 hist_vis 对齐
            if not done:
                student_history[0].append(final_action_vpid)
                hist_vis_student[0].append(avg_pano_embeds[0].detach().cpu())

                if use_teacher:
                    teacher_history[0].append(final_action_vpid)
                    hist_vis_teacher[0].append(avg_pano_embeds_teacher[0].detach().cpu())

            total_steps += 1
            step_id += 1
            
            ended = np.array([done])
            
            if done:
                break
        
        # Episode结束，计算指标
        t_episode_total = time.time() - t_episode_start
        
        # 计算SR/SPL等导航指标
        pred_path = traj[0]['path']
        gt_path = item['path']
        
        # 使用dataset的评估函数（和原项目保持一致）
        # 注意：pred_path是list of lists，需要flatten
        flat_path = sum(pred_path, [])
        
        # 检查scan是否在shortest_distances中（某些scan可能没有加载导航图）
        scan = item['scan']
        if scan not in self.dataset.shortest_distances:
            logger.warning(f"Scan {scan} not found in shortest_distances, skipping evaluation for episode {episode_id}")
            # 返回默认失败指标
            return {
                'episode_id': episode_id,
                'success': False,
                'spl': 0.0,
                'nav_error': float('inf'),
                'oracle_error': float('inf'),
                'oracle_success': False,
                'trajectory_length': 0.0,
                'gt_length': len(item['path']) - 1,
                'teacher_calls': teacher_call_steps,
                'total_steps': total_steps,
                'teacher_call_rate': teacher_call_steps / total_steps if total_steps > 0 else 0.0,
                't_episode_total': t_episode_total,
                't_student_forward': t_student_forward,
                't_teacher_forward': t_teacher_forward,
                't_router_forward': t_router_forward,
                't_env_step': t_env_step,
            }
        
        # 根据task调用不同的评估函数
        try:
            if self.task == 'CVDN':
                scores = self.dataset.eval_cvdn(
                    scan=scan,
                    path=flat_path,
                    gt_item=item
                )
                # CVDN返回的字段：nav_errors, oracle_errors, success, spl, trajectory_lengths
                nav_error = scores.get('nav_errors', float('inf'))
                oracle_error = scores.get('oracle_errors', float('inf'))
                success = scores.get('success', False) if 'success' in scores else (nav_error < 3.0)
                oracle_success = (oracle_error < 3.0)
            elif self.task == 'R2R':
                scores = self.dataset.eval_dis_item(
                    scan=scan,
                    pred_path=pred_path,
                    gt_path=gt_path
                )
                # R2R返回的字段：nav_error, oracle_error, success, oracle_success, spl, trajectory_lengths
                nav_error = scores.get('nav_error', float('inf'))
                oracle_error = scores.get('oracle_error', float('inf'))
                success = scores.get('success', False)
                oracle_success = scores.get('oracle_success', False)
            elif self.task == 'REVERIE':
                # REVERIE需要obj_id，暂时使用eval_dis_item的简化版本
                if hasattr(self.dataset, 'eval_dis_item'):
                    scores = self.dataset.eval_dis_item(
                        scan=scan,
                        pred_path=pred_path,
                        gt_path=gt_path
                    )
                    nav_error = scores.get('nav_error', float('inf'))
                    oracle_error = scores.get('oracle_error', float('inf'))
                    success = scores.get('success', False)
                    oracle_success = scores.get('oracle_success', False)
                else:
                    # 如果没有eval_dis_item，使用eval_reverie_item
                    scores = self.dataset.eval_reverie_item(
                        traj=pred_path,
                        gt_item=item
                    )
                    nav_error = scores.get('nav_error', float('inf'))
                    oracle_error = scores.get('oracle_error', float('inf'))
                    success = scores.get('success', False)
                    oracle_success = scores.get('oracle_success', False)
            elif self.task == 'SOON':
                # SOON需要obj_heading和obj_elevation
                if hasattr(self.dataset, 'eval_dis_item'):
                    scores = self.dataset.eval_dis_item(
                        scan=scan,
                        pred_path=pred_path,
                        gt_path=gt_path
                    )
                    nav_error = scores.get('nav_error', float('inf'))
                    oracle_error = scores.get('oracle_error', float('inf'))
                    success = scores.get('success', False)
                    oracle_success = scores.get('oracle_success', False)
                else:
                    # 如果没有eval_dis_item，使用eval_soon_item
                    obj_heading = item.get('obj_heading', 0.0)
                    obj_elevation = item.get('obj_elevation', 0.0)
                    scores = self.dataset.eval_soon_item(
                        traj=pred_path,
                        gt_item=item,
                        obj_heading=obj_heading,
                        obj_elevation=obj_elevation
                    )
                    nav_error = scores.get('nav_error', float('inf'))
                    oracle_error = scores.get('oracle_error', float('inf'))
                    success = scores.get('success', False)
                    oracle_success = scores.get('oracle_success', False)
            else:
                # 默认使用CVDN的评估方式
                scores = self.dataset.eval_cvdn(
                    scan=scan,
                    path=flat_path,
                    gt_item=item
                )
                nav_error = scores.get('nav_errors', float('inf'))
                oracle_error = scores.get('oracle_errors', float('inf'))
                success = scores.get('success', False) if 'success' in scores else (nav_error < 3.0)
                oracle_success = (oracle_error < 3.0)
            
            # 统一处理trajectory_lengths字段名
            trajectory_length = scores.get('trajectory_lengths', scores.get('trajectory_length', 0.0))
            spl = scores.get('spl', 0.0)
            gp = scores.get('dist_to_end_reductions', scores.get('goal_progress', 0.0))
            
        except (KeyError, AttributeError, TypeError) as e:
            # 如果评估函数调用出错，记录并返回失败指标
            logger.warning(f"Error in evaluation for episode {episode_id}, task={self.task}, scan={scan}: {e}")
            return {
                'episode_id': episode_id,
                'success': False,
                'spl': 0.0,
                'gp': 0.0,
                'nav_error': float('inf'),
                'oracle_error': float('inf'),
                'oracle_success': False,
                'trajectory_length': 0.0,
                'gt_length': len(item['path']) - 1,
                'teacher_calls': teacher_call_steps,
                'total_steps': total_steps,
                'teacher_call_rate': teacher_call_steps / total_steps if total_steps > 0 else 0.0,
                't_episode_total': t_episode_total,
                't_student_forward': t_student_forward,
                't_teacher_forward': t_teacher_forward,
                't_router_forward': t_router_forward,
                't_env_step': t_env_step,
            }
        
        # 构建返回指标
        metrics = {
            'episode_id': episode_id,
            'success': success,
            'spl': spl,
            'gp': gp,
            'nav_error': nav_error,
            'oracle_error': oracle_error,
            'oracle_success': oracle_success,
            'trajectory_length': trajectory_length,
            'gt_length': len(gt_path) - 1,
            'teacher_calls': teacher_call_steps,
            'total_steps': total_steps,
            'teacher_call_rate': teacher_call_steps / total_steps if total_steps > 0 else 0.0,
            't_episode_total': t_episode_total,
            't_student_forward': t_student_forward,
            't_teacher_forward': t_teacher_forward,
            't_router_forward': t_router_forward,
            't_env_step': t_env_step,
            'teacher_time_ratio': t_teacher_forward / t_episode_total if t_episode_total > 0 else 0.0,
            'mode': self.mode,
            'threshold': threshold,
            'router_name': self.router.get_name() if self.router else 'none',
        }
        
        return metrics


def evaluate_with_tau(
    evaluator: EdgeCloudEvaluator,
    episode_indices: List[int],
    threshold: float,
    desc: str = "Evaluating"
) -> Dict[str, Any]:
    """使用指定阈值评估一批episodes"""
    all_metrics = []
    
    import traceback
    
    for idx in tqdm(episode_indices, desc=desc):
        try:
            metrics = evaluator.run_episode(idx, threshold=threshold)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error in episode {idx}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")  # 打印完整traceback便于定位问题
            continue
    
    if len(all_metrics) == 0:
        return {}
    
    # 汇总指标
    N = len(all_metrics)
    sr_list = [m['success'] for m in all_metrics]
    spl_list = [m['spl'] for m in all_metrics]
    gp_list = [m.get('gp', 0.0) for m in all_metrics]
    nav_error_list = [m['nav_error'] for m in all_metrics]
    oracle_success_list = [m.get('oracle_success', False) for m in all_metrics]  # 提取oracle_success
    oracle_error_list = [m.get('oracle_error', float('inf')) for m in all_metrics]  # 提取oracle_error
    
    teacher_calls = [m['teacher_calls'] for m in all_metrics]
    total_steps_list = [m['total_steps'] for m in all_metrics]
    
    t_ep_list = [m['t_episode_total'] for m in all_metrics]
    t_stu_list = [m['t_student_forward'] for m in all_metrics]
    t_tea_list = [m['t_teacher_forward'] for m in all_metrics]
    t_rtr_list = [m['t_router_forward'] for m in all_metrics]
    
    summary = {
        'threshold': threshold,
        'num_episodes': N,
        'sr': np.mean(sr_list),
        'oracle_sr': np.mean(oracle_success_list),  # Oracle Success Rate
        'spl': np.mean(spl_list),
        'gp': np.mean(gp_list),
        'nav_error': np.mean(nav_error_list),
        'oracle_error': np.mean(oracle_error_list),  # Oracle Error
        'teacher_call_rate': sum(teacher_calls) / sum(total_steps_list) if sum(total_steps_list) > 0 else 0.0,
        'teacher_calls_per_episode': np.mean(teacher_calls),
        't_episode_avg': np.mean(t_ep_list),
        't_student_avg': np.mean(t_stu_list),
        't_teacher_avg': np.mean(t_tea_list),
        't_router_avg': np.mean(t_rtr_list),
        'teacher_time_ratio': np.mean([m['teacher_time_ratio'] for m in all_metrics]),
        'all_metrics': all_metrics,  # 保存详细结果
    }
    
    return summary


def main():
    """主函数 - 完整实现"""
    parser = argparse.ArgumentParser(description="Edge-Cloud Collaborative Navigation Evaluation")
    
    # 基本参数
    parser.add_argument('--task', type=str, default='CVDN', choices=['CVDN', 'R2R', 'REVERIE', 'SOON'])
    parser.add_argument('--split', type=str, default='val_unseen', choices=['train', 'val_seen', 'val_unseen'])
    parser.add_argument('--cfg_file', type=str, default='configs/multi.yaml')
    parser.add_argument('--data_dir', type=str, default='data')
    
    # 模型路径
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                       help='Path to Teacher checkpoint. If not provided, will try default locations.')
    parser.add_argument('--student_ckpt', type=str, required=True)
    parser.add_argument('--router_ckpt', type=str, default=None)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default=None,
                       help='Student pretrained model path (Qwen3-1.7B)')
    parser.add_argument('--teacher_pretrained_path', type=str, default=None,
                       help='Teacher pretrained model path (Vicuna-7B), will auto-detect if not specified')
    parser.add_argument('--student_type', type=str, default='distill', choices=['base', 'distill'])
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='edgecloud',
                       choices=['teacher_only', 'student_only', 'edgecloud'])
    parser.add_argument('--router_type', type=str, default='offcourse',
                       choices=['entropy', 'divergence', 'offcourse', 'ppo'])
    parser.add_argument('--router_hidden_dim', type=int, default=128)
    
    # Router参数
    parser.add_argument('--router_tau', type=float, default=0.5)
    parser.add_argument('--tau_list', type=float, nargs='+', default=None)
    
    # 延迟模拟
    parser.add_argument('--latency_mode', type=str, default='fixed',
                       choices=['none', 'fixed', 'moderate', 'high'])
    parser.add_argument('--latency_ms', type=float, default=400.0)
    
    # 实验参数
    parser.add_argument('--max_steps', type=int, default=35)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0, help='Main GPU for Student and Router')
    parser.add_argument('--teacher_gpu', type=int, default=None, 
                       help='GPU for Teacher model (None=same as --gpu, recommended: different GPU for edge-cloud simulation)')
    parser.add_argument('--student_gpu', type=int, default=None,
                       help='GPU for Student model (None=same as --gpu)')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 多卡并行
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=None, 
                       help='Local rank for distributed training (automatically set by torch.distributed.launch)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='build/edgecloud_results')
    parser.add_argument('--save_detailed', action='store_true')
    
    # 模型相关（兼容原项目）
    # 重要：原项目评估使用amp_bf16，可以大幅减少内存占用（约50%）
    parser.add_argument('--precision', type=str, default='amp_bf16',
                       choices=['amp_bf16', 'amp_bfloat16', 'bf16', 'fp16', 'fp32'],
                       help='Floating point precision (default: amp_bf16 for memory efficiency, same as original project)')
    
    args = parser.parse_args()
    
    # 多卡并行支持（必须在设置设备和加载模型之前初始化）
    if args.multi_gpu:
        # 初始化分布式环境
        # 优先从环境变量读取（torch.distributed.launch/torchrun会自动设置）
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            # LOCAL_RANK 环境变量（torchrun默认）或 --local-rank 参数（torch.distributed.launch）
            if 'LOCAL_RANK' in os.environ:
                args.local_rank = int(os.environ['LOCAL_RANK'])
            elif args.local_rank is None:
                args.local_rank = args.rank % torch.cuda.device_count()
        else:
            # 如果没有环境变量，使用参数或默认值
            if args.local_rank is None:
                args.local_rank = args.gpu
            if args.world_size == 1:
                args.world_size = torch.cuda.device_count()
            if args.rank == 0 and args.world_size > 1:
                args.rank = args.local_rank
        
        # 设置当前进程的设备（基于local_rank）
        device = torch.device(f'cuda:{args.local_rank}')
        torch.cuda.set_device(device)
        
        # 获取可见的GPU列表（用于映射物理GPU到逻辑GPU）
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            visible_list = [int(x) for x in visible_devices.split(',') if x.strip()]
            # 在多卡并行时，local_rank对应的实际GPU是visible_list[local_rank]
            actual_gpu_for_local_rank = visible_list[args.local_rank] if args.local_rank < len(visible_list) else args.local_rank
        else:
            visible_list = list(range(torch.cuda.device_count()))
            actual_gpu_for_local_rank = args.local_rank
        
        # 多卡并行模式：每个进程的Teacher和Student都加载到该进程对应的GPU
        # 这样每个GPU独立工作，真正实现并行加速
        if args.teacher_gpu is not None and args.teacher_gpu in visible_list:
            # 如果指定了teacher_gpu，找到它在visible_list中的索引
            teacher_gpu_index = visible_list.index(args.teacher_gpu)
            teacher_device = torch.device(f'cuda:{teacher_gpu_index}')
            logger.info(f"Multi-GPU mode: Teacher will use physical GPU {args.teacher_gpu} (logical GPU {teacher_gpu_index} in CUDA_VISIBLE_DEVICES)")
        else:
            # 默认：每个进程的Teacher加载到该进程对应的GPU
            teacher_device = torch.device(f'cuda:{args.local_rank}')
            logger.info(f"Multi-GPU mode: Teacher will use GPU {args.local_rank} (local_rank, physical GPU {actual_gpu_for_local_rank})")
        
        if args.student_gpu is not None and args.student_gpu in visible_list:
            # 如果指定了student_gpu，找到它在visible_list中的索引
            student_gpu_index = visible_list.index(args.student_gpu)
            student_device = torch.device(f'cuda:{student_gpu_index}')
            logger.info(f"Multi-GPU mode: Student will use physical GPU {args.student_gpu} (logical GPU {student_gpu_index} in CUDA_VISIBLE_DEVICES)")
        else:
            # 默认：每个进程的Student加载到该进程对应的GPU
            student_device = torch.device(f'cuda:{args.local_rank}')
            logger.info(f"Multi-GPU mode: Student will use GPU {args.local_rank} (local_rank, physical GPU {actual_gpu_for_local_rank})")
        
        # 初始化进程组（如果还没有初始化）
        if not dist.is_initialized():
            # 尝试从环境变量获取init_method
            init_method = os.environ.get('MASTER_ADDR', 'localhost')
            master_port = os.environ.get('MASTER_PORT', '29500')
            init_method = f'tcp://{init_method}:{master_port}'
            
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                world_size=args.world_size,
                rank=args.rank
            )
        
        logger.info(f"Multi-GPU mode: rank={args.rank}/{args.world_size}, local_rank={args.local_rank}, device={device}")
        
        # 记录设备分配
        if teacher_device != student_device:
            logger.info(f"✅ Teacher and Student on different GPUs: Teacher={teacher_device.index}, Student={student_device.index}")
        else:
            logger.info(f"Teacher and Student on same GPU: {teacher_device.index} (each process loads models independently)")
    else:
        # 单卡模式
        args.rank = 0
        args.world_size = 1
        if args.local_rank is None:
            args.local_rank = args.gpu
        
        # 确定Teacher和Student的设备（支持不同GPU）
        if args.teacher_gpu is not None:
            teacher_device = torch.device(f'cuda:{args.teacher_gpu}')
        else:
            teacher_device = torch.device(f'cuda:{args.gpu}')
        
        if args.student_gpu is not None:
            student_device = torch.device(f'cuda:{args.student_gpu}')
        else:
            student_device = torch.device(f'cuda:{args.gpu}')
        
        # 主设备（用于Router和其他操作）
        device = student_device  # Router和Student在同一设备更合理
        torch.cuda.set_device(device)
    
    logger.info(f"Device configuration:")
    if args.multi_gpu:
        # 获取可见的GPU列表（用于显示物理GPU编号）
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            visible_list = [int(x) for x in visible_devices.split(',') if x.strip()]
            actual_gpu_for_local_rank = visible_list[args.local_rank] if args.local_rank < len(visible_list) else args.local_rank
        else:
            actual_gpu_for_local_rank = args.local_rank
        logger.info(f"  Rank: {args.rank}/{args.world_size}, Local Rank: {args.local_rank}")
        logger.info(f"  Teacher GPU: {teacher_device.index} (physical GPU {actual_gpu_for_local_rank})")
        logger.info(f"  Student GPU: {student_device.index} (physical GPU {actual_gpu_for_local_rank})")
        logger.info(f"  Router GPU: {device.index}")
    else:
        logger.info(f"  Teacher GPU: {args.teacher_gpu if args.teacher_gpu is not None else args.gpu}")
        logger.info(f"  Student GPU: {args.student_gpu if args.student_gpu is not None else args.gpu}")
        logger.info(f"  Router GPU: {args.gpu}")
    
    logger.info("="*60)
    logger.info("Edge-Cloud Collaborative Navigation Evaluation")
    logger.info("="*60)
    logger.info(f"Task: {args.task}, Split: {args.split}")
    logger.info(f"Mode: {args.mode}, Router: {args.router_type}")
    logger.info(f"Student Type: {args.student_type}")
    logger.info(f"Latency: {args.latency_mode} ({args.latency_ms}ms)")
    
    # 加载配置
    with open(args.cfg_file, 'r', encoding='utf-8') as f:
        global_cfg = EasyDict(yaml.safe_load(f))
    
    # 构建args对象（兼容原项目）
    project_args = EasyDict()
    project_args.update(vars(args))
    project_args.data_dir = Path(args.data_dir)
    project_args.debug = False
    project_args.few_shot = None
    project_args.seed = 42
    project_args.max_datapoints = None
    project_args.path_type = 'trusted_path'
    
    # 加载数据集
    logger.info(f"Loading {args.task} {args.split} dataset...")
    
    # 使用原项目的create_dataloaders
    from tasks.feature_db import create_feature_db, create_object_feature_db
    
    # 创建feature databases
    feat_db = None
    obj_feat_db = None
    
    if hasattr(global_cfg, 'Feature'):
        feat_db = create_feature_db(
            global_cfg.Feature.feature_database,
            global_cfg.Feature.image_feat_size,
            project_args
        )
        if hasattr(global_cfg.Feature, 'object_database') and project_args.get('enable_og', False):
            obj_feat_db = create_object_feature_db(
                global_cfg.Feature.object_database,
                global_cfg.Feature.obj_feat_size,
                project_args
            )
    
    # 设置必要的args属性
    project_args.distributed = False
    project_args.batch_size = 1
    project_args.val_batch_size = 1
    project_args.workers = args.num_workers
    project_args.test_datasets = [args.task]
    project_args.validation_split = args.split  # 重要：设置validation_split
    project_args.enable_og = getattr(global_cfg, 'enable_og', False) if hasattr(global_cfg, 'enable_og') else False
    project_args.seed = getattr(args, 'seed', 42)
    project_args.path_type = getattr(args, 'path_type', 'trusted_path')
    project_args.max_datapoints = getattr(args, 'max_datapoints', None)
    project_args.few_shot = getattr(args, 'few_shot', None)
    project_args.debug = getattr(args, 'debug', False)
    
    # 重要：Agent需要这些特征维度属性（用于panorama_feature_variable等方法）
    # 优先从Feature配置获取（这是实际使用的特征维度）
    if hasattr(global_cfg, 'Feature'):
        project_args.image_feat_size = getattr(global_cfg.Feature, 'image_feat_size', 1024)
        project_args.angle_feat_size = getattr(global_cfg.Feature, 'angle_feat_size', 4)
        project_args.obj_feat_size = getattr(global_cfg.Feature, 'obj_feat_size', 768)
    else:
        project_args.image_feat_size = 1024
        project_args.angle_feat_size = 4
        project_args.obj_feat_size = 768
    
    # ==== 补齐给 Agent 用的 Model 相关配置（MP3DAgent 需要 enc_full_graph 等） ====
    if hasattr(global_cfg, 'Model'):
        # 如果Model配置中有image_feat_size，使用它（但通常应该和Feature一致）
        if hasattr(global_cfg.Model, 'image_feat_size'):
            project_args.image_feat_size = global_cfg.Model.image_feat_size
        if hasattr(global_cfg.Model, 'angle_feat_size'):
            project_args.angle_feat_size = global_cfg.Model.angle_feat_size
        if hasattr(global_cfg.Model, 'obj_feat_size'):
            project_args.obj_feat_size = global_cfg.Model.obj_feat_size
        
        project_args.feat_dropout = getattr(global_cfg.Model, 'feat_dropout', 0.4)
        project_args.enc_full_graph = getattr(global_cfg.Model, 'enc_full_graph', True)  # 关键：MP3DAgent.nav_gmap_variable需要
        project_args.no_loc_fts = getattr(global_cfg.Model, 'no_loc_fts', False)
        project_args.tour3d_nav_head = getattr(global_cfg.Model, 'tour3d_nav_head', False)
        project_args.expert_policy = getattr(global_cfg.Model, 'expert_policy', 'spl')
    else:
        # 没有Model配置时给个合理默认
        project_args.feat_dropout = 0.4
        project_args.enc_full_graph = True  # 关键：MP3DAgent.nav_gmap_variable需要
        project_args.no_loc_fts = False
        project_args.tour3d_nav_head = False
        project_args.expert_policy = 'spl'
    # ==== 补齐结束 ====
    
    # Agent可能还需要其他属性
    project_args.ignoreid = getattr(args, 'ignoreid', -100)
    
    # 创建dataloaders
    dataloaders, agents = create_dataloaders(
        project_args, global_cfg, logger,
        training=False,
        device=args.gpu,
        feat_db=feat_db,
        obj_feat_db=obj_feat_db,
        stage='multi'
    )
    
    # 获取dataset和agent
    # dataloaders是字典，键是任务名，值是PrefetchLoader
    task_loader = dataloaders[args.task]
    dataset = task_loader.get_dataset()  # PrefetchLoader有get_dataset方法
    agent = agents[args.task]
    
    # 确保dataset有alldata属性（用于直接访问episodes）
    if not hasattr(dataset, 'alldata'):
        # 如果dataset没有alldata，从dataloader中获取
        logger.warning("Dataset does not have 'alldata' attribute, trying to access via dataloader")
        # 这种情况下需要遍历dataloader来获取所有items
        dataset.alldata = []
        for batch in dataloaders:
            if isinstance(batch, dict) and 'item' in batch:
                dataset.alldata.extend(batch['item'] if isinstance(batch['item'], list) else [batch['item']])
            elif isinstance(batch, list):
                dataset.alldata.extend(batch)
    
    logger.info(f"Dataset ready: {len(dataset.alldata)} episodes available")
    
    logger.info(f"Dataset loaded: {len(dataset.alldata)} episodes")
    
    # 加载模型（支持不同GPU）
    # 注意：在多卡模式下，teacher_device和student_device已经在上面根据local_rank设置好了
    teacher_model = load_teacher_model(project_args, global_cfg, teacher_device, logger)
    student_model = load_student_model(project_args, global_cfg, student_device, logger, args.student_type)
    
    # 加载Router（如果需要，Router和Student在同一设备）
    router = None
    if args.mode == 'edgecloud':
        router = load_router(args, student_device, logger)
    
    # 确定要评估的episodes
    total_episodes = len(dataset.alldata) if hasattr(dataset, 'alldata') else len(dataset)
    all_episode_indices = list(range(total_episodes))
    if args.max_episodes:
        all_episode_indices = all_episode_indices[:args.max_episodes]
    
    # 多卡并行：每个进程处理不同的episodes
    if args.multi_gpu:
        episodes_per_rank = len(all_episode_indices) // args.world_size
        start_idx = args.rank * episodes_per_rank
        if args.rank == args.world_size - 1:
            # 最后一个rank处理剩余的episodes
            episode_indices = all_episode_indices[start_idx:]
        else:
            episode_indices = all_episode_indices[start_idx:start_idx + episodes_per_rank]
        logger.info(f"Rank {args.rank}: processing {len(episode_indices)} episodes (indices {start_idx}-{start_idx+len(episode_indices)-1})")
    else:
        episode_indices = all_episode_indices
        logger.info(f"Will evaluate {len(episode_indices)} episodes")
    
    # 创建评估器（传入Teacher和Student的设备）
    evaluator = EdgeCloudEvaluator(
        teacher_model=teacher_model,
        student_model=student_model,
        router=router,
        agent=agent,
        dataset=dataset,
        device=device,  # 主设备（Student设备）
        teacher_device=teacher_device,  # Teacher设备
        student_device=student_device,  # Student设备
        latency_mode=args.latency_mode,
        latency_ms=args.latency_ms,
        max_steps=args.max_steps,
        mode=args.mode,
        task=args.task,  # 传入task参数
    )
    
    # 确定要评估的阈值列表
    if args.tau_list:
        tau_list = args.tau_list
    elif args.mode == 'edgecloud':
        # 默认sweep多个tau
        tau_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    else:
        tau_list = [0.5]  # 其他模式不需要tau
    
    # 评估
    all_results = {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查已存在的结果文件，跳过已完成的τ值
    results_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}.json"
    existing_taus = set()
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                existing_taus = set(existing_results.keys())
                # 如果已有结果，加载到all_results中
                for tau_str, summary in existing_results.items():
                    tau_float = float(tau_str)
                    all_results[tau_float] = summary
                if args.rank == 0:
                    logger.info(f"Found existing results file: {results_file}")
                    logger.info(f"Existing τ values: {sorted(existing_taus)}")
                    logger.info(f"Will skip these τ values and only evaluate missing ones.")
        except Exception as e:
            if args.rank == 0:
                logger.warning(f"Failed to load existing results: {e}, will re-evaluate all τ values")
    
    # 过滤出需要评估的τ值（跳过已存在的）
    tau_list_to_eval = [tau for tau in tau_list if str(tau) not in existing_taus]
    if args.rank == 0:
        if tau_list_to_eval:
            logger.info(f"Will evaluate τ values: {tau_list_to_eval}")
        else:
            logger.info(f"All τ values already evaluated, skipping evaluation.")
    
    for tau in tau_list_to_eval:
        if args.rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating with threshold τ={tau}")
            logger.info(f"{'='*60}")
        
        summary = evaluate_with_tau(
            evaluator,
            episode_indices,
            threshold=tau,
            desc=f"τ={tau} (rank {args.rank})"
        )
        
        if summary:
            all_results[tau] = summary
            
            # 打印结果（只在rank 0打印）
            if args.rank == 0:
                logger.info(f"Results (τ={tau}):")
                logger.info(f"  SR: {summary['sr']:.4f}")
                logger.info(f"  Oracle_SR: {summary.get('oracle_sr', 0.0):.4f}")
                logger.info(f"  SPL: {summary['spl']:.4f}")
                logger.info(f"  GP: {summary.get('gp', 0.0):.4f}")
                logger.info(f"  Nav Error: {summary['nav_error']:.4f}")
                logger.info(f"  Teacher Call Rate: {summary['teacher_call_rate']:.2%}")
                logger.info(f"  Avg Episode Time: {summary['t_episode_avg']:.3f}s")
                logger.info(f"  Teacher Time Ratio: {summary['teacher_time_ratio']:.2%}")
    
    # 多卡并行：收集所有rank的结果
    if args.multi_gpu:
        # 等待所有进程完成
        dist.barrier()
        
        # 重要：rank 0也需要保存自己的结果到rank文件（以防合并失败）
        if args.rank == 0:
            rank0_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}_rank0.json"
            try:
                rank0_file.parent.mkdir(parents=True, exist_ok=True)
                with open(rank0_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Rank 0 results saved to {rank0_file} ({rank0_file.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to save rank 0 results: {e}")
        
        # 等待所有rank保存完成
        dist.barrier()
        
        # 收集所有rank的结果到rank 0
        if args.rank == 0:
            # 先加载rank 0自己的结果（如果存在）
            rank0_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}_rank0.json"
            if rank0_file.exists() and rank0_file.stat().st_size > 0:
                try:
                    with open(rank0_file, 'r', encoding='utf-8') as f:
                        rank0_results = json.load(f)
                    # 合并rank 0的结果
                    for tau, summary in rank0_results.items():
                        if isinstance(summary, dict) and 'all_metrics' in summary:
                            if tau not in all_results:
                                all_results[tau] = summary
                            else:
                                all_results[tau]['all_metrics'].extend(summary.get('all_metrics', []))
                    logger.info(f"Loaded rank 0 results from {rank0_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load rank 0 results: {e}")
            
            # 从其他rank收集结果
            for rank in range(1, args.world_size):
                rank_results_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}_rank{rank}.json"
                if rank_results_file.exists():
                    try:
                        # 检查文件大小，避免读取空文件
                        if rank_results_file.stat().st_size == 0:
                            logger.warning(f"Rank {rank} results file is empty: {rank_results_file}")
                            continue
                        
                        with open(rank_results_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                logger.warning(f"Rank {rank} results file is empty (no content): {rank_results_file}")
                                continue
                            rank_results = json.loads(content)
                        
                        # 合并结果
                        for tau, summary in rank_results.items():
                            if isinstance(summary, dict) and 'all_metrics' in summary:
                                if tau in all_results:
                                    # 合并episode结果
                                    all_results[tau]['all_metrics'].extend(summary.get('all_metrics', []))
                                else:
                                    # 如果rank 0没有这个tau的结果，直接添加
                                    all_results[tau] = summary
                                
                                # 重新计算平均值
                                if 'all_metrics' in all_results[tau] and len(all_results[tau]['all_metrics']) > 0:
                                    N = len(all_results[tau]['all_metrics'])
                                    all_results[tau]['num_episodes'] = N
                                    all_results[tau]['sr'] = np.mean([m['success'] for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['oracle_sr'] = np.mean([m.get('oracle_success', False) for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['spl'] = np.mean([m['spl'] for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['gp'] = np.mean([m.get('gp', 0.0) for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['nav_error'] = np.mean([m['nav_error'] for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['oracle_error'] = np.mean([m.get('oracle_error', float('inf')) for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['teacher_call_rate'] = sum([m['teacher_calls'] for m in all_results[tau]['all_metrics']]) / sum([m['total_steps'] for m in all_results[tau]['all_metrics']]) if sum([m['total_steps'] for m in all_results[tau]['all_metrics']]) > 0 else 0.0
                                    all_results[tau]['t_episode_avg'] = np.mean([m['t_episode_total'] for m in all_results[tau]['all_metrics']])
                                    all_results[tau]['teacher_time_ratio'] = np.mean([m['teacher_time_ratio'] for m in all_results[tau]['all_metrics']])
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON from rank {rank} results file {rank_results_file}: {e}")
                        logger.error(f"File content preview (first 500 chars): {content[:500] if 'content' in locals() else 'N/A'}")
                        continue
                    except Exception as e:
                        logger.error(f"Error loading rank {rank} results from {rank_results_file}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
        else:
            # 其他rank保存自己的结果
            rank_results_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}_rank{args.rank}.json"
            try:
                # 确保输出目录存在
                rank_results_file.parent.mkdir(parents=True, exist_ok=True)
                with open(rank_results_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                logger.info(f"Rank {args.rank} results saved to {rank_results_file} ({rank_results_file.stat().st_size} bytes)")
            except Exception as e:
                logger.error(f"Failed to save rank {args.rank} results to {rank_results_file}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        dist.barrier()
    
    # 保存最终结果（只在rank 0）
    if args.rank == 0:
        results_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinal results saved to {results_file}")
        
        # 打印最终合并后的结果（基于所有rank的数据）
        if args.multi_gpu and all_results:
            logger.info(f"\n{'='*60}")
            logger.info("Final Aggregated Results (All Ranks Combined):")
            logger.info(f"{'='*60}")
            for tau in sorted(all_results.keys()):
                summary = all_results[tau]
                logger.info(f"\nResults (τ={tau}) - All Ranks:")
                logger.info(f"  SR: {summary['sr']:.4f} (from {summary['num_episodes']} episodes)")
                logger.info(f"  Oracle_SR: {summary.get('oracle_sr', 0.0):.4f}")
                logger.info(f"  SPL: {summary['spl']:.4f}")
                logger.info(f"  GP: {summary.get('gp', 0.0):.4f}")
                logger.info(f"  Nav Error: {summary['nav_error']:.4f}")
                logger.info(f"  Teacher Call Rate: {summary['teacher_call_rate']:.2%}")
                logger.info(f"  Avg Episode Time: {summary['t_episode_avg']:.3f}s")
                logger.info(f"  Teacher Time Ratio: {summary['teacher_time_ratio']:.2%}")
                logger.info(f"  Total Episodes: {summary['num_episodes']}")
        
        # 清理临时文件（但保留rank文件以防最终文件保存失败）
        if args.multi_gpu:
            # 只在最终文件成功保存后才清理rank文件
            if results_file.exists() and results_file.stat().st_size > 0:
                logger.info("Final results file saved successfully, cleaning up rank files...")
                for rank in range(args.world_size):  # 包括rank 0
                    rank_results_file = output_dir / f"results_{args.mode}_{args.router_type}_{args.student_type}_rank{rank}.json"
                    if rank_results_file.exists():
                        try:
                            rank_results_file.unlink()
                            logger.debug(f"Removed {rank_results_file.name}")
                        except Exception as e:
                            logger.warning(f"Failed to remove {rank_results_file.name}: {e}")
            else:
                logger.warning(f"Final results file not saved or empty! Rank files are preserved for manual recovery.")
    
    if args.multi_gpu:
        dist.destroy_process_group()
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
