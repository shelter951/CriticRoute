"""
Off-course router: MLP-based router trained on off-course step labels
"""
import torch
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from router_code.models.router_mlp import create_router_mlp
from edgecloud_experiments.utils.router_features import get_feature_vector
from .base_router import BaseRouter


class OffCourseRouter(BaseRouter):
    """
    MLP-based router trained to identify off-course steps.
    Uses the same feature extraction as training time.
    """
    
    def __init__(self, checkpoint_path: str, input_dim: int = None, hidden_dim: int = 128, device: torch.device = None):
        """
        Args:
            checkpoint_path: Path to trained router checkpoint
            input_dim: Input feature dimension (if None, will be inferred from checkpoint)
            hidden_dim: Hidden dimension (must match training)
            device: Device to load model on
        """
        self.device = device if device is not None else torch.device('cuda:0')
        self.hidden_dim = hidden_dim
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Infer input_dim from checkpoint if not provided
        if input_dim is None:
            if 'model_state_dict' in checkpoint:
                # Get the first layer weight shape: [hidden_dim, input_dim]
                first_layer_key = 'net.0.weight' if 'net.0.weight' in checkpoint['model_state_dict'] else None
                if first_layer_key:
                    weight_shape = checkpoint['model_state_dict'][first_layer_key].shape
                    input_dim = weight_shape[1]  # [hidden_dim, input_dim]
                    logger.info(f"✅ 从checkpoint推断 input_dim = {input_dim}")
                else:
                    # Fallback: use feature_keys length if available
                    if 'feature_keys' in checkpoint:
                        input_dim = len(checkpoint['feature_keys'])
                        logger.info(f"✅ 从checkpoint的feature_keys推断 input_dim = {input_dim}")
                    else:
                        # Default: 13 (actual feature count)
                        input_dim = 13
                        logger.warning(f"⚠️  无法从checkpoint推断，使用默认值 input_dim = 13")
            else:
                # Default: 13 (actual feature count)
                input_dim = 13
                logger.warning(f"⚠️  无法从checkpoint推断，使用默认值 input_dim = 13")
        
        self.input_dim = input_dim
        
        # Create model with correct input_dim
        self.model = create_router_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.1,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get feature keys from checkpoint if available, otherwise use default
        # ⚠️ 重要：必须与训练时使用的顺序完全一致！
        # 训练时使用 sorted(keys)，即字典序排序
        if 'feature_keys' in checkpoint:
            self.feature_keys = checkpoint['feature_keys']
            logger.info(f"✅ 从checkpoint加载feature_keys，共{len(self.feature_keys)}个")
        else:
            # ⚠️ 默认feature_keys必须与训练时一致（sorted字典序）
            # 训练时：router_code/datasets/router_dataset.py 使用 sorted(self.samples[0]['features'].keys())
            # 这个顺序是字典序，必须完全匹配！
            self.feature_keys = [
                'dist_before',      # 0
                'dist_change',      # 1
                'entropy',          # 2
                'is_stop_top1',     # 3
                'margin',           # 4
                'num_cands',        # 5
                'step_ratio',       # 6
                'top1_logit',       # 7
                'top1_prob',        # 8
                'top2_logit',       # 9
                'top2_prob',        # 10
                'top3_logit',       # 11
                'top3_prob',        # 12
            ]
            logger.warning(f"⚠️  checkpoint中没有feature_keys，使用默认顺序（字典序）")
    
    def should_call_teacher(
        self,
        student_logits,
        student_action_idx: int,
        features: Dict[str, float],
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[bool, float]:
        """
        Use trained MLP to predict whether to call Teacher.
        
        Args:
            threshold: Probability threshold (0-1)
        """
        # Convert features to vector
        feat_vec = get_feature_vector(features, self.feature_keys)
        
        # Ensure correct dimension
        if len(feat_vec) != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {len(feat_vec)}, expected {self.input_dim}. "
                f"Feature keys: {list(features.keys())}"
            )
        
        # Convert to tensor and add batch dimension
        feat_tensor = torch.tensor(feat_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            logit = self.model(feat_tensor)
            confidence = torch.sigmoid(logit).item()
        
        should_call = confidence >= threshold
        
        return should_call, confidence
    
    def get_name(self) -> str:
        return "OffCourseRouter"

