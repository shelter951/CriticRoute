"""
PPO Router adapter for edgecloud evaluation.

Uses a trained DeferralPolicy (PolicyNet) to decide whether to call Teacher.
Input: cls_hidden from Student model's forward pass (2048-dim for Qwen3-1.7B)
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

from .base_router import BaseRouter

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ppo_router.deferral_head import PolicyNet


class PPORouter(BaseRouter):
    """PPO-trained deferral router using Student cls_hidden."""

    def __init__(self, checkpoint_path: str, device: torch.device = None, threshold: float = 0.5):
        self.device = device if device is not None else torch.device("cuda:0")
        self.default_threshold = threshold

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "input_dim" in ckpt:
            input_dim = ckpt["input_dim"]
        elif "policy_state_dict" in ckpt and "net.0.weight" in ckpt["policy_state_dict"]:
            input_dim = ckpt["policy_state_dict"]["net.0.weight"].shape[1]
        else:
            input_dim = 2048

        logger.info(f"PPORouter: input_dim={input_dim}")

        self.policy_net = PolicyNet(input_dim=input_dim, hidden_dim=128, dropout=0.1)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.eval()

        logger.info(f"PPORouter loaded from {checkpoint_path}")
        logger.info(f"  Iteration: {ckpt.get('iteration', 'unknown')}")
        logger.info(f"  Global step: {ckpt.get('global_step', 'unknown')}")

    def should_call_teacher(
        self,
        student_logits,
        student_action_idx: int,
        features: Dict[str, float],
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[bool, float]:
        cls_hidden = kwargs.get("cls_hidden", None)

        if cls_hidden is not None:
            if isinstance(cls_hidden, torch.Tensor):
                if cls_hidden.dim() == 2:
                    cls_hidden = cls_hidden[0]
                cls_hidden = cls_hidden.unsqueeze(0)
            else:
                cls_hidden = torch.tensor(cls_hidden, dtype=torch.float32, device=self.device).unsqueeze(0)

            cls_hidden = cls_hidden.to(self.device)

            with torch.no_grad():
                prob = self.policy_net.forward(cls_hidden).item()

            return prob >= threshold, prob

        logger.warning("PPORouter: cls_hidden not available, using features-based fallback")
        entropy = features.get("entropy", 0.0)
        top1_prob = features.get("top1_prob", 1.0)
        dist_change = features.get("dist_change", 0.0)
        confidence = max(0.0, 1.0 - top1_prob) * (1.0 if dist_change > 0 else 0.0)
        return confidence >= threshold, confidence

    def get_name(self) -> str:
        return "PPORouter"
