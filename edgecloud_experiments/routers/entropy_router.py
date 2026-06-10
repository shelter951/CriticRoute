"""
Entropy-based router: call Teacher when Student uncertainty is high
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from .base_router import BaseRouter


class EntropyRouter(BaseRouter):
    """
    Simple entropy threshold router.
    Calls Teacher when Student's entropy exceeds a threshold.
    """
    
    def __init__(self, entropy_threshold: float = None):
        """
        Args:
            entropy_threshold: Entropy threshold (if None, will be set dynamically)
        """
        self.entropy_threshold = entropy_threshold
    
    def should_call_teacher(
        self,
        student_logits: torch.Tensor,
        student_action_idx: int,
        features: Dict[str, float],
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[bool, float]:
        """
        Use entropy as the decision signal.
        
        Args:
            threshold: Normalized threshold (0-1), will be mapped to entropy range
        """
        entropy = features.get('entropy', 0.0)
        
        # If entropy_threshold is set, use it directly
        # Otherwise, map threshold (0-1) to a reasonable entropy range (e.g., 0-3)
        if self.entropy_threshold is not None:
            use_threshold = self.entropy_threshold
        else:
            # Map threshold [0, 1] to entropy range [0, 3]
            # threshold=0.5 -> entropy_threshold=1.5
            use_threshold = threshold * 3.0
        
        # Normalize entropy to [0, 1] for confidence score
        # Assuming max entropy ~3.0 for typical navigation tasks
        max_entropy = 3.0
        confidence = min(entropy / max_entropy, 1.0)
        
        should_call = entropy >= use_threshold
        
        return should_call, confidence
    
    def get_name(self) -> str:
        return "EntropyRouter"

