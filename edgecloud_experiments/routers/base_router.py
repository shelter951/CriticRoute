"""
Base router interface
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch


class BaseRouter(ABC):
    """Base class for all router strategies"""
    
    @abstractmethod
    def should_call_teacher(
        self,
        student_logits: torch.Tensor,
        student_action_idx: int,
        features: Dict[str, float],
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[bool, float]:
        """
        Decide whether to call Teacher model.
        
        Args:
            student_logits: Student model logits [num_cands]
            student_action_idx: Student's selected action index
            features: Extracted router features
            threshold: Decision threshold
            **kwargs: Additional context (e.g., teacher_action_idx for divergence router)
        
        Returns:
            (should_call: bool, confidence: float)
            confidence is the probability/score for calling teacher
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get router name for logging"""
        pass

