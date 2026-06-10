"""
Divergence-based router: call Teacher when Student and Teacher actions differ
"""
from typing import Dict, Tuple
from .base_router import BaseRouter


class DivergenceRouter(BaseRouter):
    """
    Simple divergence router.
    Calls Teacher when Student and Teacher actions are different.
    Note: This router requires teacher_action_idx to be provided in kwargs.
    """
    
    def should_call_teacher(
        self,
        student_logits,
        student_action_idx: int,
        features: Dict[str, float],
        threshold: float = 0.5,
        **kwargs
    ) -> Tuple[bool, float]:
        """
        Use action divergence as the decision signal.
        
        Args:
            teacher_action_idx: Must be provided in kwargs
        """
        teacher_action_idx = kwargs.get('teacher_action_idx', None)
        
        if teacher_action_idx is None:
            # If teacher action is not available, fall back to always calling teacher
            # (This shouldn't happen in normal operation)
            return True, 1.0
        
        # Check if actions differ
        actions_differ = (student_action_idx != teacher_action_idx)
        
        # Confidence is 1.0 if actions differ, 0.0 if they match
        confidence = 1.0 if actions_differ else 0.0
        
        # For this router, threshold doesn't matter much (binary decision)
        # But we can use it to control: if threshold > 0.5, only call when actions differ
        should_call = actions_differ and (threshold <= 0.5 or confidence >= threshold)
        
        return should_call, confidence
    
    def get_name(self) -> str:
        return "DivergenceRouter"

