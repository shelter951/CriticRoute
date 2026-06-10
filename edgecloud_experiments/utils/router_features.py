"""
统一的Router特征提取模块
确保训练时和运行时使用完全相同的特征提取逻辑，避免特征分布偏移
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union


def to_numpy(t: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """
    Convert tensor to numpy array, handling bfloat16/float16 by converting to float32 first.
    
    Args:
        t: torch.Tensor or numpy array or list
        
    Returns:
        numpy array
    """
    if isinstance(t, torch.Tensor):
        # Convert bfloat16/float16 to float32 before converting to numpy
        if t.dtype in (torch.bfloat16, torch.float16):
            t = t.to(torch.float32)
        return t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        return t
    else:
        return np.array(t)


def extract_router_features(
    student_logits: torch.Tensor,
    num_cands: int,
    step_id: int,
    max_steps: int,
    dist_before: float,
    dist_after_student: float,
    student_action_idx: int,
) -> Dict[str, float]:
    """
    Extract router input features from Student model output.
    
    This function is used BOTH in:
    1. Data collection (router_code/collect_router_data.py)
    2. Runtime evaluation (edgecloud_experiments/eval_edgecloud.py)
    
    To ensure feature consistency, DO NOT modify this function without updating both places.
    
    Args:
        student_logits: Student model logits [num_cands] or [1, num_cands]
        num_cands: Number of candidates (including stop)
        step_id: Current step index (0-indexed)
        max_steps: Maximum steps in episode
        dist_before: Distance to goal before action
        dist_after_student: Distance to goal after Student action
        student_action_idx: Student's selected action index (0=stop, 1..N-1=candidates)
    
    Returns:
        Dictionary of feature values (all float)
    """
    # Handle batch dimension: if [1, num_cands], squeeze to [num_cands]
    if student_logits.dim() == 2:
        if student_logits.shape[0] == 1:
            student_logits = student_logits.squeeze(0)
        else:
            raise ValueError(f"Expected logits shape [num_cands] or [1, num_cands], got {student_logits.shape}")
    
    # Convert to numpy for easier manipulation (handle bfloat16/float16)
    logits_np = to_numpy(student_logits)
    
    # Softmax probabilities (convert back to tensor for F.softmax, then to numpy)
    logits_tensor = torch.from_numpy(logits_np)
    probs = F.softmax(logits_tensor, dim=-1)
    probs = to_numpy(probs)
    
    # Top-1 and Top-2 probabilities
    top1_idx = np.argmax(probs)
    top1_prob = float(probs[top1_idx])
    
    # Get top-2
    if len(probs) > 1:
        top2_idx = np.argsort(probs)[-2]
        top2_prob = float(probs[top2_idx])
    else:
        top2_idx = top1_idx
        top2_prob = top1_prob
    
    # Entropy (uncertainty measure)
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    
    # Margin (confidence measure: top1 - top2 logit difference)
    margin = float(logits_np[top1_idx] - logits_np[top2_idx]) if len(logits_np) > 1 else 0.0
    
    # Distance change
    dist_change = dist_after_student - dist_before
    
    # Step ratio
    step_ratio = step_id / max_steps if max_steps > 0 else 0.0
    
    # Is stop action selected?
    is_stop_top1 = float(1.0 if student_action_idx == 0 else 0.0)
    
    # Additional features: top-K logits and probs
    topk = min(5, len(logits_np))
    topk_indices = np.argsort(logits_np)[-topk:][::-1]
    topk_logits = logits_np[topk_indices]
    topk_probs = probs[topk_indices]
    
    features = {
        # Uncertainty features
        "entropy": entropy,
        "margin": margin,
        "top1_prob": top1_prob,
        "top2_prob": top2_prob,
        
        # Context features
        "num_cands": float(num_cands),
        "step_ratio": step_ratio,
        "is_stop_top1": is_stop_top1,
        
        # Navigation progress
        "dist_before": dist_before,
        "dist_change": dist_change,
        
        # Top-K logits (for richer representation)
        "top1_logit": float(topk_logits[0]) if topk > 0 else 0.0,
        "top2_logit": float(topk_logits[1]) if topk > 1 else 0.0,
        "top3_logit": float(topk_logits[2]) if topk > 2 else 0.0,
        
        # Top-K probs (top1_prob and top2_prob already included above, add top3)
        "top3_prob": float(topk_probs[2]) if topk > 2 else 0.0,
    }
    
    return features


def get_feature_vector(features: Dict[str, float], feature_keys: list = None) -> np.ndarray:
    """
    Convert feature dictionary to a fixed-order vector.
    
    Args:
        features: Dictionary of feature values
        feature_keys: List of feature keys in desired order (if None, use sorted keys)
    
    Returns:
        numpy array of feature values
    """
    if feature_keys is None:
        feature_keys = sorted(features.keys())
    
    return np.array([features.get(key, 0.0) for key in feature_keys], dtype=np.float32)


def check_feature_statistics(features_list: list, name: str = "features"):
    """
    Print feature statistics for sanity check.
    Used to verify feature distribution consistency between training and deployment.
    
    Args:
        features_list: List of feature dictionaries
        name: Name for logging
    """
    if len(features_list) == 0:
        print(f"[{name}] No features to analyze")
        return
    
    # Get all feature keys
    all_keys = set()
    for feat in features_list:
        all_keys.update(feat.keys())
    all_keys = sorted(all_keys)
    
    print(f"\n[{name}] Feature Statistics (N={len(features_list)}):")
    print("-" * 60)
    for key in all_keys:
        values = [feat.get(key, 0.0) for feat in features_list]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"  {key:20s}: mean={mean_val:8.4f}, std={std_val:8.4f}, "
              f"range=[{min_val:8.4f}, {max_val:8.4f}]")
    print("-" * 60)





