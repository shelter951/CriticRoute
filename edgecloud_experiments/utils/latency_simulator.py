"""
网络延迟模拟模块
模拟Teacher模型在云端调用时的网络延迟
"""
import time
import random
from typing import Literal


LatencyMode = Literal['none', 'moderate', 'high', 'fixed']


def simulate_cloud_latency(
    mode: LatencyMode = 'fixed',
    fixed_ms: float = 400.0,
    moderate_range: tuple = (150.0, 300.0),
    high_range: tuple = (500.0, 800.0)
) -> None:
    """
    Simulate network latency for cloud Teacher model calls.
    
    Args:
        mode: Latency mode
            - 'none': No latency (local Teacher, upper bound)
            - 'moderate': Random latency in moderate_range (ms)
            - 'high': Random latency in high_range (ms)
            - 'fixed': Fixed latency (default: 400ms as specified)
        fixed_ms: Fixed latency in milliseconds (used when mode='fixed')
        moderate_range: (min_ms, max_ms) for moderate latency
        high_range: (min_ms, max_ms) for high latency
    """
    if mode == 'none':
        return
    elif mode == 'fixed':
        time.sleep(fixed_ms / 1000.0)
    elif mode == 'moderate':
        latency_ms = random.uniform(moderate_range[0], moderate_range[1])
        time.sleep(latency_ms / 1000.0)
    elif mode == 'high':
        latency_ms = random.uniform(high_range[0], high_range[1])
        time.sleep(latency_ms / 1000.0)
    else:
        raise ValueError(f"Unknown latency mode: {mode}")


def get_latency_description(mode: LatencyMode, fixed_ms: float = 400.0) -> str:
    """Get human-readable description of latency mode"""
    if mode == 'none':
        return "No latency (local)"
    elif mode == 'fixed':
        return f"Fixed {fixed_ms}ms"
    elif mode == 'moderate':
        return "Moderate (150-300ms)"
    elif mode == 'high':
        return "High (500-800ms)"
    else:
        return f"Unknown mode: {mode}"





