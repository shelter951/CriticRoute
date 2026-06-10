"""Heterogeneous edge-cloud routing for R2R.

This package intentionally does not reuse the older homogeneous NaviLLM/Qwen3
router code.  The edge model here is Qwen2.5-VL-R2R, while the cloud side is
represented by NaviLLM teacher trajectories or a compatible online teacher.
"""

