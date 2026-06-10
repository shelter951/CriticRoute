"""
统一的导航输入构造模块
确保Teacher和Student使用相同的输入构造逻辑（除了hist_vis维度不同）
"""
import torch
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def build_nav_inputs_from_obs(
    agent,
    obs: List[Dict],
    gmaps: List,
    pano_embeds: torch.Tensor,
    pano_masks: torch.Tensor,
    pano_inputs: Dict,
    instructions: List[str],
    history: List[List[str]],
    hist_vis: List[List[torch.Tensor]],
    data_type: str,
    model_cls_token: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Build navigation inputs from observations.
    
    This function ensures Teacher and Student use the same input construction logic,
    except for hist_vis dimension (Teacher: 4096, Student: 2048).
    
    Args:
        agent: Agent instance (MP3DAgent or similar)
        obs: List of observations
        gmaps: List of GraphMap instances
        pano_embeds: Panorama embeddings [B, N, D]
        pano_masks: Panorama masks [B, N]
        pano_inputs: Dictionary containing cand_vpids, view_lens, nav_types
        instructions: List of instruction strings
        history: List of history token lists
        hist_vis: List of historical visual embeddings (each is list of tensors)
        data_type: Data type string (e.g., 'cvdn')
        model_cls_token: CLS token string for the model
        device: Target device
    
    Returns:
        Dictionary of navigation inputs ready for model forward
    """
    # Step 1: Build graph map variables (same for Teacher and Student)
    nav_inputs = agent.nav_gmap_variable(obs, gmaps)
    
    # Step 2: Build viewpoint variables (same for Teacher and Student)
    nav_inputs.update(
        agent.nav_vp_variable(
            obs, gmaps, pano_embeds, pano_masks,
            pano_inputs['cand_vpids'],
            pano_inputs['view_lens'],
            pano_inputs['nav_types'],
        )
    )
    
    # Step 3: Add common fields (same for Teacher and Student)
    nav_inputs.update({
        'view_lens': pano_inputs['view_lens'],
        'instruction': instructions,
        'history': history,
        'hist_vis': hist_vis,  # Dimension differs: Teacher=4096, Student=2048
        'data_type': data_type,
    })
    
    # Step 4: Prepare prompts (same schema, but tokenization may differ)
    nav_inputs["prompts"] = agent.prepare_prompts(
        "navigation",
        nav_inputs,
        cls_token=model_cls_token
    )
    
    # Step 5: Move all tensors to target device
    nav_inputs = move_batch_to_device(nav_inputs, device)
    
    return nav_inputs


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensors in batch to the specified device.
    Handles nested structures like list[Tensor] (e.g., hist_vis).
    """
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], torch.Tensor):
                # List of tensors: move each
                new_batch[k] = [t.to(device) for t in v]
            elif isinstance(v[0], list) and len(v[0]) > 0 and isinstance(v[0][0], torch.Tensor):
                # Nested list of tensors (e.g., hist_vis)
                new_batch[k] = [
                    [t.to(device) if isinstance(t, torch.Tensor) else t for t in sublist]
                    for sublist in v
                ]
            else:
                new_batch[k] = v
        elif isinstance(v, dict):
            new_batch[k] = move_batch_to_device(v, device)
        else:
            new_batch[k] = v
    return new_batch


def verify_nav_inputs_consistency(
    teacher_inputs: Dict[str, Any],
    student_inputs: Dict[str, Any],
    verbose: bool = False
) -> bool:
    """
    Verify that Teacher and Student inputs are consistent (except for hist_vis dimension).
    
    Args:
        teacher_inputs: Teacher model navigation inputs
        student_inputs: Student model navigation inputs
        verbose: Whether to print detailed comparison
    
    Returns:
        True if inputs are consistent (except expected differences)
    """
    # Check that prompts are the same (text content)
    if 'prompts' in teacher_inputs and 'prompts' in student_inputs:
        teacher_prompts = teacher_inputs['prompts']
        student_prompts = student_inputs['prompts']
        if teacher_prompts != student_prompts:
            if verbose:
                print("⚠️  Warning: Prompts differ (may be due to tokenization)")
                print(f"  Teacher: {teacher_prompts[:100]}...")
                print(f"  Student: {student_prompts[:100]}...")
    
    # Check that gmap_vpids are the same
    if 'gmap_vpids' in teacher_inputs and 'gmap_vpids' in student_inputs:
        teacher_vpids = teacher_inputs['gmap_vpids']
        student_vpids = student_inputs['gmap_vpids']
        if teacher_vpids != student_vpids:
            if verbose:
                print("❌ Error: gmap_vpids differ!")
            return False
    
    # Check that instructions are the same
    if 'instruction' in teacher_inputs and 'instruction' in student_inputs:
        if teacher_inputs['instruction'] != student_inputs['instruction']:
            if verbose:
                print("❌ Error: Instructions differ!")
            return False
    
    # hist_vis dimension is expected to differ (Teacher: 4096, Student: 2048)
    # This is acceptable
    
    if verbose:
        print("✅ Navigation inputs are consistent (except hist_vis dimension)")
    
    return True





