"""
DistillDataset: Dataset for knowledge distillation training
Reads data from JSONL files collected from teacher model
"""
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import defaultdict
from .feature_db import ImageFeaturesDB, create_feature_db
from .mp3d_envs import angle_feature


class DistillDataset(Dataset):
    """
    Dataset for distillation training from teacher model's JSONL logs
    """
    
    def __init__(
        self,
        jsonl_paths: List[str],
        args,
        config: Dict,
        tokenizer,
        feat_db: Dict[str, ImageFeaturesDB],
        original_datasets: Optional[Dict] = None,
        task_filter: Optional[List[str]] = None,
        filter_success_only: bool = False,
        filter_min_spl: float = 0.0,
    ):
        """
        Args:
            jsonl_paths: List of JSONL file paths (one per task)
            args: Training arguments
            config: Dataset config
            tokenizer: Tokenizer from NavQwen3 model
            feat_db: Feature database dict {task: ImageFeaturesDB}
            original_datasets: Optional dict of original datasets for scan lookup
            task_filter: Optional list of tasks to include (e.g., ['CVDN', 'R2R'])
            filter_success_only: If True, only keep successful episodes
            filter_min_spl: Minimum SPL threshold for filtering
        """
        self.args = args
        self.config = config
        self.tokenizer = tokenizer
        self.feat_db = feat_db
        self.original_datasets = original_datasets or {}
        
        # Load and expand samples from JSONL files
        self.samples = []
        self.scan_cache = {}  # Cache for scan lookups
        
        for jsonl_path in jsonl_paths:
            if not Path(jsonl_path).exists():
                print(f"Warning: JSONL file not found: {jsonl_path}")
                continue
                
            task_name = Path(jsonl_path).stem.replace('distill_', '').replace('_train', '').upper()
            
            # Apply task filter
            if task_filter is not None and task_name not in task_filter:
                continue
            
            print(f"Loading distillation data from {jsonl_path} (task: {task_name})...")
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    try:
                        episode = json.loads(line.strip())
                        
                        # Apply filters
                        if filter_success_only and episode.get('success', 0) == 0:
                            continue
                        if episode.get('final_spl', 0.0) < filter_min_spl:
                            continue
                        
                        task = episode.get('task', task_name)
                        episode_id = episode.get('episode_id', f'{task_name}_{line_idx}')
                        success = episode.get('success', 0)
                        final_spl = episode.get('final_spl', 0.0)
                        
                        # Expand steps into individual samples
                        for step in episode.get('steps', []):
                            # Basic validation
                            cand_vpids = step.get('cand_vpids', [])
                            teacher_logits = step.get('teacher_logits', [])
                            teacher_action_idx = step.get('teacher_action_idx', 0)
                            
                            # Skip invalid samples
                            if len(cand_vpids) == 0:
                                continue
                            if len(cand_vpids) != len(teacher_logits):
                                continue
                            if teacher_action_idx < 0 or teacher_action_idx >= len(cand_vpids):
                                continue
                            
                            # Try to get scan from original dataset
                            scan = self._get_scan_from_episode(episode_id, step.get('viewpoint_id_before'))
                            
                            # Get scan from step data (preferred) or fallback to lookup
                            scan = step.get('scan') or self._get_scan_from_episode(episode_id, step.get('viewpoint_id_before'))
                            
                            sample = {
                                'task': task,
                                'episode_id': episode_id,
                                'step_t': step.get('t', 0),
                                'schema_prompt': step.get('schema_prompt', ''),
                                'cand_vpids': cand_vpids,
                                'teacher_action_idx': teacher_action_idx,
                                'teacher_logits': teacher_logits,
                                'viewpoint_id_before': step.get('viewpoint_id_before'),
                                'dist_before': step.get('dist_before', 0.0),
                                'dist_after': step.get('dist_after', 0.0),
                                'success': success,
                                'final_spl': final_spl,
                                'scan': scan,  # From step data or lookup
                            }
                            
                            self.samples.append(sample)
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_idx} in {jsonl_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing line {line_idx} in {jsonl_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.samples)} distillation samples from {len(jsonl_paths)} files")
        
        # Statistics
        task_counts = defaultdict(int)
        for sample in self.samples:
            task_counts[sample['task']] += 1
        print(f"Sample distribution by task: {dict(task_counts)}")
    
    def _get_scan_from_episode(self, episode_id: str, viewpoint_id: Optional[str]) -> Optional[str]:
        """
        Try to get scan ID from episode_id or original datasets
        """
        # Check cache first
        if episode_id in self.scan_cache:
            return self.scan_cache[episode_id]
        
        scan = None
        
        # Try to extract from original datasets
        for task_name, dataset in self.original_datasets.items():
            # Try to match episode_id with instr_id
            if hasattr(dataset, 'gt_trajs'):
                for instr_id, item in dataset.gt_trajs.items():
                    if instr_id in episode_id or episode_id in instr_id:
                        scan = item.get('scan')
                        if scan:
                            self.scan_cache[episode_id] = scan
                            return scan
        
        # Try to infer from viewpoint_id pattern (fallback)
        # Viewpoint IDs in MP3D are typically UUIDs, but we can't reliably infer scan from them
        # So we return None and handle it in __getitem__
        
        self.scan_cache[episode_id] = scan
        return scan
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a training sample
        Returns:
            dict with:
                - input_ids: Tokenized schema prompt
                - attention_mask: Attention mask
                - cand_feats: Visual features for candidates [num_cands, feat_dim]
                - label: Teacher action index
                - cand_vpids: Candidate viewpoint IDs
                - dist_before/dist_after: Distance info (for weighting)
        """
        sample = self.samples[idx]
        
        # Tokenize schema prompt
        schema_prompt = sample['schema_prompt']
        tokenized = self.tokenizer(
            schema_prompt,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        input_ids = tokenized['input_ids'][0]  # [seq_len]
        attention_mask = tokenized['attention_mask'][0]  # [seq_len]
        
        # Load visual features for candidates
        cand_vpids = sample['cand_vpids']
        scan = sample['scan']
        
        # Determine which feature DB to use based on task
        task = sample['task'].upper()
        if task in ['CVDN', 'R2R', 'REVERIE', 'SOON']:
            # Use MP3D feature DB
            feat_db_key = 'mp3d' if 'mp3d' in self.feat_db else list(self.feat_db.keys())[0]
        else:
            feat_db_key = list(self.feat_db.keys())[0]
        
        feat_db = self.feat_db[feat_db_key]
        
        # Load features for each candidate
        cand_feats_list = []
        for vpid in cand_vpids:
            if vpid is None:  # Stop action
                # Create zero feature for stop
                feat = np.zeros((36, self.args.image_feat_size), dtype=np.float32)
            else:
                if scan:
                    try:
                        feat = feat_db.get_image_feature(scan, vpid)
                    except:
                        # Fallback: try without scan (some feature DBs support this)
                        try:
                            feat = feat_db.get_image_feature(vpid)
                        except:
                            # Last resort: zero feature
                            feat = np.zeros((36, self.args.image_feat_size), dtype=np.float32)
                else:
                    # No scan available, try direct lookup
                    try:
                        feat = feat_db.get_image_feature(vpid)
                    except:
                        feat = np.zeros((36, self.args.image_feat_size), dtype=np.float32)
            
            # Ensure correct shape: [36, feat_dim] for panorama
            if len(feat.shape) == 1:
                # Single feature, expand to 36 views
                feat = np.tile(feat, (36, 1))
            elif feat.shape[0] != 36:
                # Resize or pad to 36 views
                if feat.shape[0] > 36:
                    feat = feat[:36]
                else:
                    pad_shape = (36 - feat.shape[0], feat.shape[1])
                    feat = np.concatenate([feat, np.zeros(pad_shape, dtype=feat.dtype)], axis=0)
            
            cand_feats_list.append(feat)
        
        # Stack candidate features: [num_cands, 36, feat_dim]
        cand_feats = np.stack(cand_feats_list, axis=0)
        cand_feats = torch.from_numpy(cand_feats).float()
        
        # Label is teacher action index
        label = sample['teacher_action_idx']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cand_feats': cand_feats,  # [num_cands, 36, feat_dim]
            'label': label,
            'cand_vpids': cand_vpids,
            'dist_before': sample['dist_before'],
            'dist_after': sample['dist_after'],
            'task': sample['task'],
            'episode_id': sample['episode_id'],
            'scan': scan if scan else '',  # Return empty string if None
        }
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader
        """
        # Get max sequence length
        max_seq_len = max([item['input_ids'].shape[0] for item in batch])
        max_num_cands = max([item['cand_feats'].shape[0] for item in batch])
        
        # Pad input_ids and attention_mask
        input_ids_list = []
        attention_mask_list = []
        for item in batch:
            seq_len = item['input_ids'].shape[0]
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                input_ids = torch.cat([
                    item['input_ids'],
                    torch.zeros(pad_len, dtype=item['input_ids'].dtype)
                ])
                attention_mask = torch.cat([
                    item['attention_mask'],
                    torch.zeros(pad_len, dtype=item['attention_mask'].dtype)
                ])
            else:
                input_ids = item['input_ids'][:max_seq_len]
                attention_mask = item['attention_mask'][:max_seq_len]
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        input_ids = torch.stack(input_ids_list)  # [B, max_seq_len]
        attention_mask = torch.stack(attention_mask_list)  # [B, max_seq_len]
        
        # Pad candidate features
        cand_feats_list = []
        cand_masks_list = []
        for item in batch:
            num_cands = item['cand_feats'].shape[0]
            pad_len = max_num_cands - num_cands
            if pad_len > 0:
                # Pad with zeros
                pad_shape = (pad_len, item['cand_feats'].shape[1], item['cand_feats'].shape[2])
                padded = torch.cat([
                    item['cand_feats'],
                    torch.zeros(pad_shape, dtype=item['cand_feats'].dtype)
                ], dim=0)
                # Create mask: 1 for valid, 0 for padded
                mask = torch.cat([
                    torch.ones(num_cands, dtype=torch.bool),
                    torch.zeros(pad_len, dtype=torch.bool)
                ])
            else:
                padded = item['cand_feats'][:max_num_cands]
                mask = torch.ones(num_cands, dtype=torch.bool)
            cand_feats_list.append(padded)
            cand_masks_list.append(mask)
        
        cand_feats = torch.stack(cand_feats_list)  # [B, max_num_cands, 36, feat_dim]
        cand_masks = torch.stack(cand_masks_list)  # [B, max_num_cands]
        
        # Stack labels
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        
        # Other metadata
        dist_before = torch.tensor([item['dist_before'] for item in batch], dtype=torch.float32)
        dist_after = torch.tensor([item['dist_after'] for item in batch], dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cand_feats': cand_feats,
            'cand_masks': cand_masks,
            'labels': labels,
            'dist_before': dist_before,
            'dist_after': dist_after,
            'cand_vpids': [item['cand_vpids'] for item in batch],
            'tasks': [item['task'] for item in batch],
            'episode_ids': [item['episode_id'] for item in batch],
            'scans': [item['scan'] for item in batch],
        }

