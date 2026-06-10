"""
NavQwen3: Navigation model based on Qwen3-1.7B
This is the student model for knowledge distillation
"""
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import logging
from .ops import pad_tensors_wgrad, gen_seq_masks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
from .image_embedding import ImageEmbeddings
from .modified_qwen import ModifiedQwenForCausalLM
from typing import Dict, List, Any

logging.set_verbosity_error()


def init_vis_config(args, config):
    """Initialize visual config for NavQwen3"""
    cfg_name = 'bert-large-uncased'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    vis_config.num_pano_layers = config.num_pano_layers
    vis_config.precision = args.precision
    vis_config.pretrained_model_name_or_path = args.pretrained_model_name_or_path
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.type_vocab_size = 3
    return vis_config


class NavQwen3(nn.Module):
    """
    Navigation model based on Qwen3-1.7B
    Student model for knowledge distillation
    """
    
    def __init__(self, args, logger, model_config):
        super().__init__()
        self.args = args
        config = init_vis_config(args, model_config)
        self.config = config

        self.no_loc_fts = args.no_loc_fts

        # Load Qwen3-1.7B model
        if args.resume_from_checkpoint is not None or args.from_scratch:
            logger.info("Initialize NavQwen3 from config.")
            model_config = AutoConfig.from_pretrained(
                config.pretrained_model_name_or_path, 
                trust_remote_code=True
            )
            self.lang_model = ModifiedQwenForCausalLM(model_config, config)
        else:
            # Try loading from local path first
            model_path = Path(config.pretrained_model_name_or_path)
            if model_path.exists() and (model_path / "config.json").exists():
                logger.info(f"Loading Qwen3 from local path: {model_path}")
                model_config = AutoConfig.from_pretrained(
                    str(model_path), 
                    trust_remote_code=True
                )
                self.lang_model = ModifiedQwenForCausalLM.from_pretrained(
                    str(model_path), 
                    config,
                    trust_remote_code=True
                )
            else:
                # Load from HuggingFace
                logger.info(f"Loading Qwen3 from HuggingFace: {config.pretrained_model_name_or_path}")
                model_config = AutoConfig.from_pretrained(
                    config.pretrained_model_name_or_path, 
                    trust_remote_code=True
                )
                self.lang_model = ModifiedQwenForCausalLM(model_config, config)
                # Load weights from HuggingFace
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.pretrained_model_name_or_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if 'fp16' in args.precision else torch.bfloat16 if 'bf16' in args.precision else torch.float32
                )
                self.lang_model.model.load_state_dict(base_model.model.state_dict(), strict=False)
                self.lang_model.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=False)
                del base_model

        # Initialize tokenizer
        self.lang_model.init_tokenizer(config.pretrained_model_name_or_path)

        # Freeze LLM if needed
        if self.args.freeze_llama:
            for name, param in self.lang_model.model.named_parameters():
                param.requires_grad = False
            if self.args.tune_token_emb:
                for name, param in self.lang_model.get_input_embeddings().named_parameters():
                    param.requires_grad = True

        self.hidden_size = self.lang_model.hidden_size
        self.model_type = self.lang_model.model_type

        # Panorama Encoding
        config.output_size = self.hidden_size  # Align with Qwen's hidden size
        self.img_embeddings = ImageEmbeddings(config, use_obj=args.enable_og, fuse_obj=args.fuse_obj)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.hidden_size)

        # global encoding, sum/nav-task
        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(config.max_action_steps, self.hidden_size)

        # local encoding, nav/3dqa/sum-task
        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size * 2 + 6, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        # objgrounding-task
        self.obj_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )

        if self.config.obj_feat_size > 0:
            self.og_head = nn.Sequential(
                nn.Linear(self.hidden_size, 100)
            ).to(self.lang_model.model_type)

        # Classification head for navigation
        self.out_head = nn.Sequential(
            nn.Linear(self.hidden_size, 100)
        ).to(self.lang_model.model_type)

        self.instruction = None
        self.history = None
        self.hist_vis = None

        self.drop_env = nn.Dropout(p=args.feat_dropout)

        logger.info(f"NavQwen3 model type: {self.model_type}")
        logger.info(f"NavQwen3 hidden size: {self.hidden_size}")
        
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        logger.info(f"Trainable params count: {len(trainable_params)}")

    def forward(self, mode: str, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Main forward method"""
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            return self.img_embeddings.forward_panorama_per_step(
                batch['view_img_fts'],
                batch['view_lens'],
                None if self.no_loc_fts else batch['loc_fts'],
                batch['nav_types'],
                batch['obj_img_fts'],
                batch['obj_lens'],
                batch['obj_loc_fts'],
            )

        elif mode == 'navigation':
            return self.forward_navigation(mode, batch, **kwargs)

        else:
            raise NotImplementedError(f'NavQwen3 does not support mode: {mode}')

    def forward_navigation(
        self, 
        mode, 
        batch: Dict[str, Any], 
        training: bool=True, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for navigation task
        This is the main method for distillation training
        """
        data_type = batch['data_type']
        vp_img_embeds = batch['vp_img_embeds']
        batch_size = vp_img_embeds.size(0)
        
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, \
            gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids \
            = batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], \
            batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],

        # global branch [B, Nums, D]
        gmap_embeds = torch.zeros_like(gmap_img_embeds)
        for b_ix in range(len(data_type)):
            gmap_embeds[b_ix:b_ix + 1] = gmap_img_embeds[b_ix:b_ix + 1] + \
                                            self.gmap_step_embeddings(gmap_step_ids[b_ix:b_ix + 1]) + \
                                            self.gmap_pos_embeddings(gmap_pos_fts[b_ix:b_ix + 1])

        # local branch
        vp_img_embeds, vp_pos_fts, vp_nav_masks, vp_cand_vpids = \
            batch['vp_img_embeds'], batch['vp_pos_fts'], batch['vp_nav_masks'], batch['vp_cand_vpids']

        pano_masks = batch['pano_masks']

        vp_embeds = torch.zeros_like(vp_img_embeds)
        for b_ix in range(len(data_type)):
            vp_embeds[b_ix:b_ix + 1] = vp_img_embeds[b_ix:b_ix + 1] \
                                        + self.vp_pos_embeddings(vp_pos_fts[b_ix:b_ix + 1])

        # fuse embeds
        gmap_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        gmap_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)
        cand_token_type_ids = torch.zeros((gmap_embeds.shape[0], gmap_embeds.shape[1])).int().to(gmap_embeds.device)

        local_vp_embeds = vp_embeds
        local_vp_embeds.masked_fill_(pano_masks.logical_not().unsqueeze(-1), 0.)

        fuse_embeds = torch.clone(gmap_embeds)

        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                if j > 0:
                    if cand_vpid in visited_nodes:
                        pass  # Skip visited nodes
                    else:
                        tmp[cand_vpid] = local_vp_embeds[i, j]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fuse_embeds[i, j] += tmp[vp]
                    else:
                        cand_token_type_ids[i, j] = 1

        fuse_embeds += self.token_type_embeddings(cand_token_type_ids).to(fuse_embeds.device)
        fuse_embeds.masked_fill_(gmap_visited_masks.unsqueeze(-1), 0.)
        fuse_embeds.masked_fill_(gmap_masks.logical_not().unsqueeze(-1), 0.)

        cand_masks = torch.clone(gmap_masks & gmap_visited_masks.logical_not())
        cand_nums = cand_masks.sum(dim=-1)
        
        instruction = batch['instruction']
        history = batch['history']
        hist_vis = batch['hist_vis']
        hist_vis_input = []
        for vis in hist_vis:
            hist_vis_input.extend(vis)
        if hist_vis_input != []:
            hist_vis_input = torch.stack(hist_vis_input, dim=0)
        else:
            hist_vis_input = None

        hist_nums = [len(his) for his in history]

        text_input = self.lang_model.tokenize(batch["prompts"]).to(fuse_embeds.device)

        # Prepare candidate embeddings
        cand_embeds = []
        inv_perms = []
        for bn in range(batch_size):
            cand_embed = fuse_embeds[bn][cand_masks[bn]][1:]  # Remove stop
            rand_perm = torch.randperm(cand_embed.shape[0])
            inv_perm = torch.arange(cand_embed.shape[0])
            inv_perm[rand_perm] = torch.arange(cand_embed.shape[0])
            inv_perms.append(inv_perm)
            cand_embeds.append(cand_embed[rand_perm])
        cand_embeds = torch.cat(cand_embeds, dim=0)

        # Forward through language model
        output = self.lang_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            cand_vis=cand_embeds,
            hist_vis=hist_vis_input,
        )
        
        hidden_states = output.hidden_states

        # Extract logits at <cls_1> position
        cls_logits = []
        for bn in range(batch_size):
            cls_pos = (text_input['input_ids'][bn] == self.lang_model.cls_token_id[0]).nonzero(as_tuple=True)[0]
            if len(cls_pos) > 0:
                cls_hidden = hidden_states[bn, cls_pos[0]]
                cls_logits.append(self.out_head(cls_hidden))
            else:
                # Fallback: use last token
                cls_logits.append(self.out_head(hidden_states[bn, -1]))

        fuse_logits = torch.stack(cls_logits, dim=0)

        # Apply inverse permutation and add stop
        final_logits = []
        for bn in range(batch_size):
            cand_num = cand_nums[bn].item()
            if cand_num > 0:
                perm_logits = fuse_logits[bn, :cand_num-1][inv_perms[bn]]
                # Add stop logit (initialize with small value)
                stop_logit = torch.zeros(1, device=perm_logits.device, dtype=perm_logits.dtype)
                final_logits.append(torch.cat([stop_logit, perm_logits], dim=0))
            else:
                final_logits.append(torch.zeros(1, device=fuse_logits.device, dtype=fuse_logits.dtype))

        # Pad to max length
        max_cand_num = max([logits.shape[0] for logits in final_logits])
        padded_logits = []
        for logits in final_logits:
            pad_len = max_cand_num - logits.shape[0]
            if pad_len > 0:
                padded = F.pad(logits, (0, 0, 0, pad_len), value=float('-inf'))
            else:
                padded = logits
            padded_logits.append(padded)
        
        fuse_logits = torch.stack(padded_logits, dim=0)

        # Mask invalid candidates
        for bn in range(batch_size):
            cand_num = cand_nums[bn].item()
            if cand_num < max_cand_num:
                fuse_logits[bn, cand_num:] = float('-inf')

        return {
            'fuse_logits': fuse_logits,
            'gmap_vpids': gmap_vpids,
            'gmap_masks': gmap_masks,
            'gmap_visited_masks': gmap_visited_masks,
        }
    
    def forward_distill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cand_feats: torch.Tensor,
        cand_masks: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Simplified forward pass for distillation training
        Args:
            input_ids: [B, seq_len] Tokenized schema prompt
            attention_mask: [B, seq_len] Attention mask
            cand_feats: [B, max_num_cands, 36, feat_dim] Candidate visual features
            cand_masks: [B, max_num_cands] Mask for valid candidates (1=valid, 0=padding)
        Returns:
            logits: [B, max_num_cands+1] Classification logits (first is stop)
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        max_num_cands = cand_feats.shape[1]
        
        # Process candidate features through image embeddings
        # Reshape: [B * max_num_cands, 36, feat_dim]
        cand_feats_flat = cand_feats.view(-1, cand_feats.shape[2], cand_feats.shape[3])
        
        # Process through panorama encoding
        view_lens = torch.full((batch_size * max_num_cands,), 36, dtype=torch.long, device=device)
        loc_fts = torch.zeros(batch_size * max_num_cands, 36, 7, dtype=torch.float32, device=device)
        nav_types = torch.ones(batch_size * max_num_cands, 36, dtype=torch.long, device=device)
        
        pano_inputs = {
            'view_img_fts': cand_feats_flat,
            'view_lens': view_lens,
            'loc_fts': loc_fts,
            'nav_types': nav_types,
            'obj_img_fts': None,
            'obj_lens': None,
            'obj_loc_fts': None,
        }
        
        panorama_out = self.img_embeddings.forward_panorama_per_step(**pano_inputs)
        pano_embeds = panorama_out['pano_embeds']  # [B * max_num_cands, 36, hidden_size]
        pano_masks = panorama_out['pano_masks']  # [B * max_num_cands, 36]
        
        # Average pool over views: [B * max_num_cands, hidden_size]
        cand_embeds_flat = (pano_embeds * pano_masks.unsqueeze(-1)).sum(dim=1) / pano_masks.sum(dim=1, keepdim=True).clamp(min=1)
        cand_embeds = cand_embeds_flat.view(batch_size, max_num_cands, -1)  # [B, max_num_cands, hidden_size]
        
        # Count <cand> tokens in each prompt
        cand_token_id = self.lang_model.cand_token_id[0]
        cand_locations = (input_ids == cand_token_id)  # [B, seq_len]
        
        # Prepare candidate visual features for injection
        cand_vis_list = []
        for b in range(batch_size):
            num_cand_tokens = cand_locations[b].sum().item()
            num_valid_cands = cand_masks[b].sum().item()
            
            if num_cand_tokens > 0 and num_valid_cands > 0:
                # Use valid candidates (skip stop)
                num_to_use = min(num_cand_tokens, num_valid_cands)
                cand_vis_list.append(cand_embeds[b, :num_to_use])
                # If more <cand> tokens than candidates, repeat last candidate
                if num_cand_tokens > num_to_use:
                    pad_len = num_cand_tokens - num_to_use
                    if num_to_use > 0:
                        cand_vis_list[-1] = torch.cat([
                            cand_vis_list[-1],
                            cand_embeds[b, num_to_use-1:num_to_use].repeat(pad_len, 1)
                        ], dim=0)
                    else:
                        # No valid candidates, use zero
                        cand_vis_list[-1] = torch.zeros(num_cand_tokens, cand_embeds.shape[2], device=device)
            else:
                cand_vis_list.append(torch.zeros(0, cand_embeds.shape[2], device=device))
        
        if len(cand_vis_list) > 0 and cand_vis_list[0].shape[0] > 0:
            cand_vis = torch.cat(cand_vis_list, dim=0)  # [total_cand_tokens, hidden_size]
        else:
            cand_vis = None
        
        # Forward through language model
        output = self.lang_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cand_vis=cand_vis,
            hist_vis=None,
        )
        
        hidden_states = output.hidden_states  # [B, seq_len, hidden_size]
        
        # Extract hidden state at <cls_1> position
        cls_token_id = self.lang_model.cls_token_id[0]
        cls_hiddens = []
        
        for b in range(batch_size):
            cls_pos = (input_ids[b] == cls_token_id).nonzero(as_tuple=True)[0]
            if len(cls_pos) > 0:
                cls_hidden = hidden_states[b, cls_pos[0]]  # [hidden_size]
            else:
                # Fallback: use last token
                cls_hidden = hidden_states[b, -1]
            cls_hiddens.append(cls_hidden)
        
        cls_hiddens = torch.stack(cls_hiddens, dim=0)  # [B, hidden_size]
        
        # Get logits from classification head
        all_logits = self.out_head(cls_hiddens)  # [B, 100]
        
        # Map to candidate logits
        # The out_head outputs [B, 100], we need to map to [B, max_num_cands+1]
        # We'll create a projection or use direct indexing
        # For simplicity, we'll create a learnable projection layer if needed
        # For now, use first max_num_cands+1 dimensions
        
        # Get max number of candidates across batch (including stop)
        max_cands_per_batch = []
        for b in range(batch_size):
            num_valid = cand_masks[b].sum().item()
            max_cands_per_batch.append(num_valid + 1)  # +1 for stop
        max_output_cands = max(max_cands_per_batch) if max_cands_per_batch else (max_num_cands + 1)
        
        # Create logits for each sample
        fuse_logits_list = []
        for b in range(batch_size):
            num_valid = cand_masks[b].sum().item()
            num_needed = num_valid + 1  # +1 for stop
            
            if all_logits.shape[1] >= num_needed:
                sample_logits = all_logits[b, :num_needed]  # [num_needed]
            else:
                # Pad if needed
                sample_logits = F.pad(all_logits[b], (0, num_needed - all_logits.shape[1]), value=float('-inf'))
            
            # Pad to max_output_cands for batching
            if num_needed < max_output_cands:
                sample_logits = F.pad(sample_logits, (0, max_output_cands - num_needed), value=float('-inf'))
            
            fuse_logits_list.append(sample_logits)
        
        fuse_logits = torch.stack(fuse_logits_list, dim=0)  # [B, max_output_cands]
        
        # Apply masks: stop (position 0) is always valid, candidates 1+ need to be masked if invalid
        for b in range(batch_size):
            num_valid = cand_masks[b].sum().item()
            # Position 0 (stop) is always valid
            # Positions 1 to num_valid are valid candidates
            # Positions num_valid+1 to max_output_cands should be masked
            if num_valid + 1 < max_output_cands:
                fuse_logits[b, num_valid+1:] = float('-inf')
        
        return {
            'fuse_logits': fuse_logits,
        }

