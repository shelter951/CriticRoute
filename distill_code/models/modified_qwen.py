import sys
from pathlib import Path

# 添加项目根目录到Python路径（必须在其他导入之前）
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessor
from tools.trie import Trie

# 尝试导入Qwen2，如果失败则使用Qwen1或AutoModel
try:
    from transformers import Qwen2ForCausalLM
    QwenBaseModel = Qwen2ForCausalLM
except ImportError:
    try:
        from transformers import QwenForCausalLM
        QwenBaseModel = QwenForCausalLM
    except ImportError:
        # 如果都没有，使用AutoModelForCausalLM，但需要特殊处理
        QwenBaseModel = AutoModelForCausalLM


class ModifiedQwenForCausalLM(QwenBaseModel):
    """
    Modified Qwen3 model for navigation tasks, similar to ModifiedLlamaForCausalLM
    """
    
    def __init__(self, config, extra_config):
        QwenBaseModel.__init__(self, config)
        self._init_modified_lm(extra_config)
    
    def _init_modified_lm(self, extra_config):
        """Initialize ModifiedLM components"""
        if extra_config.precision == 'fp16':
            self.model_type = torch.float16
        elif 'bf16' in extra_config.precision or 'bfloat16' in extra_config.precision:
            self.model_type = torch.bfloat16
        else:
            self.model_type = torch.float32

        self.model = self.model.to(self.model_type)
        self.lm_head = self.lm_head.to(self.model_type)
        
        self.hidden_size = self.config.hidden_size

    def init_tokenizer(self, pretrained_model_name_or_path: str):
        """Initialize tokenizer with special tokens"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            padding_side="left", 
            truncation_side='left',
            trust_remote_code=True
        )

        # Add special tokens for navigation
        self.cand_token = ['<cand>']
        self.hist_token = ['<hist>']
        self.obj_token = ['<obj>']
        self.cls_token = ['<cls_1>', '<cls_2>']
        
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.cand_token + self.hist_token + self.obj_token + self.cls_token}
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        # Get token IDs
        self.cand_token_id = self.tokenizer.encode("".join(self.cand_token), add_special_tokens=False)
        self.hist_token_id = self.tokenizer.encode("".join(self.hist_token), add_special_tokens=False)
        self.obj_token_id = self.tokenizer.encode("".join(self.obj_token), add_special_tokens=False)
        self.cls_token_id = self.tokenizer.encode("".join(self.cls_token), add_special_tokens=False)
        self.special_token_ids = self.cand_token_id + self.hist_token_id + self.obj_token_id + self.cls_token_id
        
        # Resize token embeddings
        self.resize_token_embeddings(len(self.tokenizer))

    def tokenize(self, text: str, add_special_tokens: bool=True):
        """Tokenize text input"""
        if isinstance(text, list):
            batch_text = self.tokenizer(
                text,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
                return_token_type_ids=True
            )
        else:
            batch_text = self.tokenizer(
                text,
                max_length=1024,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
                return_token_type_ids=True
            )
        return batch_text

    def forward(
        self, 
        input_ids,
        attention_mask, 
        labels=None,
        cand_vis=None, 
        hist_vis=None, 
        obj_vis=None, 
        **kwargs
    ):
        """Forward pass with visual feature injection"""
        # Find locations of special tokens
        hist_locations = (input_ids >= self.hist_token_id[0]) & (input_ids <= self.hist_token_id[-1])
        cand_locations = (input_ids >= self.cand_token_id[0]) & (input_ids <= self.cand_token_id[-1])
        obj_locations = (input_ids >= self.obj_token_id[0]) & (input_ids <= self.obj_token_id[-1])

        # Get input embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Inject visual features at special token locations
        # Ensure all visual features are on the same device as inputs_embeds
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        
        if cand_locations.sum() != 0 and cand_vis is not None:
            # Handle cand_vis: ensure it's a tensor on the correct device
            if isinstance(cand_vis, (list, tuple)):
                cand_vis = torch.stack([t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor) else t for t in cand_vis], dim=0)
            if isinstance(cand_vis, torch.Tensor):
                cand_vis = cand_vis.to(device=device, dtype=dtype)
                inputs_embeds[cand_locations] += cand_vis
        
        if hist_locations.sum() != 0 and hist_vis is not None:
            # Handle hist_vis: ensure it's a tensor on the correct device
            # hist_vis might come as:
            # 1. A tensor (from nav_qwen3.py after processing)
            # 2. A list of tensors
            # 3. A nested list of tensors
            
            if isinstance(hist_vis, torch.Tensor):
                # Already a tensor, just move to device
                hist_vis = hist_vis.to(device=device, dtype=dtype)
            elif isinstance(hist_vis, (list, tuple)):
                # Handle list cases
                if len(hist_vis) == 0:
                    hist_vis = None
                elif isinstance(hist_vis[0], torch.Tensor):
                    # Simple list of tensors: stack them
                    hist_vis = torch.stack([t.to(device=device, dtype=dtype) for t in hist_vis], dim=0)
                elif isinstance(hist_vis[0], (list, tuple)) and len(hist_vis[0]) > 0:
                    # Nested list: flatten and stack
                    flattened = []
                    for sublist in hist_vis:
                        if isinstance(sublist, (list, tuple)):
                            for t in sublist:
                                if isinstance(t, torch.Tensor):
                                    flattened.append(t.to(device=device, dtype=dtype))
                        elif isinstance(sublist, torch.Tensor):
                            flattened.append(sublist.to(device=device, dtype=dtype))
                    if len(flattened) > 0:
                        hist_vis = torch.stack(flattened, dim=0)
                    else:
                        hist_vis = None
                else:
                    hist_vis = None
            else:
                hist_vis = None
            
            # Now hist_vis should be a tensor on the correct device, or None
            if hist_vis is not None and isinstance(hist_vis, torch.Tensor):
                # Ensure hist_vis is on the correct device (double check)
                hist_vis = hist_vis.to(device=device, dtype=dtype)
                # Match the shape for indexing
                num_hist = hist_locations.sum().item()
                if num_hist > 0:
                    if hist_vis.dim() == 1:
                        # 1D tensor: match the number of hist locations
                        if hist_vis.shape[0] != num_hist:
                            if hist_vis.shape[0] < num_hist:
                                # Repeat the last element
                                hist_vis = torch.cat([hist_vis, hist_vis[-1:].repeat(num_hist - hist_vis.shape[0])], dim=0)
                            else:
                                hist_vis = hist_vis[:num_hist]
                        inputs_embeds[hist_locations] += hist_vis
                    elif hist_vis.dim() == 2 and hist_vis.shape[0] == num_hist:
                        # 2D tensor with matching batch size
                        inputs_embeds[hist_locations] += hist_vis
                    else:
                        # Try to broadcast or reshape
                        inputs_embeds[hist_locations] += hist_vis
        
        if obj_locations.sum() != 0 and obj_vis is not None:
            # Handle obj_vis: ensure it's a tensor on the correct device
            if isinstance(obj_vis, (list, tuple)):
                obj_vis = torch.stack([t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor) else t for t in obj_vis], dim=0)
            if isinstance(obj_vis, torch.Tensor):
                obj_vis = obj_vis.to(device=device, dtype=dtype)
                inputs_embeds[obj_locations] += obj_vis

        # Remove inputs_embeds from kwargs if present (to avoid conflict)
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        
        # Forward through model
        outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Mask special tokens in logits
        logits_mask = torch.ones_like(logits, dtype=torch.bool).to(logits.device)
        logits_mask[:, :, self.special_token_ids] = False
        logits = logits.masked_fill(~logits_mask, float('-inf'))

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
    
    def get_encoder(self):
        """Get the encoder (model)"""
        return self.model

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, 
        cand_vis=None, hist_vis=None, obj_vis=None, **kwargs
    ):
        """Prepare inputs for generation"""
        # 使用父类的prepare_inputs_for_generation
        if hasattr(super(), 'prepare_inputs_for_generation'):
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values,
                attention_mask,
                inputs_embeds,
                **kwargs
            )
        else:
            # 降级方案：手动构建
            model_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if past_key_values is not None:
                model_inputs['past_key_values'] = past_key_values
            if inputs_embeds is not None:
                model_inputs['inputs_embeds'] = inputs_embeds
            model_inputs.update(kwargs)
        if not past_key_values:
            for k in ['cand_vis', 'hist_vis', 'obj_vis']:
                model_inputs[k] = eval(k)

        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, extra_config, **kwargs):
        """Load pretrained model"""
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model = cls(config, extra_config)
        
        # Load weights
        state_dict = torch.load(
            f"{pretrained_model_name_or_path}/pytorch_model.bin",
            map_location="cpu"
        ) if (Path(pretrained_model_name_or_path) / "pytorch_model.bin").exists() else None
        
        if state_dict is None:
            # Try loading from HuggingFace using AutoModel
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True,
                **kwargs
            )
            # 尝试加载权重
            try:
                model.model.load_state_dict(base_model.model.state_dict(), strict=False)
                if hasattr(model, 'lm_head') and hasattr(base_model, 'lm_head'):
                    model.lm_head.load_state_dict(base_model.lm_head.state_dict(), strict=False)
            except Exception as e:
                print(f"Warning: Could not load all weights: {e}")
                # 尝试部分加载
                model.load_state_dict(base_model.state_dict(), strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        
        return model

