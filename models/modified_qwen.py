import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.logits_process import LogitsProcessor
from tools.trie import Trie


class ModifiedQwenForCausalLM(Qwen2ForCausalLM):
    """
    Modified Qwen3 model for navigation tasks, similar to ModifiedLlamaForCausalLM
    """
    
    def __init__(self, config, extra_config):
        Qwen2ForCausalLM.__init__(self, config)
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
        if cand_locations.sum() != 0 and cand_vis is not None:
            inputs_embeds[cand_locations] += cand_vis
        if hist_locations.sum() != 0 and hist_vis is not None:
            inputs_embeds[hist_locations] += hist_vis
        if obj_locations.sum() != 0 and obj_vis is not None:
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
        model_inputs = Qwen2ForCausalLM.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values,
            attention_mask,
            inputs_embeds,
            **kwargs
        )
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
            # Try loading from HuggingFace
            from transformers import Qwen2ForCausalLM
            base_model = Qwen2ForCausalLM.from_pretrained(
                pretrained_model_name_or_path, 
                trust_remote_code=True,
                **kwargs
            )
            model.model.load_state_dict(base_model.model.state_dict())
            model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        else:
            model.load_state_dict(state_dict, strict=False)
        
        return model

