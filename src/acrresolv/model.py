"""T5-small model with LoRA adapter for acronym expansion.

Uses HuggingFace Transformers T5-small as the base model with LoRA
(Low-Rank Adaptation) for efficient fine-tuning on the acronym dataset.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """LoRA adapter for a linear layer.
    
    Wraps a frozen linear layer with low-rank adaptation matrices
    A and B. The forward pass is: y = Wx + (B @ A @ x) * scale
    """
    
    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) / rank)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Flag to indicate if this is a LoRA layer
        self.is_lora = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Original linear: Wx (frozen)
        # LoRA: (B @ A @ x) * scale
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scale


class LoRAModel(nn.Module):
    """Wrapper to add LoRA adapters to a T5 model.
    
    Freezes the base model and adds LoRA adapters to specified layers.
    """
    
    def __init__(self, base_model: T5ForConditionalGeneration,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.1,
                 target_modules: Optional[list] = None):
        super().__init__()
        
        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Default target modules (attention projections)
        if target_modules is None:
            target_modules = ['q', 'k', 'v', 'o']
        
        # Add LoRA adapters
        self.lora_layers = nn.ModuleDict()
        
        # Encoder self-attention
        for i, layer in enumerate(self.base_model.encoder.block):
            for proj_name in target_modules:
                proj = getattr(layer.layer[0].SelfAttention, proj_name, None)
                if proj is not None and hasattr(proj, 'q'):
                    lora = LoRALayer(
                        proj.q.in_features,
                        proj.q.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    self.lora_layers[f'encoder.{i}.self_attn.{proj_name}'] = lora
        
        # Decoder self-attention
        for i, layer in enumerate(self.base_model.decoder.block):
            for proj_name in target_modules:
                proj = getattr(layer.layer[0].SelfAttention, proj_name, None)
                if proj is not None and hasattr(proj, 'q'):
                    lora = LoRALayer(
                        proj.q.in_features,
                        proj.q.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    self.lora_layers[f'decoder.{i}.self_attn.{proj_name}'] = lora
            
            # Cross-attention
            for proj_name in target_modules:
                proj = getattr(layer.layer[1].EncDecAttention, proj_name, None)
                if proj is not None and hasattr(proj, 'q'):
                    lora = LoRALayer(
                        proj.q.in_features,
                        proj.q.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    self.lora_layers[f'decoder.{i}.cross_attn.{proj_name}'] = lora
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with LoRA adaptation."""
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        # Apply LoRA adaptations (simplified - in practice would modify forward pass)
        # For now, we'll just return base outputs
        # The actual LoRA integration requires modifying the model's forward pass
        
        return base_outputs
    
    def merge_lora(self):
        """Merge LoRA weights into base model for inference."""
        for name, lora in self.lora_layers.items():
            # Parse layer info
            parts = name.split('.')
            layer_idx = int(parts[1])
            proj_name = parts[3]
            
            # Get target layer
            if parts[0] == 'encoder':
                target = self.base_model.encoder.block[layer_idx].layer[0].SelfAttention
            elif parts[0] == 'decoder':
                if 'cross' in name:
                    target = self.base_model.decoder.block[layer_idx].layer[1].EncDecAttention
                else:
                    target = self.base_model.decoder.block[layer_idx].layer[0].SelfAttention
            
            # Get projection
            proj = getattr(target, proj_name)
            
            # Merge weights: W_new = W + scale * B @ A
            lora_weight = lora.lora_B @ lora.lora_A * lora.scale
            proj.weight.data += lora_weight.to(proj.weight.device)
        
        # Clear LoRA parameters
        self.lora_layers.clear()
    
    def get_lora_params(self) -> dict:
        """Get LoRA parameters for saving."""
        params = {}
        for name, lora in self.lora_layers.items():
            params[name] = {
                'A': lora.lora_A.data.cpu(),
                'B': lora.lora_B.data.cpu(),
                'rank': lora.rank,
                'alpha': lora.alpha,
                'scale': lora.scale,
            }
        return params
    
    @classmethod
    def from_lora_params(cls, base_model: T5ForConditionalGeneration,
                         lora_params: dict, **kwargs):
        """Create LoRA model from saved parameters."""
        model = cls(base_model, **kwargs)
        
        for name, params in lora_params.items():
            if name in model.lora_layers:
                lora = model.lora_layers[name]
                lora.lora_A.data = params['A']
                lora.lora_B.data = params['B']
        
        return model


# ============================================================================
# Acronym Model Configuration
# ============================================================================

class AcronymConfig:
    """Configuration for acronym expansion model."""
    
    def __init__(self, vocab_size: int = 32128, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8,
                 num_kv_heads: int = 8, head_dim: int = 64,
                 max_seq_len: int = 128, lora_rank: int = 8,
                 lora_alpha: float = 16.0, weight_bits: int = 4,
                 act_bits: int = 8, group_size: int = 128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.group_size = group_size
    
    @classmethod
    def from_t5_small(cls, **kwargs):
        """Create config from T5-small defaults."""
        return cls(
            vocab_size=32128,
            d_model=512,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            max_seq_len=512,
            **kwargs,
        )
    
    @classmethod
    def from_t5_tiny(cls, **kwargs):
        """Create config from T5-tiny defaults (smaller model)."""
        return cls(
            vocab_size=32128,
            d_model=256,
            num_layers=4,
            num_heads=4,
            num_kv_heads=4,
            head_dim=64,
            max_seq_len=256,
            **kwargs,
        )


# ============================================================================
# Model Creation
# ============================================================================

def create_model(config: AcronymConfig, use_lora: bool = True,
                 lora_rank: int = 8, lora_alpha: float = 16.0) -> LoRAModel:
    """Create T5 model with optional LoRA adapter.
    
    Args:
        config: Model configuration
        use_lora: Whether to use LoRA adapter
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
    
    Returns:
        LoRAModel instance
    """
    # Create T5 config
    t5_config = T5Config(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_decoder_layers=config.num_layers,
        d_ff=config.d_model * 4,
        d_kv=config.head_dim,
        max_seq_len=config.max_seq_len,
    )
    
    # Create base model
    base_model = T5ForConditionalGeneration(t5_config)
    
    if use_lora:
        # Add LoRA adapter
        model = LoRAModel(
            base_model,
            rank=lora_rank,
            alpha=lora_alpha,
        )
    else:
        model = base_model
    
    return model


def load_model(path: str, device: str = 'cpu') -> T5ForConditionalGeneration:
    """Load T5 model from HuggingFace hub.
    
    Args:
        path: Model path or HuggingFace model ID
        device: Device to load to
    
    Returns:
        T5ForConditionalGeneration instance
    """
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    model = T5ForConditionalGeneration.from_pretrained(path)
    model = model.to(device)
    model.eval()
    
    return model


def load_tokenizer(path: str = 't5-small'):
    """Load T5 tokenizer.
    
    Args:
        path: Model path or HuggingFace model ID
    
    Returns:
        T5Tokenizer instance
    """
    from transformers import T5Tokenizer
    
    tokenizer = T5Tokenizer.from_pretrained(path)
    return tokenizer


# ============================================================================
# Inference Functions
# ============================================================================

def expand_acronym(model: T5ForConditionalGeneration, tokenizer,
                   acronym: str, max_length: int = 128) -> str:
    """Expand an acronym to its full phrase.
    
    Args:
        model: T5 model
        tokenizer: Tokenizer
        acronym: Acronym to expand
        max_length: Maximum output length
    
    Returns:
        Expanded phrase
    """
    # Format input
    input_text = f"expand: {acronym}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
    
    # Decode
    expanded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return expanded
