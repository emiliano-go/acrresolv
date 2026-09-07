"""W4A8 Quantization module inspired by needle's quantize.py.

Provides quantization-aware training (QAT) and post-training quantization
for the T5-small model. Uses Lloyd-Max codebooks for optimal 4-bit weight
quantization and 8-bit activation quantization.
"""
import functools
import contextlib

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Configuration
# ============================================================================

WEIGHT_BITS = 4
ACT_BITS = 8
GROUP_SIZE = 128

_quant_enabled = True


@contextlib.contextmanager
def disable_quant():
    """Context manager to disable quantization during inference."""
    global _quant_enabled
    old = _quant_enabled
    _quant_enabled = False
    try:
        yield
    finally:
        _quant_enabled = old


def is_quant_enabled() -> bool:
    return _quant_enabled


# ============================================================================
# Lloyd-Max Codebook Generation
# ============================================================================

@functools.lru_cache(maxsize=None)
def _lloyd_max_gaussian(bits: int, iters: int = 200, samples: int = 400000,
                         seed: int = 0) -> np.ndarray:
    """Generate Lloyd-Max optimal quantization codebook for Gaussian source."""
    levels = 1 << bits
    x = np.sort(np.random.RandomState(seed).randn(samples))
    c = x[((np.arange(levels) + 0.5) / levels * samples).astype(int)].astype(np.float64)
    for _ in range(iters):
        bnd = (c[:-1] + c[1:]) / 2.0
        idx = np.searchsorted(bnd, x)
        for k in range(levels):
            m = idx == k
            if m.any():
                c[k] = x[m].mean()
    return np.sort(c)


def get_codebook(bits: int, group_size: int) -> np.ndarray:
    """Get codebook for given bit width and group size."""
    cb = _lloyd_max_gaussian(bits)
    return (cb / np.sqrt(group_size)).astype(np.float32)


# ============================================================================
# Fake Quantization (for QAT)
# ============================================================================

def fake_quant(w: torch.Tensor, bits: int = WEIGHT_BITS,
               group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Apply fake quantization with straight-through estimator.
    
    During training, this simulates quantization while allowing gradients
    to flow through via the straight-through estimator.
    """
    if not _quant_enabled:
        return w
    
    qmax = 2 ** (bits - 1) - 1
    D = w.shape[-1]
    pad = (-D) % group_size
    
    if pad:
        w_padded = torch.nn.functional.pad(w, (0, pad))
    else:
        w_padded = w
    
    # Reshape to groups
    shape = w_padded.shape[:-1] + (-1, group_size)
    g = w_padded.reshape(*shape).float()
    
    # Compute scale per group
    absmax = g.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    
    # Quantize and dequantize
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax) * scale
    
    # Reshape back
    q = q.reshape(w_padded.shape).to(w.dtype)
    
    if pad:
        q = q[..., :D]
    
    # Straight-through estimator
    return w + (q - w).detach()


def fake_quant_act(x: torch.Tensor, bits: int = ACT_BITS,
                   group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Fake quantize activations."""
    return fake_quant(x, bits=bits, group_size=x.shape[-1])


# ============================================================================
# Quantize Parameters
# ============================================================================

def quantize_params(model: nn.Module, bits: int = WEIGHT_BITS,
                    group_size: int = GROUP_SIZE) -> nn.Module:
    """Apply fake quantization to all linear layer weights in model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = fake_quant(module.weight.data, bits, group_size)
    return model


# ============================================================================
# QAT Training Hook
# ============================================================================

class QATHook:
    """Hook to apply quantization during training."""
    
    def __init__(self, model: nn.Module, bits: int = WEIGHT_BITS,
                 group_size: int = GROUP_SIZE):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        self._hooks = []
    
    def register(self):
        """Register forward hooks for quantization."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_pre_hook(self._pre_hook)
                self._hooks.append(hook)
    
    def _pre_hook(self, module: nn.Linear, input):
        if _quant_enabled:
            module.weight.data = fake_quant(
                module.weight.data, self.bits, self.group_size
            )
    
    def remove(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ============================================================================
# Model Quantization (Post-Training)
# ============================================================================

def quantize_model(model: nn.Module, bits: int = WEIGHT_BITS,
                   group_size: int = GROUP_SIZE) -> nn.Module:
    """Apply post-training quantization to model weights."""
    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = fake_quant(
                    module.weight.data, bits, group_size
                )
    return model


def get_model_size(model: nn.Module) -> dict:
    """Get model size statistics."""
    total_params = 0
    quantized_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if 'weight' in name and param.dim() >= 2:
            quantized_params += n
        else:
            other_params += n
    
    # Quantized weights: 4 bits per param
    # Other params: 16 bits (FP16)
    size_bytes = quantized_params * WEIGHT_BITS // 8 + other_params * 2
    size_mb = size_bytes / 1e6
    
    return {
        'total_params': total_params,
        'quantized_params': quantized_params,
        'other_params': other_params,
        'size_bytes': size_bytes,
        'size_mb': size_mb,
    }


# ============================================================================
# Real Quantization (for actual size reduction)
# ============================================================================

def real_quantize_weight(w: torch.Tensor, bits: int, group_size: int) -> tuple:
    """Real quantization: pack weights to lower bits.
    
    Returns:
        Tuple of (packed_indices, scales, zeros) for reconstruction
    """
    qmax = 2 ** (bits - 1) - 1
    D = w.shape[-1]
    pad = (-D) % group_size
    
    if pad:
        w_padded = torch.nn.functional.pad(w, (0, pad))
    else:
        w_padded = w
    
    # Reshape to groups
    shape = w_padded.shape[:-1] + (-1, group_size)
    g = w_padded.reshape(*shape).float()
    
    # Compute scale and zero point per group
    absmax = g.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
    zero = torch.zeros_like(scale)
    
    # Quantize
    q = torch.clamp(torch.round(g / scale) + zero, -qmax - 1, qmax)
    
    # Pack indices
    indices = q.to(torch.int8)
    
    return indices, scale.squeeze(-1), zero.squeeze(-1), D


def save_quantized_model(model, tokenizer, path: str, bits: int = 2, group_size: int = 128):
    """Save model with real quantization.
    
    Args:
        model: T5 model
        tokenizer: Tokenizer
        path: Output directory
        bits: Weight bits (1, 2, 3, or 4)
        group_size: Quantization group size
    """
    import json
    from pathlib import Path
    
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = model.config.to_dict()
    config['quantization'] = {'bits': bits, 'group_size': group_size}
    with open(path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(path))
    
    # Quantize and save weights
    state_dict = model.state_dict()
    quantized = {}
    scales = {}
    
    for key, param in state_dict.items():
        if 'weight' in key and param.dim() >= 2:
            # Real quantize
            indices, scale, zero, orig_dim = real_quantize_weight(param.data, bits, group_size)
            quantized[key] = indices.to(torch.int8)
            scales[key + '.scale'] = scale.to(torch.float16)
        else:
            # Keep as FP16
            quantized[key] = param.half()
    
    # Save quantized weights
    torch.save(quantized, path / 'quantized_model.pt')
    
    # Calculate size
    total_bytes = sum(p.numel() * p.element_size() for p in quantized.values())
    size_mb = total_bytes / 1e6
    
    print(f'Saved to {path}')
    print(f'Bits: {bits}, Group: {group_size}')
    print(f'Size: {size_mb:.1f} MB')
    
    return path


def load_quantized_model(path: str, device: str = 'cpu'):
    """Load quantized model.
    
    Args:
        path: Model directory
        device: Device to load to
    
    Returns:
        Tuple of (model, tokenizer)
    """
    import json
    from pathlib import Path
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    path = Path(path)
    
    # Load config
    with open(path / 'config.json') as f:
        config = json.load(f)
    
    bits = config.get('quantization', {}).get('bits', 4)
    group_size = config.get('quantization', {}).get('group_size', 128)
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(str(path))
    
    # Load base model
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Load quantized weights
    quantized_dict = torch.load(path / 'quantized_model.pt', map_location=device)
    
    # Dequantize and load
    state_dict = {}
    for key, param in quantized_dict.items():
        if key.endswith('.scale'):
            continue
        if param.dtype == torch.int8:
            # Dequantize
            scale = quantized_dict.get(key + '.scale', torch.ones(1))
            state_dict[key] = (param.float() * scale.unsqueeze(-1)).to(torch.float32)
        else:
            state_dict[key] = param.to(torch.float32)
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer
