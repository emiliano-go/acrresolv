"""Binary export module for .acr format inspired by needle's export.py.

Provides export and import of quantized T5-small models with LoRA adapters
in a custom binary format optimized for fast loading and inference.

Format: W4A8 quantized weights with Lloyd-Max codebooks.
"""
import struct
from pathlib import Path

import numpy as np
import torch


# ============================================================================
# Format Constants
# ============================================================================

# Magic bytes for .acr format
MAGIC = b'ACR1'
VERSION = 1

# Data types
DTYPE_FP16 = 1
DTYPE_FP32 = 2
DTYPE_INT4 = 3
DTYPE_RAW = 4

# Alignment
ALIGN = 64

# Header format: magic(4) + version(u32) + num_tensors(u32) + 
# codebook_len(u32) + vocab_size(u32) + d_model(u32) + num_layers(u32) +
# num_heads(u32) + num_kv_heads(u32) + head_dim(u32) + max_seq_len(u32) +
# lora_rank(u32) + lora_alpha(u32) + weight_bits(u32) + act_bits(u32) +
# group_size(u32)
_HDR_FMT = '<4sIIIIIIIIIIIIIII'
_HDR_SIZE = struct.calcsize(_HDR_FMT)

# Tensor record: dtype(u8) + ndim(u8) + _pad(u16) + shape(4xu32) +
# offset(u64) + nbytes(u64) + group_size(u32) + bits(u32)
_REC_FMT = '<BBHIIIIQQII'
REC_SIZE = struct.calcsize(_REC_FMT)


# ============================================================================
# Alignment Helper
# ============================================================================

def _align(n: int) -> int:
    """Align to 64-byte boundary."""
    return (n + ALIGN - 1) & ~(ALIGN - 1)


# ============================================================================
# Quantization Helpers
# ============================================================================

def _nearest_idx(x: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Find nearest codebook index for each element."""
    flat = x.reshape(-1)
    pos = np.clip(np.searchsorted(cb, flat), 1, len(cb) - 1)
    left, right = cb[pos - 1], cb[pos]
    idx = np.where(np.abs(flat - left) <= np.abs(flat - right), pos - 1, pos)
    return idx.reshape(x.shape).astype(np.uint8)


def _pack_lsb(idx: np.ndarray, bits: int) -> np.ndarray:
    """Pack indices LSB-first."""
    out, in_pad = idx.shape
    g = idx.reshape(out, in_pad // 8, 8).astype(np.uint64)
    word = np.zeros(g.shape[:-1], np.uint64)
    for i in range(8):
        word |= g[..., i] << (i * bits)
    packed = np.empty(word.shape + (bits,), np.uint8)
    for b in range(bits):
        packed[..., b] = (word >> (8 * b)) & 0xFF
    return packed.reshape(out, in_pad * bits // 8)


def _unpack_lsb(packed: np.ndarray, bits: int, in_pad: int) -> np.ndarray:
    """Unpack LSB-first indices."""
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << (8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = (1 << bits) - 1
    for i in range(8):
        idx[..., i] = (word >> (i * bits)) & mask
    return idx.reshape(out, in_pad)


def _pack_int4(w: np.ndarray, bits: int = 4, group: int = 128) -> tuple:
    """Pack weights to INT4 format with Lloyd-Max codebook."""
    from .quantize import get_codebook
    
    cb = get_codebook(bits, group)
    out, D = w.shape
    pad = (-D) % group
    wp = np.pad(w, ((0, 0), (0, pad))) if pad else w
    in_pad = wp.shape[1]
    
    # Simple quantization (without Hadamard for now)
    g = wp.reshape(out, in_pad // group, group).astype(np.float32)
    
    # Compute norms
    norm = np.sqrt((g ** 2).sum(-1, keepdims=True))
    unit = g / np.maximum(norm, 1e-12)
    
    # Find nearest codebook entries
    idx = _nearest_idx(unit, cb).reshape(out, in_pad)
    
    # Pack indices
    packed = _pack_lsb(idx, bits)
    
    return packed, norm[:, :, 0].astype(np.float16)


def _unpack_int4(packed: np.ndarray, norms: np.ndarray, out: int,
                 in_dim: int, bits: int = 4, group: int = 128) -> np.ndarray:
    """Unpack INT4 weights."""
    from .quantize import get_codebook
    
    cb = get_codebook(bits, group)
    in_pad = (in_dim + group - 1) // group * group
    
    # Unpack indices
    idx = _unpack_lsb(packed, bits, in_pad)
    
    # Dequantize
    unit = cb[idx].reshape(out, in_pad // group, group)
    rot = unit * norms.reshape(out, in_pad // group, 1).astype(np.float32)
    w = rot.reshape(out, in_pad)
    
    return w[:, :in_dim]


# ============================================================================
# Tensor Serialization
# ============================================================================

class _Tensor:
    """Internal tensor representation for export."""
    __slots__ = ("name", "dtype", "shape", "blob", "group", "bits", "offset")
    
    def __init__(self, name, dtype, shape, blob, group=0, bits=0):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)
        self.blob, self.group, self.bits, self.offset = blob, group, bits, 0


def _fp16_tensor(name: str, arr: np.ndarray) -> _Tensor:
    """Create FP16 tensor."""
    arr = np.asarray(arr, np.float16)
    return _Tensor(name, DTYPE_FP16, arr.shape, arr.tobytes())


def _fp32_tensor(name: str, arr: np.ndarray) -> _Tensor:
    """Create FP32 tensor."""
    arr = np.asarray(arr, np.float32)
    return _Tensor(name, DTYPE_FP32, arr.shape, arr.tobytes())


def _int4_tensor(name: str, mat: np.ndarray, bits: int = 4,
                 group: int = 128) -> _Tensor:
    """Create INT4 quantized tensor."""
    packed, norms = _pack_int4(np.asarray(mat, np.float32), bits, group)
    blob = packed.tobytes() + norms.tobytes()
    return _Tensor(name, DTYPE_INT4, mat.shape, blob, group, bits)


def _raw_tensor(name: str, data: bytes) -> _Tensor:
    """Create RAW tensor (e.g., tokenizer)."""
    return _Tensor(name, DTYPE_RAW, (), data)


# ============================================================================
# Export Functions
# ============================================================================

def export_model(model, tokenizer, config, output_path: str,
                 bits: int = 4, group: int = 128) -> dict:
    """Export T5 model with LoRA to .acr format.
    
    Args:
        model: T5 model with LoRA adapter
        tokenizer: Tokenizer
        config: Model config
        output_path: Output file path
        bits: Weight quantization bits
        group: Quantization group size
    
    Returns:
        dict with export statistics
    """
    # Get model state dict
    state_dict = model.state_dict()
    
    # Collect tensors
    tensors = []
    
    # Embedding
    if 'shared.weight' in state_dict:
        tensors.append(_int4_tensor('embedding', state_dict['shared'].numpy(), bits, group))
    
    # Encoder layers
    for i in range(config.num_layers):
        prefix = f'encoder.block.{i}'
        
        # Self-attention
        for proj in ['q', 'k', 'v']:
            key = f'{prefix}.layer.0.SelfAttention.{proj}.weight'
            if key in state_dict:
                tensors.append(_int4_tensor(
                    f'layer{i:02d}.{proj}_proj',
                    state_dict[key].numpy().T, bits, group
                ))
        
        # Output projection
        key = f'{prefix}.layer.0.SelfAttention.o.weight'
        if key in state_dict:
            tensors.append(_int4_tensor(
                f'layer{i:02d}.out_proj',
                state_dict[key].numpy().T, bits, group
            ))
        
        # Layer norm
        key = f'{prefix}.layer.0.layer_norm.weight'
        if key in state_dict:
            tensors.append(_fp16_tensor(f'layer{i:02d}.norm_in', state_dict[key].numpy()))
        
        # FFN
        for proj in ['wi', 'wo']:
            key = f'{prefix}.layer.1.DenseReluDense.{proj}.weight'
            if key in state_dict:
                tensors.append(_int4_tensor(
                    f'layer{i:02d}.{proj}_proj',
                    state_dict[key].numpy().T, bits, group
                ))
        
        # FFN norm
        key = f'{prefix}.layer.1.layer_norm.weight'
        if key in state_dict:
            tensors.append(_fp16_tensor(f'layer{i:02d}.norm_ffn', state_dict[key].numpy()))
    
    # Decoder layers (simplified - similar to encoder)
    for i in range(config.decoder_layers):
        prefix = f'decoder.block.{i}'
        
        # Self-attention
        for proj in ['q', 'k', 'v']:
            key = f'{prefix}.layer.0.SelfAttention.{proj}.weight'
            if key in state_dict:
                tensors.append(_int4_tensor(
                    f'decoder{i:02d}.{proj}_proj',
                    state_dict[key].numpy().T, bits, group
                ))
        
        # Output projection
        key = f'{prefix}.layer.0.SelfAttention.o.weight'
        if key in state_dict:
            tensors.append(_int4_tensor(
                f'decoder{i:02d}.out_proj',
                state_dict[key].numpy().T, bits, group
            ))
        
        # Layer norm
        key = f'{prefix}.layer.0.layer_norm.weight'
        if key in state_dict:
            tensors.append(_fp16_tensor(f'decoder{i:02d}.norm_in', state_dict[key].numpy()))
        
        # Cross-attention
        for proj in ['q', 'k', 'v']:
            key = f'{prefix}.layer.1.EncDecAttention.{proj}.weight'
            if key in state_dict:
                tensors.append(_int4_tensor(
                    f'decoder{i:02d}.cross.{proj}_proj',
                    state_dict[key].numpy().T, bits, group
                ))
        
        # Cross-attention output
        key = f'{prefix}.layer.1.EncDecAttention.o.weight'
        if key in state_dict:
            tensors.append(_int4_tensor(
                f'decoder{i:02d}.cross.out_proj',
                state_dict[key].numpy().T, bits, group
            ))
        
        # Cross-attention norm
        key = f'{prefix}.layer.1.layer_norm.weight'
        if key in state_dict:
            tensors.append(_fp16_tensor(f'decoder{i:02d}.norm_cross', state_dict[key].numpy()))
        
        # FFN
        for proj in ['wi', 'wo']:
            key = f'{prefix}.layer.2.DenseReluDense.{proj}.weight'
            if key in state_dict:
                tensors.append(_int4_tensor(
                    f'decoder{i:02d}.{proj}_proj',
                    state_dict[key].numpy().T, bits, group
                ))
        
        # FFN norm
        key = f'{prefix}.layer.2.layer_norm.weight'
        if key in state_dict:
            tensors.append(_fp16_tensor(f'decoder{i:02d}.norm_ffn', state_dict[key].numpy()))
    
    # Final norms
    if 'encoder.norm.weight' in state_dict:
        tensors.append(_fp16_tensor('encoder_final_norm', state_dict['encoder.norm.weight'].numpy()))
    if 'decoder.norm.weight' in state_dict:
        tensors.append(_fp16_tensor('decoder_final_norm', state_dict['decoder.norm.weight'].numpy()))
    
    # Tokenizer (RAW)
    tokenizer_blob = _serialize_tokenizer(tokenizer)
    tensors.append(_raw_tensor('tokenizer', tokenizer_blob))
    
    # LoRA weights (if present)
    lora_tensors = []
    for key, param in state_dict.items():
        if 'lora' in key:
            lora_tensors.append(_fp16_tensor(f'lora.{key}', param.numpy()))
    tensors.extend(lora_tensors)
    
    # Build header
    codebook = np.concatenate([
        _get_codebook(4, group),
        _get_codebook(3, group),
        _get_codebook(2, group),
    ]).astype(np.float32)
    
    header = struct.pack(
        _HDR_FMT,
        MAGIC,
        VERSION,
        len(tensors),
        len(codebook),
        config.vocab_size,
        config.d_model,
        config.num_layers,
        config.num_heads,
        getattr(config, 'num_kv_heads', config.num_heads),
        config.d_model // config.num_heads,
        config.max_seq_len,
        getattr(config, 'lora_rank', 8),
        getattr(config, 'lora_alpha', 16),
        bits,
        8,  # act_bits
        group,
    ) + codebook.tobytes()
    
    # Calculate offsets
    pos = len(header) + len(tensors) * REC_SIZE
    for t in tensors:
        pos = _align(pos)
        t.offset = pos
        pos += len(t.blob)
    
    # Build directory
    directory = b''.join(
        struct.pack(
            _REC_FMT,
            t.dtype,
            len(t.shape),
            0,
            *(list(t.shape) + [0, 0, 0, 0])[:4],
            t.offset,
            len(t.blob),
            t.group,
            t.bits,
        )
        for t in tensors
    )
    
    # Build buffer
    buf = bytearray(header + directory)
    for t in tensors:
        buf.extend(b'\x00' * (t.offset - len(buf)))
        buf.extend(t.blob)
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(bytes(buf))
    
    return {
        'path': str(output_path),
        'bytes': len(buf),
        'tensors': len(tensors),
        'lora_tensors': len(lora_tensors),
    }


def _get_codebook(bits: int, group: int) -> np.ndarray:
    """Get codebook for given bit width and group size."""
    from .quantize import get_codebook
    return get_codebook(bits, group)


def _serialize_tokenizer(tokenizer) -> bytes:
    """Serialize tokenizer to binary format."""
    from .quantize import DTYPE_RAW
    
    # Simple tokenizer serialization
    # Store vocab size and special token IDs
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 1
    bos_id = tokenizer.bos_token_id or 2
    unk_id = tokenizer.unk_token_id or 3
    
    # Pack header
    header = struct.pack('<IIIII', vocab_size, pad_id, eos_id, bos_id, unk_id)
    
    # Store vocab as JSON
    import json
    vocab_json = json.dumps(tokenizer.get_vocab()).encode('utf-8')
    vocab_len = struct.pack('<I', len(vocab_json))
    
    return header + vocab_len + vocab_json


# ============================================================================
# Import Functions
# ============================================================================

def import_model(path: str, device: str = 'cpu') -> tuple:
    """Import model from .acr format.
    
    Args:
        path: Path to .acr file
        device: Device to load to
    
    Returns:
        Tuple of (state_dict, config_dict, lora_dict)
    """
    with open(path, 'rb') as f:
        raw = f.read()
    
    # Parse header
    hdr = struct.unpack_from(_HDR_FMT, raw, 0)
    (magic, version, num_tensors, cb_n, vocab_size, d_model, num_layers,
     num_heads, num_kv_heads, head_dim, max_seq_len, lora_rank, lora_alpha,
     weight_bits, act_bits, group_size) = hdr
    
    assert magic == MAGIC, f"Invalid magic: {magic}"
    assert version == VERSION, f"Unsupported version: {version}"
    
    # Parse codebook
    off = _HDR_SIZE
    codebook = np.frombuffer(raw[off:off + cb_n * 4], np.float32).copy()
    off += cb_n * 4
    
    # Parse tensor directory
    tensors = []
    for _ in range(num_tensors):
        rec = struct.unpack(_REC_FMT, raw[off:off + REC_SIZE])
        off += REC_SIZE
        
        dtype, ndim = rec[0], rec[1]
        shape = tuple(rec[3:3 + ndim])
        offset, nbytes, group, bits = rec[7], rec[8], rec[9], rec[10]
        blob = raw[offset:offset + nbytes]
        
        tensors.append({
            'dtype': dtype,
            'shape': shape,
            'blob': blob,
            'group': group,
            'bits': bits,
        })
    
    # Dequantize tensors
    state_dict = {}
    lora_dict = {}
    
    for tensor in tensors:
        name = tensor['name'] if 'name' in tensor else f'tensor_{len(state_dict)}'
        
        if tensor['dtype'] == DTYPE_FP16:
            arr = np.frombuffer(tensor['blob'], np.float16).reshape(tensor['shape'])
            state_dict[name] = torch.from_numpy(arr.astype(np.float32))
        
        elif tensor['dtype'] == DTYPE_FP32:
            arr = np.frombuffer(tensor['blob'], np.float32).reshape(tensor['shape'])
            state_dict[name] = torch.from_numpy(arr)
        
        elif tensor['dtype'] == DTYPE_INT4:
            # Dequantize INT4
            out, in_dim = tensor['shape']
            in_pad = (in_dim + tensor['group'] - 1) // tensor['group'] * tensor['group']
            
            n_packed = out * in_pad * tensor['bits'] // 8
            packed = np.frombuffer(tensor['blob'][:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(tensor['blob'][n_packed:], np.float16).reshape(
                out, in_pad // tensor['group']
            )
            
            arr = _unpack_int4(packed, norms, out, in_dim, tensor['bits'], tensor['group'])
            state_dict[name] = torch.from_numpy(arr.astype(np.float32))
        
        elif tensor['dtype'] == DTYPE_RAW:
            state_dict[name] = tensor['blob']
    
    # Extract LoRA weights
    for key in list(state_dict.keys()):
        if 'lora' in key:
            lora_dict[key] = state_dict.pop(key)
    
    config = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'max_seq_len': max_seq_len,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'weight_bits': weight_bits,
        'act_bits': act_bits,
        'group_size': group_size,
    }
    
    return state_dict, config, lora_dict
