"""Lazy model loading with memory-mapped files inspired by needle's approach.

Provides memory-mapped loading for .acr model files, loading weights
on-demand during first inference and caching in memory afterward.
"""
import mmap
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ============================================================================
# Memory-Mapped File Reader
# ============================================================================

class MMapReader:
    """Memory-mapped file reader for efficient large file access."""
    
    def __init__(self, path: str):
        self.path = Path(path)
        self._file = None
        self._mmap = None
        self._size = self.path.stat().st_size
    
    def open(self):
        """Open file with memory mapping."""
        self._file = open(self.path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        return self
    
    def close(self):
        """Close memory-mapped file."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
        self._mmap = None
        self._file = None
    
    def __enter__(self):
        return self.open()
    
    def __exit__(self, *args):
        self.close()
    
    def read(self, offset: int, size: int) -> bytes:
        """Read bytes at offset."""
        self._mmap.seek(offset)
        return self._mmap.read(size)
    
    @property
    def size(self) -> int:
        return self._size


# ============================================================================
# Tensor Cache
# ============================================================================

class TensorCache:
    """LRU cache for loaded tensors."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}
        self._order = []
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        if key in self._cache:
            # Move to end (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, tensor: torch.Tensor):
        """Put tensor in cache."""
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            oldest = self._order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = tensor
        self._order.append(key)
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._order.clear()


# ============================================================================
# Lazy Model Loader
# ============================================================================

class LazyModelLoader:
    """Lazy loader for .acr model files with memory mapping.
    
    Loads tensor data on-demand from memory-mapped file and caches
    in memory for subsequent access.
    """
    
    def __init__(self, path: str, device: str = 'cpu', cache_size: int = 100):
        """
        Args:
            path: Path to .acr file
            device: Device to load tensors to
            cache_size: Maximum number of tensors to cache
        """
        self.path = Path(path)
        self.device = device
        self.cache = TensorCache(cache_size)
        
        # File structure
        self._reader: Optional[MMapReader] = None
        self._header = None
        self._tensors = []
        self._codebook = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Ensure header and tensor directory are loaded."""
        if self._loaded:
            return
        
        self._reader = MMapReader(self.path).open()
        
        # Parse header
        hdr_fmt = '<4sIIIIIIIIIIIIIII'
        hdr_size = struct.calcsize(hdr_fmt)
        raw = self._reader.read(0, hdr_size)
        hdr = struct.unpack(hdr_fmt, raw)
        
        (magic, version, num_tensors, cb_n, vocab_size, d_model, num_layers,
         num_heads, num_kv_heads, head_dim, max_seq_len, lora_rank, lora_alpha,
         weight_bits, act_bits, group_size) = hdr
        
        assert magic == b'ACR1', f"Invalid magic: {magic}"
        assert version == 1, f"Unsupported version: {version}"
        
        self._header = {
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
        
        # Parse codebook
        off = hdr_size
        raw = self._reader.read(off, cb_n * 4)
        self._codebook = np.frombuffer(raw, np.float32).copy()
        off += cb_n * 4
        
        # Parse tensor directory
        rec_fmt = '<BBHIIIIQQII'
        rec_size = struct.calcsize(rec_fmt)
        
        for _ in range(num_tensors):
            raw = self._reader.read(off, rec_size)
            rec = struct.unpack(rec_fmt, raw)
            off += rec_size
            
            dtype, ndim = rec[0], rec[1]
            shape = tuple(rec[3:3 + ndim])
            offset, nbytes, group, bits = rec[7], rec[8], rec[9], rec[10]
            
            self._tensors.append({
                'dtype': dtype,
                'shape': shape,
                'offset': offset,
                'nbytes': nbytes,
                'group': group,
                'bits': bits,
            })
        
        self._loaded = True
    
    def get_tensor(self, name: str, idx: int) -> torch.Tensor:
        """Get tensor by name or index."""
        self._ensure_loaded()
        
        # Check cache
        cached = self.cache.get(name)
        if cached is not None:
            return cached.to(self.device)
        
        # Find tensor by name or index
        if isinstance(idx, int):
            tensor_info = self._tensors[idx]
        else:
            # Search by name (simplified - in practice would need name mapping)
            tensor_info = self._tensors[idx]
        
        # Read blob from file
        blob = self._reader.read(tensor_info['offset'], tensor_info['nbytes'])
        
        # Dequantize based on dtype
        if tensor_info['dtype'] == 1:  # FP16
            arr = np.frombuffer(blob, np.float16).reshape(tensor_info['shape'])
            tensor = torch.from_numpy(arr.astype(np.float32))
        elif tensor_info['dtype'] == 2:  # FP32
            arr = np.frombuffer(blob, np.float32).reshape(tensor_info['shape'])
            tensor = torch.from_numpy(arr)
        elif tensor_info['dtype'] == 3:  # INT4
            # Dequantize INT4
            out, in_dim = tensor_info['shape']
            group = tensor_info['group']
            bits = tensor_info['bits']
            in_pad = (in_dim + group - 1) // group * group
            
            n_packed = out * in_pad * bits // 8
            packed = np.frombuffer(blob[:n_packed], np.uint8).reshape(out, -1)
            norms = np.frombuffer(blob[n_packed:], np.float16).reshape(
                out, in_pad // group
            )
            
            # Dequantize
            from .export import _unpack_int4
            arr = _unpack_int4(packed, norms, out, in_dim, bits, group)
            tensor = torch.from_numpy(arr.astype(np.float32))
        else:
            raise ValueError(f"Unknown dtype: {tensor_info['dtype']}")
        
        # Cache and return
        self.cache.put(name, tensor)
        return tensor.to(self.device)
    
    def get_config(self) -> dict:
        """Get model configuration."""
        self._ensure_loaded()
        return self._header.copy()
    
    def get_num_tensors(self) -> int:
        """Get number of tensors."""
        self._ensure_loaded()
        return len(self._tensors)
    
    def close(self):
        """Close reader and clear cache."""
        if self._reader:
            self._reader.close()
        self.cache.clear()
        self._loaded = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# ============================================================================
# Lazy Model Wrapper
# ============================================================================

class LazyModel:
    """Wrapper for lazy-loaded T5 model."""
    
    def __init__(self, loader: LazyModelLoader):
        self.loader = loader
        self._config = None
    
    @property
    def config(self) -> dict:
        if self._config is None:
            self._config = self.loader.get_config()
        return self._config
    
    def get_encoder_layer(self, layer_idx: int) -> dict:
        """Get encoder layer weights."""
        return {
            'self_attn': {
                'q_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.q_proj', layer_idx * 4),
                'k_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.k_proj', layer_idx * 4 + 1),
                'v_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.v_proj', layer_idx * 4 + 2),
                'out_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.out_proj', layer_idx * 4 + 3),
            },
            'norm_in': self.loader.get_tensor(f'layer{layer_idx:02d}.norm_in', layer_idx * 4 + 4),
            'ffn': {
                'wi_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.wi_proj', layer_idx * 4 + 5),
                'wo_proj': self.loader.get_tensor(f'layer{layer_idx:02d}.wo_proj', layer_idx * 4 + 6),
            },
            'norm_ffn': self.loader.get_tensor(f'layer{layer_idx:02d}.norm_ffn', layer_idx * 4 + 7),
        }
    
    def get_decoder_layer(self, layer_idx: int) -> dict:
        """Get decoder layer weights."""
        offset = self.config['num_layers'] * 8
        return {
            'self_attn': {
                'q_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.q_proj', offset + layer_idx * 7),
                'k_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.k_proj', offset + layer_idx * 7 + 1),
                'v_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.v_proj', offset + layer_idx * 7 + 2),
                'out_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.out_proj', offset + layer_idx * 7 + 3),
            },
            'norm_in': self.loader.get_tensor(f'decoder{layer_idx:02d}.norm_in', offset + layer_idx * 7 + 4),
            'cross_attn': {
                'q_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.cross.q_proj', offset + layer_idx * 7 + 5),
                'k_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.cross.k_proj', offset + layer_idx * 7 + 6),
                'v_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.cross.v_proj', offset + layer_idx * 7 + 7),
                'out_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.cross.out_proj', offset + layer_idx * 7 + 8),
            },
            'norm_cross': self.loader.get_tensor(f'decoder{layer_idx:02d}.norm_cross', offset + layer_idx * 7 + 9),
            'ffn': {
                'wi_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.wi_proj', offset + layer_idx * 7 + 10),
                'wo_proj': self.loader.get_tensor(f'decoder{layer_idx:02d}.wo_proj', offset + layer_idx * 7 + 11),
            },
            'norm_ffn': self.loader.get_tensor(f'decoder{layer_idx:02d}.norm_ffn', offset + layer_idx * 7 + 12),
        }
    
    def get_embedding(self) -> torch.Tensor:
        """Get embedding weights."""
        return self.loader.get_tensor('embedding', 0)


# ============================================================================
# Convenience Functions
# ============================================================================

def load_lazy(path: str, device: str = 'cpu') -> LazyModel:
    """Create lazy model loader.
    
    Args:
        path: Path to .acr file
        device: Device to load tensors to
    
    Returns:
        LazyModel instance
    """
    loader = LazyModelLoader(path, device)
    return LazyModel(loader)
