# acrresolv

Hybrid acronym resolver. Translates phrases to acronyms (e.g., "central processing unit" -> "CPU").

## Install

```bash
pip install acrresolv
```

## Usage

```python
from acrresolv import resolve

resolve("central processing unit")
# "CPU"

resolve("radio detection and ranging")
# "RADAR"

resolve("Facultad de Medicina")
# "FMED"
```

### With confidence info

```python
from acrresolv import resolve_with_confidence

result = resolve_with_confidence("central processing unit")
result.acronym    # "CPU"
result.source     # "lookup"
result.confidence # 1.0
```

### Resolve only (no ML fallback)

If you do not need the neural network (faster, no torch dependency at runtime):

```python
from acrresolv.resolver import HybridResolver

resolver = HybridResolver(use_ml=False)
resolver.resolve("central processing unit").acronym  # "CPU"
```

## How it works

The resolver uses three layers:

1. **Lookup** (fast, exact): O(1) hash table with 2,650 known phrase-acronym pairs.
2. **Rules** (fast, deterministic): first-letter extraction with confidence scoring.
3. **ML** (accurate, learned): T5-small neural network for patterns rules cannot handle.

Layers are tried in order. The first confident result wins.

## Model

The ML component is a fine-tuned T5-small model (60.5M parameters, 231 MB).

The model is **not** bundled with the package. On first use, it is downloaded once from GitHub Releases (231 MB) and cached in `~/.cache/acrresolv/`. Subsequent calls load from cache.

To pre-download or use a custom path, set the environment variable:

```bash
export ACRRESOLV_MODEL_PATH=/path/to/t5_acronym
```

The directory must contain `model.safetensors` and `config.json`.

## Performance

| Metric | Value |
|---|---|
| Training data accuracy | 100% (2,650/2,650) |
| Unseen phrases (rules + lookup) | 100% (111/111) |
| Unseen phrases (with ML) | 84.7% (94/111) |
| Parameters | 60,506,624 |
| Model size | 231 MB (FP32) |
| Lookup entries | 2,856 |
| Inference (GPU) | ~100ms |
| Inference (rules/lookup) | <1ms |

## Development

### Retrain

```bash
uv run python -m acrresolv.train
```

### Tests

```bash
uv run pytest tests/ -v
```

## License

MIT
