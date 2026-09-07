"""ML resolver with lazy model loading and auto-download.

The T5 model (231 MB) is not bundled with the package.
It is downloaded once from GitHub Releases on first use and cached
in ~/.cache/acrresolv/ for subsequent calls.
"""
import os
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

warnings.filterwarnings("ignore", message="number of unique classes")

_MODEL = None
_TOKENIZER = None

CACHE_DIR = Path(os.environ.get(
    "ACRRESOLV_CACHE_DIR",
    Path.home() / ".cache" / "acrresolv",
))

MODEL_SUBDIR = "t5_acronym"

RELEASE_TAG = "v1.0.0"
REPO = "emiliano-go/acrresolv"
ASSET_NAME = "acrresolv-model.tar.gz"


def _model_dir() -> Path:
    """Return the directory where the trained model lives."""
    return Path(__file__).parent / "models" / MODEL_SUBDIR


def _cached_model_dir() -> Path:
    """Return the user-cache directory for the model."""
    return CACHE_DIR / MODEL_SUBDIR


def _is_model_present(path: Path) -> bool:
    """Check whether a model directory looks complete."""
    return (path / "model.safetensors").exists() and (path / "config.json").exists()


def _download_model() -> Path:
    """Download the model archive from GitHub Releases and extract it.

    Returns the path to the extracted model directory.
    """
    url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / ASSET_NAME
        print(f"Downloading model from {url} ...")
        urlretrieve(url, str(archive_path))

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmp)

        # The archive contains t5_acronym/ at the top level
        extracted = Path(tmp) / MODEL_SUBDIR
        if not extracted.exists():
            raise RuntimeError(
                f"Model archive missing {MODEL_SUBDIR}/ directory"
            )

        dest = _cached_model_dir()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(extracted), str(dest))

    return dest


def _resolve_model_path() -> Path:
    """Find or download the model, returning the directory path.

    Priority:
    1. ACRRESOLV_MODEL_PATH env var (explicit override)
    2. Bundled model (development / editable install)
    3. Cached model in ~/.cache/acrresolv/
    4. Download from GitHub Releases
    """
    # 1. Explicit override
    env_path = os.environ.get("ACRRESOLV_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if _is_model_present(p):
            return p
        raise FileNotFoundError(
            f"ACRRESOLV_MODEL_PATH points to {p} but no valid model found there"
        )

    # 2. Bundled model (source tree)
    bundled = _model_dir()
    if _is_model_present(bundled):
        return bundled

    # 3. Cached model
    cached = _cached_model_dir()
    if _is_model_present(cached):
        return cached

    # 4. Download
    return _download_model()


def _get_model():
    """Lazy load T5 model and tokenizer."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        model_path = _resolve_model_path()
        _MODEL = T5ForConditionalGeneration.from_pretrained(str(model_path))
        _MODEL.eval()

        _TOKENIZER = T5Tokenizer.from_pretrained(str(model_path))
    return _MODEL, _TOKENIZER


def ml_resolve(text: str) -> str:
    """Resolve using ML model.

    Args:
        text: Input text (phrase or acronym)

    Returns:
        Expanded phrase or acronym
    """
    model, tokenizer = _get_model()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    with __import__("torch").no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
