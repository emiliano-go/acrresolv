"""Data loader for JSON configuration files.

Provides generic functions to load JSON data from the data/ directory.
"""
import json
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Configuration
# ============================================================================

_DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# Generic Loader
# ============================================================================

def load_json(filename: str, data_dir: Optional[Path] = None) -> Any:
    """Load a JSON file from the data directory.
    
    Args:
        filename: Name of the JSON file (e.g., 'stop_words.json')
        data_dir: Optional custom data directory path
    
    Returns:
        Parsed JSON content (list, dict, etc.)
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if data_dir is None:
        data_dir = _DATA_DIR
    
    filepath = data_dir / filename
    
    with open(filepath) as f:
        return json.load(f)


def load_set(filename: str, data_dir: Optional[Path] = None) -> set:
    """Load a JSON array as a set.
    
    Args:
        filename: Name of the JSON file containing an array
        data_dir: Optional custom data directory path
    
    Returns:
        Set of values from the JSON array
    """
    data = load_json(filename, data_dir)
    return set(data)


def load_dict(filename: str, data_dir: Optional[Path] = None) -> dict:
    """Load a JSON file as a dictionary.
    
    Args:
        filename: Name of the JSON file containing an object
        data_dir: Optional custom data directory path
    
    Returns:
        Dictionary from the JSON object
    """
    data = load_json(filename, data_dir)
    return dict(data)


def load_lookup_table(filename: str, data_dir: Optional[Path] = None) -> dict:
    """Load a JSON dict as a lookup table with tuple keys.
    
    Converts {"phrase": "acronym"} to {("phrase",): "acronym"}
    for use with the StandardAbbreviationRule.
    
    Args:
        filename: Name of the JSON file containing phrase-acronym pairs
        data_dir: Optional custom data directory path
    
    Returns:
        Dictionary with tuple keys for phrase matching
    """
    data = load_json(filename, data_dir)
    return {(phrase,): acronym for phrase, acronym in data.items()}


# ============================================================================
# Cached Loaders
# ============================================================================

_cache: dict[str, Any] = {}


def load_json_cached(filename: str, data_dir: Optional[Path] = None) -> Any:
    """Load a JSON file with caching.
    
    Args:
        filename: Name of the JSON file
        data_dir: Optional custom data directory path
    
    Returns:
        Cached parsed JSON content
    """
    cache_key = f"{filename}:{data_dir}"
    if cache_key not in _cache:
        _cache[cache_key] = load_json(filename, data_dir)
    return _cache[cache_key]


def load_set_cached(filename: str, data_dir: Optional[Path] = None) -> set:
    """Load a JSON array as a set with caching.
    
    Args:
        filename: Name of the JSON file containing an array
        data_dir: Optional custom data directory path
    
    Returns:
        Cached set of values
    """
    cache_key = f"set:{filename}:{data_dir}"
    if cache_key not in _cache:
        _cache[cache_key] = load_set(filename, data_dir)
    return _cache[cache_key]


def load_dict_cached(filename: str, data_dir: Optional[Path] = None) -> dict:
    """Load a JSON file as a dictionary with caching.
    
    Args:
        filename: Name of the JSON file containing an object
        data_dir: Optional custom data directory path
    
    Returns:
        Cached dictionary
    """
    cache_key = f"dict:{filename}:{data_dir}"
    if cache_key not in _cache:
        _cache[cache_key] = load_dict(filename, data_dir)
    return _cache[cache_key]


def load_lookup_table_cached(filename: str, data_dir: Optional[Path] = None) -> dict:
    """Load a lookup table with caching.
    
    Args:
        filename: Name of the JSON file containing phrase-acronym pairs
        data_dir: Optional custom data directory path
    
    Returns:
        Cached dictionary with tuple keys
    """
    cache_key = f"lookup:{filename}:{data_dir}"
    if cache_key not in _cache:
        _cache[cache_key] = load_lookup_table(filename, data_dir)
    return _cache[cache_key]


def clear_cache():
    """Clear the loader cache."""
    _cache.clear()
