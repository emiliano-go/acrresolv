"""Hash table lookup for acronym expansion.

Provides O(1) exact match lookup for known phrase-acronym pairs.
Supports case-insensitive matching and fuzzy matching.
"""
import json
from pathlib import Path
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

_DATA_PATH = Path(__file__).parent / "data.json"


# ============================================================================
# Hash Table
# ============================================================================

class LookupTable:
    """Hash table for acronym lookup."""
    
    def __init__(self):
        self._forward = {}  # phrase -> acronym
        self._reverse = {}  # acronym -> phrase
        self._loaded = False
    
    def load(self, data_path: Optional[Path] = None):
        """Load data from JSON file.
        
        Args:
            data_path: Path to data.json file
        """
        if data_path is None:
            data_path = _DATA_PATH
        
        if not data_path.exists():
            return
        
        with open(data_path) as f:
            data = json.load(f)
        
        for phrase, acronym in data:
            self.add(phrase, acronym)
        
        self._loaded = True
    
    def add(self, phrase: str, acronym: str):
        """Add a phrase-acronym pair.
        
        Args:
            phrase: Full phrase
            acronym: Acronym
        """
        # Normalize
        phrase_lower = phrase.lower().strip()
        acronym_upper = acronym.upper().strip()
        
        # Store both directions
        self._forward[phrase_lower] = acronym_upper
        self._reverse[acronym_upper] = phrase_lower
    
    def get_acronym(self, phrase: str) -> Optional[str]:
        """Get acronym for a phrase.
        
        Args:
            phrase: Full phrase
        
        Returns:
            Acronym if found, None otherwise
        """
        phrase_lower = phrase.lower().strip()
        return self._forward.get(phrase_lower)
    
    def get_phrase(self, acronym: str) -> Optional[str]:
        """Get phrase for an acronym.
        
        Args:
            acronym: Acronym
        
        Returns:
            Phrase if found, None otherwise
        """
        acronym_upper = acronym.upper().strip()
        return self._reverse.get(acronym_upper)
    
    def contains_phrase(self, phrase: str) -> bool:
        """Check if a phrase exists in the table."""
        return phrase.lower().strip() in self._forward
    
    def contains_acronym(self, acronym: str) -> bool:
        """Check if an acronym exists in the table."""
        return acronym.upper().strip() in self._reverse
    
    def get_all_phrases(self) -> list[str]:
        """Get all phrases in the table."""
        return list(self._forward.keys())
    
    def get_all_acronyms(self) -> list[str]:
        """Get all acronyms in the table."""
        return list(self._reverse.keys())
    
    def __len__(self) -> int:
        """Get number of entries."""
        return len(self._forward)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists (phrase or acronym)."""
        return (key.lower().strip() in self._forward or
                key.upper().strip() in self._reverse)


# ============================================================================
# Fuzzy Matching
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertion
            insertions = prev_row[j + 1] + 1
            # Deletion
            deletions = curr_row[j] + 1
            # Substitution
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def fuzzy_lookup(table: LookupTable, query: str,
                 max_distance: int = 2) -> Optional[tuple[str, str, int]]:
    """Find closest match using fuzzy matching.
    
    Args:
        table: Lookup table
        query: Query string (phrase or acronym)
        max_distance: Maximum Levenshtein distance
    
    Returns:
        Tuple of (matched_key, value, distance) or None
    """
    query_lower = query.lower().strip()
    query_upper = query.upper().strip()
    
    best_match = None
    best_distance = max_distance + 1
    
    # Check phrases
    for phrase in table.get_all_phrases():
        distance = levenshtein_distance(query_lower, phrase)
        if distance < best_distance:
            best_distance = distance
            best_match = (phrase, table.get_acronym(phrase), distance)
    
    # Check acronyms
    for acronym in table.get_all_acronyms():
        distance = levenshtein_distance(query_upper, acronym)
        if distance < best_distance:
            best_distance = distance
            best_match = (acronym, table.get_phrase(acronym), distance)
    
    if best_match and best_distance <= max_distance:
        return best_match
    
    return None


# ============================================================================
# Singleton Instance
# ============================================================================

_table = None


def get_table() -> LookupTable:
    """Get singleton lookup table instance."""
    global _table
    if _table is None:
        _table = LookupTable()
        _table.load()
    return _table


def lookup_phrase(phrase: str) -> Optional[str]:
    """Look up acronym for a phrase.
    
    Args:
        phrase: Full phrase
    
    Returns:
        Acronym if found, None otherwise
    """
    table = get_table()
    return table.get_acronym(phrase)


def lookup_acronym(acronym: str) -> Optional[str]:
    """Look up phrase for an acronym.
    
    Args:
        acronym: Acronym
    
    Returns:
        Phrase if found, None otherwise
    """
    table = get_table()
    return table.get_phrase(acronym)


def fuzzy_find(query: str, max_distance: int = 2) -> Optional[tuple[str, str, int]]:
    """Find closest match using fuzzy matching.
    
    Args:
        query: Query string
        max_distance: Maximum edit distance
    
    Returns:
        Tuple of (matched_key, value, distance) or None
    """
    table = get_table()
    return fuzzy_lookup(table, query, max_distance)


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_data(data_path: Optional[Path] = None) -> list[tuple[str, str]]:
    """Load phrase-acronym pairs from JSON file.
    
    Args:
        data_path: Path to data.json file
    
    Returns:
        List of (phrase, acronym) pairs
    """
    if data_path is None:
        data_path = _DATA_PATH
    
    if not data_path.exists():
        return []
    
    with open(data_path) as f:
        return [tuple(pair) for pair in json.load(f)]


def save_data(data: list[tuple[str, str]], data_path: Optional[Path] = None):
    """Save phrase-acronym pairs to JSON file.
    
    Args:
        data: List of (phrase, acronym) pairs
        data_path: Path to data.json file
    """
    if data_path is None:
        data_path = _DATA_PATH
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
