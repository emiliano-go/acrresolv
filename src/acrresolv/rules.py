"""Rules engine for acronym expansion.

Provides deterministic, rule-based acronym expansion for common patterns.
Covers ~53% of training data with high confidence.
"""
import re
from typing import Optional

from .data_loader import load_set_cached, load_lookup_table_cached


# ============================================================================
# Stop Words (loaded from JSON)
# ============================================================================

STOP_WORDS = load_set_cached("stop_words.json")

# Prepositions that sometimes ARE included in acronyms
# (e.g., "return on assets" -> ROA includes 'O' from "on")
# These are NOT removed by default when include_prepositions=True
PREPOSITION_WORDS = load_set_cached("preposition_words.json")


# ============================================================================
# Standard Abbreviations (loaded from JSON)
# ============================================================================

# Common multi-word abbreviations that don't follow first-letter rules
STANDARD_ABBREVIATIONS = load_lookup_table_cached("standard_abbreviations.json")


# ============================================================================
# Word Cleaning
# ============================================================================

def clean_word(word: str) -> str:
    """Clean a word by removing non-alphabetic characters."""
    return re.sub(r'[^a-zA-Z]', '', word).lower()


def is_stop_word(word: str) -> bool:
    """Check if a word is a stop word."""
    return clean_word(word) in STOP_WORDS


def get_initials(words: list[str], remove_stopwords: bool = True) -> list[str]:
    """Extract first letter from each word."""
    initials = []
    for word in words:
        cleaned = clean_word(word)
        if not cleaned:
            continue
        if remove_stopwords and is_stop_word(word):
            continue
        if cleaned[0].isalpha():
            initials.append(cleaned[0].upper())
    return initials


# ============================================================================
# Rules for Acronym Generation
# ============================================================================

class Rule:
    """Base class for acronym generation rules."""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
    
    def match(self, words: list[str]) -> bool:
        """Check if this rule matches the input words."""
        raise NotImplementedError
    
    def apply(self, words: list[str]) -> Optional[str]:
        """Apply the rule to generate an acronym."""
        raise NotImplementedError
    
    def confidence(self, words: list[str], acronym: str) -> float:
        """Calculate confidence score for this rule."""
        return 0.8


class FirstLetterRule(Rule):
    """Rule: Take first letter of each word (excluding stop words).
    
    Example: "central processing unit" -> "CPU"
    """
    
    def __init__(self):
        super().__init__("first_letter", priority=10)
    
    def match(self, words: list[str]) -> bool:
        return len(words) >= 2
    
    def apply(self, words: list[str]) -> Optional[str]:
        initials = get_initials(words, remove_stopwords=True)
        if initials:
            return ''.join(initials)
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        # High confidence for 2-6 letter acronyms
        if 2 <= len(acronym) <= 6:
            return 0.9
        return 0.7


class FirstTwoLettersRule(Rule):
    """Rule: Take first two letters of each word.
    
    Example: "central processing" -> "CEPR"
    """
    
    def __init__(self):
        super().__init__("first_two_letters", priority=8)
    
    def match(self, words: list[str]) -> bool:
        return len(words) >= 2
    
    def apply(self, words: list[str]) -> Optional[str]:
        initials = []
        for word in words:
            cleaned = clean_word(word)
            if len(cleaned) >= 2 and not is_stop_word(word):
                initials.append(cleaned[:2].upper())
        if initials:
            return ''.join(initials)[:6]  # Limit to 6 chars
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.7


class FirstAndLastRule(Rule):
    """Rule: Take first letter of first word and last letter of last word.
    
    Example: "central unit" -> "CU"
    """
    
    def __init__(self):
        super().__init__("first_and_last", priority=6)
    
    def match(self, words: list[str]) -> bool:
        non_stop = [w for w in words if not is_stop_word(w)]
        return len(non_stop) >= 2
    
    def apply(self, words: list[str]) -> Optional[str]:
        non_stop = [w for w in words if not is_stop_word(w) and clean_word(w)]
        if len(non_stop) >= 2:
            first = clean_word(non_stop[0])[0].upper()
            last = clean_word(non_stop[-1])[-1].upper()
            return first + last
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.6


class ConsonantRule(Rule):
    """Rule: Take first consonant of each word.
    
    Example: "central processing" -> "CP"
    """
    
    def __init__(self):
        super().__init__("consonant", priority=4)
    
    def match(self, words: list[str]) -> bool:
        return len(words) >= 2
    
    def apply(self, words: list[str]) -> Optional[str]:
        initials = []
        for word in words:
            cleaned = clean_word(word)
            if not cleaned or is_stop_word(word):
                continue
            # Find first consonant
            for char in cleaned:
                if char not in 'aeiou':
                    initials.append(char.upper())
                    break
        if initials:
            return ''.join(initials)
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.5


class VowelRule(Rule):
    """Rule: Take first vowel of each word.
    
    Example: "central processing" -> "EO"
    """
    
    def __init__(self):
        super().__init__("vowel", priority=2)
    
    def match(self, words: list[str]) -> bool:
        return len(words) >= 2
    
    def apply(self, words: list[str]) -> Optional[str]:
        initials = []
        for word in words:
            cleaned = clean_word(word)
            if not cleaned or is_stop_word(word):
                continue
            # Find first vowel
            for char in cleaned:
                if char in 'aeiou':
                    initials.append(char.upper())
                    break
        if initials:
            return ''.join(initials)
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.4


class StandardAbbreviationRule(Rule):
    """Rule: Match known standard abbreviations (CAPEX, OPEX, ROA, etc.).
    
    These are well-established abbreviations that don't follow first-letter rules.
    Example: "capital expenditure" -> "CAPEX" (not "CE")
    """
    
    def __init__(self):
        super().__init__("standard_abbreviation", priority=100)  # Highest priority
    
    def match(self, words: list[str]) -> bool:
        phrase = ' '.join(words).lower().strip()
        return (phrase,) in STANDARD_ABBREVIATIONS
    
    def apply(self, words: list[str]) -> Optional[str]:
        phrase = ' '.join(words).lower().strip()
        return STANDARD_ABBREVIATIONS.get((phrase,))
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.95  # Very high confidence for known abbreviations


class IncludePrepositionsRule(Rule):
    """Rule: Take first letter of EACH word including prepositions.
    
    Example: "return on assets" -> "ROA" (includes 'O' from "on")
    This handles cases where the standard acronym includes preposition letters.
    """
    
    def __init__(self):
        super().__init__("include_prepositions", priority=9)
    
    def match(self, words: list[str]) -> bool:
        # Only match if there are prepositions that would be removed
        has_preps = any(clean_word(w).lower() in PREPOSITION_WORDS for w in words)
        return len(words) >= 2 and has_preps
    
    def apply(self, words: list[str]) -> Optional[str]:
        initials = []
        for word in words:
            cleaned = clean_word(word)
            if cleaned and cleaned[0].isalpha():
                initials.append(cleaned[0].upper())
        if initials:
            return ''.join(initials)
        return None
    
    def confidence(self, words: list[str], acronym: str) -> float:
        return 0.85


# ============================================================================
# Rule Engine
# ============================================================================

class RulesEngine:
    """Engine for applying rules to generate acronyms."""
    
    def __init__(self):
        self.rules = [
            StandardAbbreviationRule(),  # Highest priority - check known abbreviations first
            FirstLetterRule(),
            IncludePrepositionsRule(),
            FirstTwoLettersRule(),
            FirstAndLastRule(),
            ConsonantRule(),
            VowelRule(),
        ]
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def expand(self, phrase: str, remove_stopwords: bool = True) -> tuple[str, float, str]:
        """Expand a phrase to an acronym using rules.
        
        Args:
            phrase: Input phrase
            remove_stopwords: Whether to remove stop words
        
        Returns:
            Tuple of (acronym, confidence, rule_name)
        """
        words = phrase.split()
        
        if not words:
            return '', 0.0, 'empty'
        
        # Try each rule
        for rule in self.rules:
            if rule.match(words):
                acronym = rule.apply(words)
                if acronym:
                    confidence = rule.confidence(words, acronym)
                    return acronym, confidence, rule.name
        
        # Fallback: first letter of each word
        initials = get_initials(words, remove_stopwords)
        if initials:
            # Single-word phrases: first letter is almost always correct
            if len(words) == 1:
                return ''.join(initials), 0.9, 'fallback_first_letter'
            return ''.join(initials), 0.7, 'fallback_first_letter'
        
        return '', 0.0, 'no_match'
    
    def is_confident(self, confidence: float, threshold: float = 0.8) -> bool:
        """Check if confidence is above threshold."""
        return confidence >= threshold


# ============================================================================
# Singleton Instance
# ============================================================================

_engine = None


def get_engine() -> RulesEngine:
    """Get singleton rules engine instance."""
    global _engine
    if _engine is None:
        _engine = RulesEngine()
    return _engine


def rule_expand(phrase: str, remove_stopwords: bool = True) -> tuple[str, float, str]:
    """Expand a phrase using rules.
    
    Args:
        phrase: Input phrase
        remove_stopwords: Whether to remove stop words
    
    Returns:
        Tuple of (acronym, confidence, rule_name)
    """
    engine = get_engine()
    return engine.expand(phrase, remove_stopwords)


def is_rule_confident(phrase: str, acronym: str, confidence: float) -> bool:
    """Check if rule-based result is confident enough.
    
    Args:
        phrase: Input phrase
        acronym: Generated acronym
        confidence: Confidence score
    
    Returns:
        True if confident enough to use rule-based result
    """
    # Single-letter acronyms are valid for single-word phrases
    words = phrase.split()
    if len(words) == 1 and len(acronym) == 1:
        return confidence >= 0.5
    
    # Check if acronym is reasonable length
    if not (2 <= len(acronym) <= 6):
        return False
    
    # Check if acronym matches first letters pattern
    initials = [w[0].upper() for w in words if w and w[0].isalpha()]
    if ''.join(initials) == acronym:
        return True
    
    # Use confidence threshold
    return confidence >= 0.8
