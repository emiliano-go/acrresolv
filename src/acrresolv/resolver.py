"""Hybrid resolver for acronym expansion.

Combines three layers of resolution:
1. Rules engine (fast, deterministic) - 53.3% coverage
2. Lookup table (fast, exact) - 0% coverage
3. ML model (accurate, learned) - 46.7% coverage

Provides graceful degradation from fast rules to accurate ML.
"""
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

RULE_CONFIDENCE_THRESHOLD = 0.8
LOOKUP_CONFIDENCE = 1.0


# ============================================================================
# Resolver Result
# ============================================================================

class ResolveResult:
    """Result of acronym resolution."""
    
    def __init__(self, acronym: str, source: str, confidence: float = 1.0):
        """
        Args:
            acronym: Resolved acronym
            source: Source of resolution ('rules', 'lookup', 'ml')
            confidence: Confidence score (0.0 to 1.0)
        """
        self.acronym = acronym
        self.source = source
        self.confidence = confidence
    
    def __str__(self) -> str:
        return self.acronym
    
    def __repr__(self) -> str:
        return f"ResolveResult({self.acronym!r}, source={self.source!r}, confidence={self.confidence:.2f})"


# ============================================================================
# Hybrid Resolver
# ============================================================================

class HybridResolver:
    """Hybrid resolver combining rules, lookup, and ML."""
    
    def __init__(self, use_ml: bool = True):
        """
        Args:
            use_ml: Whether to use ML model as fallback
        """
        self.use_ml = use_ml
        self._rules_engine = None
        self._lookup_table = None
    
    def _get_rules_engine(self):
        """Lazy load rules engine."""
        if self._rules_engine is None:
            from .rules import get_engine
            self._rules_engine = get_engine()
        return self._rules_engine
    
    def _get_lookup_table(self):
        """Lazy load lookup table."""
        if self._lookup_table is None:
            from .lookup import get_table
            self._lookup_table = get_table()
        return self._lookup_table
    
    def resolve(self, text: str, remove_stopwords: bool = True) -> ResolveResult:
        """Resolve a phrase to an acronym.
        
        Args:
            text: Input text (phrase to expand to acronym)
            remove_stopwords: Whether to remove stop words
        
        Returns:
            ResolveResult with acronym, source, and confidence
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        
        text = text.strip()
        if not text:
            raise ValueError("Input text cannot be empty")
        
        # Layer 1: Try lookup first (fast, exact) - highest priority for known entries
        lookup_table = self._get_lookup_table()
        lookup_result = lookup_table.get_acronym(text)
        
        if lookup_result:
            return ResolveResult(lookup_result, 'lookup', LOOKUP_CONFIDENCE)
        
        # Layer 2: Try rules (fast, deterministic)
        rules_engine = self._get_rules_engine()
        acronym, confidence, rule_name = rules_engine.expand(text, remove_stopwords)
        
        if acronym and confidence >= RULE_CONFIDENCE_THRESHOLD:
            return ResolveResult(acronym, 'rules', confidence)
        
        # Layer 3: Use ML model (accurate, learned)
        if self.use_ml:
            try:
                ml_result = self._ml_resolve(text)
                if ml_result:
                    return ResolveResult(ml_result, 'ml', 0.9)
            except Exception:
                pass
        
        # Fallback to rules even if confidence is low
        if acronym:
            return ResolveResult(acronym, 'rules', confidence)
        
        # Final fallback: return input truncated
        return ResolveResult(text[:6].upper(), 'fallback', 0.3)
    
    def _ml_resolve(self, text: str) -> Optional[str]:
        """Resolve using ML model.
        
        Args:
            text: Input text
        
        Returns:
            Acronym if successful, None otherwise
        """
        try:
            from .ml import ml_resolve
            result = ml_resolve(text)
            
            if result and result.strip():
                return result.strip()
        except Exception:
            pass
        
        return None


# ============================================================================
# Singleton Instance
# ============================================================================

_resolver = None


def get_resolver(use_ml: bool = True) -> HybridResolver:
    """Get singleton resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = HybridResolver(use_ml=use_ml)
    return _resolver


# ============================================================================
# Public API
# ============================================================================

def resolve(text: str, remove_stopwords: bool = True) -> str:
    """Resolve a phrase to an acronym.
    
    This is the main entry point for the package.
    
    Args:
        text: Input text (phrase to expand to acronym)
        remove_stopwords: Whether to remove stop words
    
    Returns:
        Acronym string
    
    Example:
        >>> resolve("central processing unit")
        'CPU'
    """
    resolver = get_resolver()
    result = resolver.resolve(text, remove_stopwords)
    return result.acronym


def resolve_with_confidence(text: str, remove_stopwords: bool = True) -> ResolveResult:
    """Resolve with confidence information.
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stop words
    
    Returns:
        ResolveResult with acronym, source, and confidence
    """
    resolver = get_resolver()
    return resolver.resolve(text, remove_stopwords)
