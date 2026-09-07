"""Tests for hybrid resolver."""

import pytest

from acrresolv.resolver import (
    resolve,
    resolve_with_confidence,
    HybridResolver,
    ResolveResult,
)


class TestResolveResult:
    def test_basic_result(self):
        result = ResolveResult("CPU", "rules", 0.9)
        assert result.acronym == "CPU"
        assert result.source == "rules"
        assert result.confidence == 0.9

    def test_str_conversion(self):
        result = ResolveResult("CPU", "rules", 0.9)
        assert str(result) == "CPU"

    def test_repr(self):
        result = ResolveResult("CPU", "rules", 0.9)
        assert "CPU" in repr(result)
        assert "rules" in repr(result)


class TestHybridResolver:
    def test_rules_layer(self):
        resolver = HybridResolver(use_ml=False)
        # CPU is in the lookup table, so it resolves via lookup
        result = resolver.resolve("central processing unit")
        assert result.acronym == "CPU"
        assert result.source == "lookup"
        assert result.confidence == 1.0

    def test_lookup_layer(self):
        resolver = HybridResolver(use_ml=False)
        # Add to lookup table
        resolver._get_lookup_table().add("test phrase", "TP")
        result = resolver.resolve("test phrase")
        assert result.acronym == "TP"
        assert result.source == "lookup"
        assert result.confidence == 1.0

    def test_fallback(self):
        resolver = HybridResolver(use_ml=False)
        result = resolver.resolve("a")
        assert isinstance(result, ResolveResult)
        assert len(result.acronym) > 0


class TestResolve:
    def test_basic_resolve(self):
        result = resolve("central processing unit")
        assert result == "CPU"

    def test_resolve_with_stopwords(self):
        result = resolve("the united states of america")
        assert result == "USA"

    def test_resolve_whitespace(self):
        result = resolve("  central processing unit  ")
        assert result == "CPU"

    def test_resolve_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve("")

    def test_resolve_non_string(self):
        with pytest.raises(TypeError, match="Expected str"):
            resolve(123)


class TestResolveWithConfidence:
    def test_basic_resolve(self):
        result = resolve_with_confidence("central processing unit")
        assert isinstance(result, ResolveResult)
        assert result.acronym == "CPU"
        assert result.source in ["rules", "lookup", "ml"]
        assert 0 <= result.confidence <= 1.0

    def test_rules_confidence(self):
        result = resolve_with_confidence("central processing unit")
        # CPU is in lookup table, so it resolves via lookup
        assert result.source == "lookup"
        assert result.confidence == 1.0


class TestIntegration:
    """Integration tests for hybrid resolver."""

    def test_known_acronyms(self):
        assert resolve("central processing unit") == "CPU"
        assert resolve("random access memory") == "RAM"
        assert resolve("read only memory") == "ROM"

    def test_non_obvious_acronyms(self):
        assert resolve("radio detection and ranging") == "RADAR"
        assert resolve("light detection and ranging") == "LIDAR"

    def test_spanish_acronyms(self):
        assert resolve("Facultad de Ingenieria") == "FING"
        assert resolve("Facultad de Medicina") == "FMED"

    def test_consistency(self):
        r1 = resolve("central processing unit")
        r2 = resolve("central processing unit")
        assert r1 == r2
