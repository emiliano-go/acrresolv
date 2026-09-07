"""Tests for rules.py module."""

import pytest

from acrresolv.rules import (
    RulesEngine,
    get_initials,
    is_stop_word,
    clean_word,
    rule_expand,
    is_rule_confident,
)


class TestCleanWord:
    def test_basic_clean(self):
        assert clean_word("hello") == "hello"
        assert clean_word("Hello") == "hello"

    def test_remove_numbers(self):
        assert clean_word("test123") == "test"
        assert clean_word("123test") == "test"

    def test_remove_special_chars(self):
        assert clean_word("test!") == "test"
        assert clean_word("test@#$") == "test"

    def test_empty_string(self):
        assert clean_word("") == ""


class TestIsStopWord:
    def test_common_stop_words(self):
        assert is_stop_word("the") == True
        assert is_stop_word("and") == True
        assert is_stop_word("or") == True
        assert is_stop_word("in") == True
        assert is_stop_word("on") == True

    def test_non_stop_words(self):
        assert is_stop_word("central") == False
        assert is_stop_word("processing") == False
        assert is_stop_word("unit") == False


class TestGetInitials:
    def test_basic_initials(self):
        words = ["central", "processing", "unit"]
        result = get_initials(words)
        assert result == ["C", "P", "U"]

    def test_with_stop_words(self):
        words = ["the", "central", "processing", "unit"]
        result = get_initials(words, remove_stopwords=True)
        assert result == ["C", "P", "U"]

    def test_without_stop_words(self):
        words = ["the", "central", "processing", "unit"]
        result = get_initials(words, remove_stopwords=False)
        assert result == ["T", "C", "P", "U"]

    def test_empty_words(self):
        words = []
        result = get_initials(words)
        assert result == []


class TestRulesEngine:
    def test_first_letter_rule(self):
        engine = RulesEngine()
        acronym, confidence, rule_name = engine.expand("central processing unit")
        assert acronym == "CPU"
        assert confidence >= 0.8
        assert rule_name == "first_letter"

    def test_radar_rules_gives_first_letters(self):
        engine = RulesEngine()
        acronym, confidence, rule_name = engine.expand("radio detection and ranging")
        # Rules engine gives first-letter pattern; RADAR itself is in lookup
        assert acronym == "RDR"
        assert rule_name == "first_letter"

    def test_fing_rules_gives_first_letters(self):
        engine = RulesEngine()
        acronym, confidence, rule_name = engine.expand("Facultad de Ingenieria")
        # Rules engine gives first-letter pattern; FING is in lookup
        assert acronym == "FDI"
        assert rule_name == "first_letter"

    def test_fmed_rules_gives_first_letters(self):
        engine = RulesEngine()
        acronym, confidence, rule_name = engine.expand("Facultad de Medicina")
        # Rules engine gives first-letter pattern; FMED is in lookup
        assert acronym == "FDM"
        assert rule_name == "first_letter"


class TestRuleExpand:
    def test_basic_expand(self):
        acronym, confidence, rule_name = rule_expand("central processing unit")
        assert acronym == "CPU"
        assert confidence >= 0.8

    def test_expand_with_stopwords(self):
        acronym, confidence, rule_name = rule_expand("the united states of america")
        assert acronym == "USA"
        assert confidence >= 0.8


class TestIsRuleConfident:
    def test_confident_result(self):
        assert is_rule_confident("central processing unit", "CPU", 0.9) == True

    def test_not_confident_short(self):
        assert is_rule_confident("test", "T", 0.9) == False

    def test_not_confident_long(self):
        # AVLP is 4 chars which is within 2-6 range, so it IS confident
        assert is_rule_confident("a very long phrase", "AVLP", 0.9) == True
        # But a very short acronym should not be confident
        assert is_rule_confident("hello world", "H", 0.9) == False

    def test_not_confident_low_score(self):
        assert is_rule_confident("test", "T", 0.5) == False
