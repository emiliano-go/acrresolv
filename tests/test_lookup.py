"""Tests for lookup.py module."""

import pytest

from acrresolv.lookup import (
    LookupTable,
    levenshtein_distance,
    fuzzy_lookup,
    get_table,
    lookup_phrase,
    lookup_acronym,
)


class TestLevenshteinDistance:
    def test_same_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_one_edit(self):
        assert levenshtein_distance("hello", "hallo") == 1

    def test_two_edits(self):
        assert levenshtein_distance("hello", "hallo") == 1
        assert levenshtein_distance("hello", "hullo") == 1

    def test_insertion(self):
        assert levenshtein_distance("hello", "helloo") == 1

    def test_deletion(self):
        assert levenshtein_distance("hello", "hell") == 1

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5


class TestLookupTable:
    def test_add_and_get(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        assert table.get_acronym("central processing unit") == "CPU"
        assert table.get_phrase("CPU") == "central processing unit"

    def test_case_insensitive(self):
        table = LookupTable()
        table.add("Central Processing Unit", "CPU")
        assert table.get_acronym("central processing unit") == "CPU"
        assert table.get_acronym("CENTRAL PROCESSING UNIT") == "CPU"

    def test_contains(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        assert table.contains_phrase("central processing unit") == True
        assert table.contains_acronym("CPU") == True
        assert table.contains_phrase("random access memory") == False

    def test_get_all(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        table.add("random access memory", "RAM")
        assert len(table.get_all_phrases()) == 2
        assert len(table.get_all_acronyms()) == 2

    def test_len(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        table.add("random access memory", "RAM")
        assert len(table) == 2

    def test_contains_operator(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        assert "central processing unit" in table
        assert "CPU" in table
        assert "random access memory" not in table


class TestFuzzyLookup:
    def test_exact_match(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        result = fuzzy_lookup(table, "central processing unit", max_distance=0)
        assert result is not None
        assert result[0] == "central processing unit"
        assert result[1] == "CPU"

    def test_close_match(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        result = fuzzy_lookup(table, "central processing unit", max_distance=2)
        assert result is not None

    def test_no_match(self):
        table = LookupTable()
        table.add("central processing unit", "CPU")
        result = fuzzy_lookup(table, "random access memory", max_distance=2)
        assert result is None


class TestSingletonTable:
    def test_get_table(self):
        table = get_table()
        assert table is not None
        assert isinstance(table, LookupTable)

    def test_table_has_data(self):
        table = get_table()
        assert len(table) > 0


class TestLookupFunctions:
    def test_lookup_phrase(self):
        result = lookup_phrase("central processing unit")
        assert result == "CPU"

    def test_lookup_acronym(self):
        result = lookup_acronym("CPU")
        assert result == "central processing unit"

    def test_lookup_not_found(self):
        result = lookup_phrase("nonexistent phrase")
        assert result is None
