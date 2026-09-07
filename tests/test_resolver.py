import pytest

from acrresolv.resolver import resolve


class TestResolveInputValidation:
    def test_empty_string(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resolve("   ")

    def test_non_string_int(self):
        with pytest.raises(TypeError, match="Expected str"):
            resolve(123)

    def test_non_string_none(self):
        with pytest.raises(TypeError, match="Expected str"):
            resolve(None)

    def test_non_string_list(self):
        with pytest.raises(TypeError, match="Expected str"):
            resolve(["cpu"])

    def test_non_string_bool(self):
        with pytest.raises(TypeError, match="Expected str"):
            resolve(True)

    def test_strips_whitespace(self):
        result = resolve("  central processing unit  ")
        assert result == "CPU"


class TestResolveKnownAcronyms:
    def test_cpu(self):
        assert resolve("central processing unit") == "CPU"

    def test_ram(self):
        assert resolve("random access memory") == "RAM"

    def test_rom(self):
        assert resolve("read only memory") == "ROM"

    def test_bios(self):
        assert resolve("basic input output system") == "BIOS"

    def test_api(self):
        assert resolve("application programming interface") == "API"

    def test_ai(self):
        assert resolve("artificial intelligence") == "AI"

    def test_ml(self):
        assert resolve("machine learning") == "ML"

    def test_gps(self):
        assert resolve("global positioning system") == "GPS"

    def test_nasa(self):
        assert resolve("national aeronautics and space administration") == "NASA"

    def test_dna(self):
        assert resolve("deoxyribonucleic acid") == "DNA"

    def test_fbi(self):
        assert resolve("federal bureau of investigation") == "FBI"

    def test_cia(self):
        assert resolve("central intelligence agency") == "CIA"

    def test_nato(self):
        assert resolve("north atlantic treaty organization") == "NATO"


class TestResolveNonObviousAcronyms:
    """Acronyms where first-letter-of-each-word doesn't match."""

    def test_radar(self):
        assert resolve("radio detection and ranging") == "RADAR"

    def test_lidar(self):
        assert resolve("light detection and ranging") == "LIDAR"

    def test_sonar(self):
        assert resolve("sound navigation and ranging") == "SONAR"

    def test_scuba(self):
        assert resolve("self contained underwater breathing apparatus") == "SCUBA"

    def test_ibm(self):
        assert resolve("international business machines") == "IBM"

    def test_att(self):
        assert resolve("american telephone and telegraph") == "ATT"

    def test_arms(self):
        assert resolve("advanced risc machines") == "ARM"


class TestResolveAmbiguousAcronyms:
    """Similar phrases with different acronyms."""

    def test_surface_to_air_vs_air_to_surface(self):
        assert resolve("surface to air missile") == "SAM"
        assert resolve("air to surface missile") == "ASM"

    def test_sar_vs_isar(self):
        assert resolve("synthetic aperture radar") == "SAR"
        assert resolve("inverse synthetic aperture radar") == "ISAR"

    def test_vhf_vs_uhf_vs_shf(self):
        assert resolve("very high frequency") == "VHF"
        assert resolve("ultra high frequency") == "UHF"
        assert resolve("super high frequency") == "SHF"

    def test_ins_vs_irs(self):
        assert resolve("inertial navigation system") == "INS"
        assert resolve("inertial reference system") == "IRS"


class TestResolveStopwordToggle:
    def test_with_stopwords_default(self):
        result = resolve("the united states of america")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_without_stopwords(self):
        result = resolve("the united states of america", remove_stopwords=False)
        assert isinstance(result, str)
        assert len(result) > 0


class TestResolveEdgeCases:
    def test_single_word(self):
        result = resolve("radar")
        assert isinstance(result, str)

    def test_two_words(self):
        result = resolve("solid state")
        assert isinstance(result, str)

    def test_very_long_phrase(self):
        result = resolve("multiple independently targetable reentry vehicle")
        assert result == "MIRV"

    def test_acronym_with_numbers(self):
        result = resolve("universal serial bus version two")
        assert isinstance(result, str)

    def test_all_caps_input(self):
        result = resolve("CPU")
        assert isinstance(result, str)
