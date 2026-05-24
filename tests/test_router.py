"""Unit tests for field difficulty routing config."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.rag.router import FIELD_CONFIG, FieldConfig, FieldDifficulty

# All text fields that require synthesis + care_info and petal_color_hex
_ALL_EXPECTED_FIELDS = {
    "description",
    "fun_fact",
    "wiki_description",
    "habitat",
    "etymology",
    "cultural_info",
    "petal_color_hex",
    "care_info",
}


def test_field_config_complete():
    """Every expected field has a config entry."""
    assert _ALL_EXPECTED_FIELDS.issubset(set(FIELD_CONFIG.keys())), (
        f"Missing fields: {_ALL_EXPECTED_FIELDS - set(FIELD_CONFIG.keys())}"
    )


def test_care_info_is_none_difficulty():
    assert FIELD_CONFIG["care_info"].difficulty == FieldDifficulty.NONE


def test_etymology_is_complex():
    assert FIELD_CONFIG["etymology"].difficulty == FieldDifficulty.COMPLEX


def test_description_is_simple():
    assert FIELD_CONFIG["description"].difficulty == FieldDifficulty.SIMPLE


def test_complex_fields_have_top_k_5():
    for field in ("etymology", "cultural_info", "fun_fact"):
        assert FIELD_CONFIG[field].top_k == 5, f"{field} should have top_k=5"


def test_hyde_only_on_etymology_and_cultural():
    """HyDE is expensive — only etymology and cultural_info use it."""
    assert FIELD_CONFIG["etymology"].use_hyde is True
    assert FIELD_CONFIG["cultural_info"].use_hyde is True
    assert FIELD_CONFIG["fun_fact"].use_hyde is False


def test_none_fields_have_no_sources():
    """NONE-difficulty fields have an empty sources list."""
    assert FIELD_CONFIG["care_info"].sources == []


def test_all_configs_are_field_config_instances():
    for name, cfg in FIELD_CONFIG.items():
        assert isinstance(cfg, FieldConfig), f"{name} is not a FieldConfig"


def test_simple_fields_have_sources():
    """Every SIMPLE field must specify at least one source."""
    for name, cfg in FIELD_CONFIG.items():
        if cfg.difficulty == FieldDifficulty.SIMPLE:
            assert cfg.sources, f"SIMPLE field '{name}' has no sources"
