"""Field difficulty router — classifies output fields by retrieval complexity.

NONE   → care_info: no RAG needed, taken directly from scraper
SIMPLE → description, habitat, etc.: single-query vector retrieval
COMPLEX → etymology, cultural_info, fun_fact: multi-query + optional HyDE
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FieldDifficulty(Enum):
    NONE = "none"       # care_info — no RAG needed
    SIMPLE = "simple"   # description, wiki_description, habitat, petal_color_hex
    COMPLEX = "complex" # etymology, cultural_info, fun_fact


@dataclass
class FieldConfig:
    difficulty: FieldDifficulty
    sources: list[str] = field(default_factory=list)  # [] means no RAG
    top_k: int = 3
    use_hyde: bool = False


# "web_ddg" matches any source starting with "web_ddg" (web_ddg_0, web_ddg_1, …)
# "all" means no source filter — search across all sources for this flower.
FIELD_CONFIG: dict[str, FieldConfig] = {
    "description": FieldConfig(FieldDifficulty.SIMPLE, sources=["wikipedia", "pfaf"], top_k=3),
    "wiki_description": FieldConfig(FieldDifficulty.SIMPLE, sources=["wikipedia"], top_k=2),
    "habitat": FieldConfig(
        FieldDifficulty.SIMPLE, sources=["pfaf", "gbif", "wikipedia"], top_k=3,
    ),
    "care_info": FieldConfig(FieldDifficulty.NONE),
    "petal_color_hex": FieldConfig(
        FieldDifficulty.SIMPLE, sources=["wikipedia", "web_ddg"], top_k=2,
    ),
    "etymology": FieldConfig(
        FieldDifficulty.COMPLEX, sources=["all"], top_k=5, use_hyde=True,
    ),
    "cultural_info": FieldConfig(
        FieldDifficulty.COMPLEX, sources=["all"], top_k=5, use_hyde=True,
    ),
    "fun_fact": FieldConfig(
        FieldDifficulty.COMPLEX, sources=["all"], top_k=5, use_hyde=False,
    ),
}
