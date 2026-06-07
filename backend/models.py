from datetime import datetime
from typing import Any

from database import Base
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship


class Flower(Base):
    __tablename__ = "flowers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    latin_name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    common_name: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="pending")

    # Generated fields (RAG)
    description: Mapped[str | None] = mapped_column(Text)
    fun_fact: Mapped[str | None] = mapped_column(Text)
    wiki_description: Mapped[str | None] = mapped_column(Text)
    habitat: Mapped[str | None] = mapped_column(Text)
    etymology: Mapped[str | None] = mapped_column(Text)
    cultural_info: Mapped[str | None] = mapped_column(Text)
    petal_color_hex: Mapped[str | None] = mapped_column(Text)
    care_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # PFAF ratings
    edibility_rating: Mapped[int | None] = mapped_column(Integer)
    other_uses_rating: Mapped[int | None] = mapped_column(Integer)
    weed_potential: Mapped[str | None] = mapped_column(Text)
    medicinal_rating: Mapped[int | None] = mapped_column(Integer)

    # Image references
    info_image_path: Mapped[str | None] = mapped_column(Text)
    info_image_author: Mapped[str | None] = mapped_column(Text)
    main_image_path: Mapped[str | None] = mapped_column(Text)
    lock_image_path: Mapped[str | None] = mapped_column(Text)

    # Feature date (random, for Flora app)
    feature_year: Mapped[int | None] = mapped_column(Integer)
    feature_month: Mapped[int | None] = mapped_column(Integer)
    feature_day: Mapped[int | None] = mapped_column(Integer)

    confidence_scores: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    wikipedia_url: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    raw_sources: Mapped[list["RawSource"]] = relationship(
        back_populates="flower", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["SourceEmbedding"]] = relationship(
        back_populates="flower", cascade="all, delete-orphan"
    )
    translations: Mapped[list["Translation"]] = relationship(
        back_populates="flower", cascade="all, delete-orphan"
    )


class RawSource(Base):
    __tablename__ = "raw_sources"
    __table_args__ = (UniqueConstraint("flower_id", "source"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    flower_id: Mapped[int] = mapped_column(ForeignKey("flowers.id"))
    source: Mapped[str] = mapped_column(Text, nullable=False)  # pfaf|wikipedia|wikidata|gbif
    raw_content: Mapped[str | None] = mapped_column(Text)
    parsed_content: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    scraped_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    flower: Mapped["Flower"] = relationship(back_populates="raw_sources")
    embeddings: Mapped[list["SourceEmbedding"]] = relationship(back_populates="raw_source")


class SourceEmbedding(Base):
    __tablename__ = "source_embeddings"
    __table_args__ = (
        Index("ix_source_embeddings_hnsw", "embedding", postgresql_using="hnsw",
              postgresql_with={"m": 16, "ef_construction": 64},
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    raw_source_id: Mapped[int] = mapped_column(ForeignKey("raw_sources.id"))
    flower_id: Mapped[int] = mapped_column(ForeignKey("flowers.id"))
    chunk_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    raw_source: Mapped["RawSource"] = relationship(back_populates="embeddings")
    flower: Mapped["Flower"] = relationship(back_populates="embeddings")


class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("flower_id", "language"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    flower_id: Mapped[int] = mapped_column(ForeignKey("flowers.id"))
    language: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    fun_fact: Mapped[str | None] = mapped_column(Text)
    wiki_description: Mapped[str | None] = mapped_column(Text)
    habitat: Mapped[str | None] = mapped_column(Text)
    etymology: Mapped[str | None] = mapped_column(Text)
    cultural_info: Mapped[str | None] = mapped_column(Text)
    source_method: Mapped[str | None] = mapped_column(Text)  # native_wiki|llm_translation|hybrid

    flower: Mapped["Flower"] = relationship(back_populates="translations")
