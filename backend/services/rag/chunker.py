"""Recursive text splitter — splits long text into overlapping chunks.

No LangChain. Tries separators in order (paragraph → line → sentence → word),
falling back to the next if a split piece still exceeds chunk_size.
"""
from __future__ import annotations


class RecursiveTextSplitter:
    """Split text into token-approximate chunks at semantic boundaries.

    Token counting uses a word-count heuristic (words ≈ tokens × 0.75), which is
    accurate enough for chunking without a heavy tokenizer dependency.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split(self, text: str) -> list[str]:
        """Return list of chunk strings."""
        chunks = self._split(text.strip(), self.separators)
        return [c for c in chunks if c.strip()]

    # ── internals ─────────────────────────────────────────────────────────────

    def _count(self, text: str) -> int:
        return len(text.split())

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if self._count(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        parts = text.split(sep)

        # If separator didn't split anything, try the next one
        if len(parts) == 1:
            return self._split(text, separators[1:])

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = self._count(part)

            if part_len > self.chunk_size:
                # This piece is too big on its own — flush and recurse
                if current:
                    chunks.append(sep.join(current))
                    current = []
                    current_len = 0
                chunks.extend(self._split(part, separators[1:]))
                continue

            added_len = (self._count(sep) if current else 0) + part_len

            if current_len + added_len > self.chunk_size and current:
                # Flush current chunk, carry overlap into next
                chunks.append(sep.join(current))
                current = self._overlap_tail(current, sep)
                current_len = sum(self._count(p) for p in current)
                if current:
                    current_len += self._count(sep) * (len(current) - 1)

            current.append(part)
            current_len += (self._count(sep) if len(current) > 1 else 0) + part_len

        if current:
            chunks.append(sep.join(current))

        return chunks

    def _overlap_tail(self, parts: list[str], sep: str) -> list[str]:
        """Return the tail of parts that fits within chunk_overlap tokens."""
        tail: list[str] = []
        count = 0
        for p in reversed(parts):
            p_len = self._count(p)
            if count + p_len > self.chunk_overlap:
                break
            tail.insert(0, p)
            count += p_len
        return tail

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by raw word count."""
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - self.chunk_overlap
        return chunks
