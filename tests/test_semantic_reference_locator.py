"""Default chunker path of the semantic reference locator (no torch, no hub)."""

import numpy as np
import pytest

from citation_index.core.segmenters.semantic_reference_locator import (
    locate_reference_sections_semantic,
    pre_chunk_text,
)

# Sections must be big enough that neighbours don't fit in one 8192-char chunk.
DOC = (
    "# Introduction\n\n"
    + "This paper discusses social history in the archive. " * 100
    + "\n\n# Methods\n\n"
    + "We read many letters and diaries from the period. " * 100
    + "\n\n# References\n\n"
    + "Smith, J. (2018). Title of the paper. Journal of X, 12(3), 123-145.\n" * 50
    + "Lee, K. (2020). Book Title. Oxford University Press. doi:10.1000/xyz\n"
)


class FakeEmbedClient:
    """Reference-looking texts point one way, everything else the other."""

    def get_embeddings(self, texts, model, timeout):
        def vec(t):
            ref = "Smith" in t or "Find in the document" in t
            return [1.0, 0.0] if ref else [0.0, 1.0]

        return np.array([vec(t) for t in texts], dtype=float)


def test_pre_chunk_text_splits_on_headings():
    texts = [c.text for c in pre_chunk_text(DOC)]
    assert len(texts) == 3
    assert texts[2].startswith("# References")


def test_locate_without_chunker_returns_references():
    out = locate_reference_sections_semantic(DOC, FakeEmbedClient())
    assert out.startswith("# References")
    assert "Lee, K. (2020)" in out
    assert "Introduction" not in out


class FailingEmbedClient:
    def get_embeddings(self, texts, model, timeout):
        raise RuntimeError("Error code: 404")


def test_embedding_failure_raises_instead_of_lexical_fallback():
    with pytest.raises(RuntimeError, match="404"):
        locate_reference_sections_semantic(DOC, FailingEmbedClient())
