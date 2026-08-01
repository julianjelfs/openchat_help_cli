"""Embeddings with a disk cache.

Vectors are cached in a single ``.npz`` per model, keyed on the chunk's
``content_hash`` (so re-ingesting does not mean re-embedding) and on a text
hash for ad-hoc strings such as queries. At this corpus size the "index" is
just a numpy array — no vector database (see CLAUDE.md non-goals).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_CACHE_DIR = Path("cache")


def text_key(text: str) -> str:
    """Cache key for ad-hoc text (queries). Chunks use their content_hash."""
    return "q:" + hashlib.sha256(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, path: Path):
        self._path = path
        self._vectors: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path) as archive:
                self._vectors = {key: archive[key] for key in archive.files}

    def get(self, key: str) -> np.ndarray | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = vector

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path, **self._vectors)


class OpenAIEmbedder:
    def __init__(
        self,
        client,
        model: str = DEFAULT_EMBED_MODEL,
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self._client = client
        self.model = model
        self._cache = EmbeddingCache(cache_dir / f"embeddings-{model}.npz")

    def embed(self, items: list[tuple[str, str]]) -> np.ndarray:
        """Embed (key, text) pairs, returning one row per item in order.

        Cache hits never touch the API; misses go up in a single batch call.
        """
        missing = [(key, text) for key, text in items if self._cache.get(key) is None]
        if missing:
            response = self._client.embeddings.create(
                model=self.model,
                input=[text for _, text in missing],
            )
            for (key, _), datum in zip(missing, response.data, strict=True):
                self._cache.put(key, np.asarray(datum.embedding, dtype=np.float32))
            self._cache.save()
        return np.stack([self._cache.get(key) for key, _ in items])
