from __future__ import annotations

import hashlib
import re

import torch
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+-]+")


class HashingTextEncoder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._hash_cache: dict[str, tuple[int, float]] = {}

    def encode(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.dim, dtype=torch.float32)
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            # Use cache for repeated tokens
            if token in self._hash_cache:
                bucket, sign = self._hash_cache[token]
            else:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                self._hash_cache[token] = (bucket, sign)
            vector[bucket] += sign

        return F.normalize(vector, dim=0)


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SemanticTextEncoder:
    """Semantic sentence embeddings with a deterministic hashing fallback.

    Uses sentence-transformers when available (defaulting to a multilingual model
    so cross-lingual descriptions such as "binary search" and "二分探索" land near
    each other). When the library or model weights are unavailable -- offline CI,
    minimal installs -- it transparently falls back to :class:`HashingTextEncoder`
    so callers keep a working, deterministic encoder.

    All vectors are L2-normalized, matching the cosine-similarity contract that
    :class:`~true_lora.adapter.AdapterBank` relies on.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        fallback_dim: int = 256,
    ) -> None:
        self._model = None
        self._fallback: HashingTextEncoder | None = None
        self._cache: dict[str, torch.Tensor] = {}
        self.requested_model = model_name
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
            self.model_name: str | None = model_name
        except Exception:
            # Library missing or weights cannot be fetched -> stay usable offline.
            self._fallback = HashingTextEncoder(dim=fallback_dim)
            self.dim = fallback_dim
            self.backend = "hashing-fallback"
            self.model_name = None

    def encode(self, text: str) -> torch.Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached.clone()

        if self._model is not None:
            vector = self._model.encode(text, convert_to_numpy=False, normalize_embeddings=True)
            vector = torch.as_tensor(vector, dtype=torch.float32).detach().cpu()
            vector = F.normalize(vector, dim=0)
        else:
            assert self._fallback is not None
            vector = self._fallback.encode(text)

        self._cache[text] = vector
        return vector.clone()
