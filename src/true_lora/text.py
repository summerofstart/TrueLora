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
