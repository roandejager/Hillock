"""CPU-bound Context Compressor using Vector Symbolic Architectures (VSA)."""

import numpy as np
from typing import Dict, List, Tuple
from config import HDC_DIMENSION, HDC_DECAY


class HyperdimensionalReservoir:
    def __init__(self, d: int = HDC_DIMENSION):
        self.d = d
        self.state = np.zeros(self.d, dtype=np.float64)
        self.codebook: Dict[str, np.ndarray] = {}
        self.vocab_book: Dict[str, np.ndarray] = {}

    def get_or_allocate_hypervector(self, name_id: str, is_vocab_token: bool = False) -> np.ndarray:
        book = self.vocab_book if is_vocab_token else self.codebook
        if name_id not in book:
            vector = np.random.choice([-1, 1], size=self.d).astype(np.int32)
            book[name_id] = vector
        return book[name_id]

    def step(self, token_hv: np.ndarray, decay: float = HDC_DECAY) -> np.ndarray:
        """Correct leaky reservoir update."""
        bound_token = np.roll(self.state, shift=1) * token_hv
        self.state = (decay * self.state) + bound_token
        return self.state

    def get_context_fingerprint(self, top_k: int = 3) -> List[Tuple[str, float]]:
        scores = []
        ctx_norm = np.linalg.norm(self.state)
        if ctx_norm == 0:
            return []

        for entity_id, hv in self.codebook.items():
            hv_norm = np.linalg.norm(hv)
            if hv_norm == 0:
                continue
            similarity = np.dot(self.state, hv) / (ctx_norm * hv_norm)
            scores.append((entity_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]