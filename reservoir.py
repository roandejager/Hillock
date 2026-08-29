"""
CPU-bound Context Compressor using Vector Symbolic Architectures (VSA).
Integrates Subword Morphology and GloVe SimHash Semantic Projections (v0.3).
"""

import os
import re
import zipfile
import urllib.request
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import HDC_DIMENSION, HDC_DECAY, GLOVE_PATH, GLOVE_MAX_VOCAB, GLOVE_DIM


def load_lightweight_glove(
    glove_path=GLOVE_PATH,
    max_vocab=GLOVE_MAX_VOCAB,
    embedding_dim=GLOVE_DIM
) -> Dict[str, np.ndarray]:
    """
    Downloads (if necessary) and loads a trimmed GloVe dictionary into memory.
    Memory footprint for 50,000 words at 50 dimensions is ~10MB RAM.
    """
    zip_path = "glove.6B.zip"

    # Check if extracted file exists and is valid
    if not os.path.exists(glove_path) or os.path.getsize(glove_path) == 0:
        # Check zip file validity
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extract("glove.6B.50d.txt")
            except zipfile.BadZipFile:
                print("[v0.3 HDC] Corrupted glove.6B.zip detected. Removing bad file...")
                os.remove(zip_path)

        if not os.path.exists(glove_path):
            if not os.path.exists(zip_path):
                print("[v0.3 HDC] Downloading lightweight Stanford GloVe embeddings (50d)...")
                url = "https://nlp.stanford.edu/data/glove.6B.zip"
                urllib.request.urlretrieve(url, zip_path)
            print("[v0.3 HDC] Extracting glove.6B.50d.txt...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract("glove.6B.50d.txt")

    glove_dict = {}
    if os.path.exists(glove_path) and os.path.getsize(glove_path) > 0:
        print(f"[v0.3 HDC] Loading top {max_vocab} words from {glove_path}...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_vocab:
                    break
                parts = line.strip().split(' ')
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                if len(vector) == embedding_dim:
                    glove_dict[word] = vector
        print(f"[v0.3 HDC] Loaded {len(glove_dict)} word vectors into memory (~10MB RAM).")
    return glove_dict

class SubwordHDCEncoder:
    """Extracts subword character n-grams and superposes them into HDC space."""
    def __init__(self, dimension=HDC_DIMENSION, n_gram_range=(3, 4, 5)):
        self.D = dimension
        self.n_gram_range = n_gram_range

    def _hash_ngram(self, ngram_str: str) -> np.ndarray:
        md5_hash = hashlib.md5(ngram_str.encode('utf-8')).digest()
        seed = int.from_bytes(md5_hash[:4], byteorder='little')
        rng = np.random.RandomState(seed)
        return rng.choice([-1, 1], size=self.D).astype(np.int8)

    def encode(self, text: str) -> np.ndarray:
        padded_text = f"#{text.lower().strip()}#"
        accumulated_sum = np.zeros(self.D, dtype=np.int32)
        count = 0
        for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
            for i in range(len(padded_text) - n + 1):
                ngram = padded_text[i:i+n]
                accumulated_sum += self._hash_ngram(ngram)
                count += 1
        if count == 0:
            return np.random.choice([-1, 1], size=self.D).astype(np.int8)

        res = np.sign(accumulated_sum).astype(np.int8)
        res[res == 0] = 1
        return res


class SignRandomProjectionSimHash:
    """Locality-Sensitive Sign Random Projection (SimHash) for dense embeddings."""
    def __init__(self, input_dim=GLOVE_DIM, hdc_dim=HDC_DIMENSION, seed=42):
        self.d = input_dim
        self.D = hdc_dim
        rng = np.random.RandomState(seed)
        self.R = rng.normal(0.0, 1.0, size=(self.D, self.d)).astype(np.float32)

    def project(self, dense_vector: np.ndarray) -> np.ndarray:
        dense_vec = np.asarray(dense_vector, dtype=np.float32)
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm

        projected = np.dot(self.R, dense_vec)
        res = np.sign(projected).astype(np.int8)
        res[res == 0] = 1
        return res


class HyperdimensionalReservoir:
    """
    Reservoir with fading memory (decay) integrating Subword Morphology
    and GloVe SimHash Semantic Projections.
    """
    def __init__(
        self,
        dimension=HDC_DIMENSION,
        decay=HDC_DECAY,
        glove_dict=None,
        glove_dim=GLOVE_DIM,
        seed=42
    ):
        self.D = dimension
        self.decay = decay
        self.glove_dict = glove_dict if glove_dict is not None else {}
        self.glove_dim = glove_dim

        self.subword_encoder = SubwordHDCEncoder(dimension=self.D)
        self.simhash_projector = SignRandomProjectionSimHash(
            input_dim=self.glove_dim,
            hdc_dim=self.D,
            seed=seed
        )
        self.codebook: Dict[str, np.ndarray] = {}
        self.vocab_book: Dict[str, np.ndarray] = {}
        self.state = np.zeros(self.D, dtype=np.float64)

    @property
    def d(self) -> int:
        """Backward compatibility property returning hypervector dimension D."""
        return self.D

    def get_or_allocate_hypervector(self, name_id: str, is_vocab_token: bool = False) -> np.ndarray:
        book = self.vocab_book if is_vocab_token else self.codebook
        if name_id not in book:
            book[name_id] = self.resolve_predicate_hypervector(name_id)
        return book[name_id]

    def _get_glove_dense_vector(self, predicate_str: str) -> Optional[np.ndarray]:
        tokens = [t.lower() for t in re.split(r'[^a-zA-Z0-9]+', predicate_str) if t]
        valid_vectors = []
        for token in tokens:
            if token in self.glove_dict:
                valid_vectors.append(self.glove_dict[token])

        if not valid_vectors:
            return None

        mean_vec = np.mean(valid_vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec

    def resolve_predicate_hypervector(self, predicate_str: str) -> np.ndarray:
        # 1. Morphological Subword Hypervector (Fallback/Structural)
        h_subword = self.subword_encoder.encode(predicate_str)

        # 2. Semantic GloVe SimHash Hypervector (Conceptual)
        dense_vec = self._get_glove_dense_vector(predicate_str)
        if dense_vec is not None:
            h_simhash = self.simhash_projector.project(dense_vec)
            combined_sum = h_simhash.astype(np.int32) + h_subword.astype(np.int32)
            h_final = np.sign(combined_sum).astype(np.int8)
            h_final[h_final == 0] = 1
            return h_final
        else:
            return h_subword

    def step(self, token_hv: np.ndarray, decay: float = None) -> np.ndarray:
        if decay is None:
            decay = self.decay
        bound_token = np.roll(self.state, shift=1) * token_hv
        self.state = (decay * self.state) + token_hv.astype(np.float64) + bound_token
        return self.state

    def get_bipolar_state(self) -> np.ndarray:
        bipolar = np.sign(self.state).astype(np.int8)
        bipolar[bipolar == 0] = 1
        return bipolar

    def cosine_similarity_against_state(self, candidate_predicate_str: str) -> float:
        h_candidate = self.resolve_predicate_hypervector(candidate_predicate_str)
        current_bipolar_state = self.get_bipolar_state()
        dot_prod = np.dot(current_bipolar_state.astype(np.float32), h_candidate.astype(np.float32))
        return float(dot_prod / self.D)

    def hydra_late_interaction_maxsim(self, query_hvs: List[np.ndarray], fact_hvs: List[np.ndarray], tau_early: float = 0.20) -> float:
        """
        HYDRA: Bipolar Late-Interaction MaxSim scoring via Sub-Dimensional Projection Cascade.
        Operates directly in raw Cosine space [-1.0, 1.0] without artificial noise floor inflation.
        """
        if not query_hvs or not fact_hvs:
            return 0.0

        N_q = len(query_hvs)

        Q_full = np.array(query_hvs, dtype=np.float32)
        F_full_T = np.array(fact_hvs, dtype=np.float32).T
        
        D_full = self.D
        D_sub = 2000

        # Stage 1: Fast evaluation on 2,000 dimensions (Raw Cosine scale)
        Q_sub = Q_full[:, :D_sub]
        F_sub_T = F_full_T[:D_sub, :]

        dot_sub = np.dot(Q_sub, F_sub_T)
        max_sims_sub = np.max(dot_sub, axis=1) / D_sub
        score_sub = float(np.mean(max_sims_sub))

        # Early Rejection Gate in raw cosine space
        if score_sub < tau_early:
            return score_sub

        # Stage 2: Full evaluation on 10,000 dimensions
        dot_full = np.dot(Q_full, F_full_T)
        max_sims_full = np.max(dot_full, axis=1) / D_full
        score_full = float(np.mean(max_sims_full))

        return score_full
    
    def permute(self, hv: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Applies positional permutation (cyclic coordinate shift) to a hypervector.
        This breaks commutativity so we can preserve sequential order in multi-hop paths.
        """
        return np.roll(hv, shift=shift)

    def bind_sequential_path(self, entities: List[str], predicates: List[str]) -> np.ndarray:
        """
        HYPERGRAPH-HDC: Binds a multi-hop relational path into a single macro-vector.
        Follows the mathematical formulation: E_0 * R_1 * Pi(E_1) * Pi^2(R_2) * Pi^3(E_2) ...
        """
        if not entities:
            return np.ones(self.D, dtype=np.int8)

        # Start with the anchor entity (E_0)
        e0_hv = self.get_or_allocate_hypervector(entities[0], is_vocab_token=(entities[0] not in self.codebook))
        path_hv = e0_hv.astype(np.int8).copy()
        
        for i in range(len(predicates)):
            # 1. Bind the Predicate (R_{i+1})
            r_hv = self.resolve_predicate_hypervector(predicates[i]).astype(np.int8)
            p_shift = 2 * i
            
            if p_shift > 0:
                r_hv = self.permute(r_hv, shift=p_shift)
            path_hv = path_hv * r_hv
            
            # 2. Bind the next Entity (E_{i+1})
            if i + 1 < len(entities):
                e_next = entities[i + 1]
                e_hv = self.get_or_allocate_hypervector(e_next, is_vocab_token=(e_next not in self.codebook)).astype(np.int8)
                
                e_shift = 2 * i + 1
                e_hv = self.permute(e_hv, shift=e_shift)
                path_hv = path_hv * e_hv
                
        return path_hv

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


# Self-test execution block
if __name__ == "__main__":
    print("=" * 60)
    print("      v0.3 HYPERDIMENSIONAL RESERVOIR SELF-TEST      ")
    print("=" * 60)

    # Setup Mock GloVe dictionary
    mock_glove = {
        "worked": np.random.normal(0, 1, 50).astype(np.float32),
        "alongside": np.random.normal(0, 1, 50).astype(np.float32),
        "collaborated": np.random.normal(0, 1, 50).astype(np.float32),
        "with": np.random.normal(0, 1, 50).astype(np.float32)
    }
    # Make 'worked_alongside' and 'collaborated_with' semantically close in continuous space
    mock_glove["collaborated"] = mock_glove["worked"] + np.random.normal(0, 0.1, 50).astype(np.float32)

    reservoir = HyperdimensionalReservoir(
        dimension=10000,
        decay=0.85,
        glove_dict=mock_glove,
        glove_dim=50
    )

    # Resolve predicate hypervectors
    hv1 = reservoir.resolve_predicate_hypervector("worked_alongside")
    hv2 = reservoir.resolve_predicate_hypervector("collaborated_with")

    # Compute SimHash Hamming similarity
    sim = np.dot(hv1.astype(np.float32), hv2.astype(np.float32)) / 10000.0
    print(f"  * Predicate A : 'worked_alongside'")
    print(f"  * Predicate B : 'collaborated_with'")
    print(f"  * SimHash Bipolar Cosine Similarity: {sim:.4f}")
    print("=" * 60)
    print("Self-test completed successfully!")