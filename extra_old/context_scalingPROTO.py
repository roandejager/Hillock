"""
ResFormer Memory Engine: Phase 2
Implements a CPU-driven Echo State Network (ESN) for long-term dialogue compression
integrated with a GPU-simulated Cross-Attention Readout Layer.

This architecture achieves linear O(n) time and constant O(1) space complexity
relative to conversation context length, preventing VRAM overflow.
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ContextScaling")


class EchoStateReservoir:
    """
    CPU-bound Recurrent Reservoir (Echo State Network) to compress history.
    Maintains the Echo State Property to ensure stable, fading memory of past turns.
    """

    def __init__(self, d_emb: int = 64, d_res: int = 128, leak_rate: float = 0.3, spectral_radius: float = 0.9):
        self.d_emb = d_emb  # Dimension of word embeddings
        self.d_res = d_res  # Dimension of reservoir states (hidden size)
        self.leak_rate = leak_rate  # Speed of memory updating (alpha)

        # Initialize the reservoir state to zeros
        self.state = np.zeros(self.d_res)

        # Seed generator for reproducibility in testing
        np.random.seed(42)

        # Fixed, random input projection weights
        self.W_in = np.random.uniform(-0.1, 0.1, (self.d_res, self.d_emb))

        # Fixed, random recurrent weights
        self.W_res = np.random.uniform(-0.5, 0.5, (self.d_res, self.d_res))

        # Scale recurrent weights to guarantee the Echo State Property
        self._ensure_echo_state_property(spectral_radius)

        logger.info(f"Initialized CPU ESN Reservoir (Size: {self.d_res}, Spectral Radius: {spectral_radius})")

    def _ensure_echo_state_property(self, target_radius: float) -> None:
        """
        Scales the recurrent weight matrix so its spectral radius (largest eigenvalue)
        is strictly less than 1.0. This prevents chaotic activation escalation.
        """
        eigenvalues = np.linalg.eigvals(self.W_res)
        current_radius = np.max(np.abs(eigenvalues))

        if current_radius > 0:
            # Rescale the matrix to meet our target stable radius
            self.W_res = self.W_res * (target_radius / current_radius)
            logger.info(f"Recurrent weight matrix scaled to target spectral radius {target_radius:.2f}")
        else:
            logger.warning("Failed to calculate spectral radius. Running with raw weights.")

    def step(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Performs a single recurrent update step on CPU:
        x_t = (1 - alpha) * x_t-1 + alpha * tanh(W_in * u_t + W_res * x_t-1)
        """
        # Ensure input dimensions are compatible
        assert input_vector.shape == (self.d_emb,), "Input vector dimension mismatch."

        # Compute raw reservoir activation
        raw_activation = np.dot(self.W_in, input_vector) + np.dot(self.W_res, self.state)
        new_state = np.tanh(raw_activation)

        # Apply leaking rate (how much of the old state is retained vs new activation)
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state

    def reset(self) -> None:
        """Clears the short-term memory state."""
        self.state = np.zeros(self.d_res)
        logger.info("ESN memory state reset to zero.")


class CrossAttentionReadout:
    """
    Lightweight projection layer simulating the cross-attention handshake
    between CPU history (ESN) and active GPU embeddings (Transformer).
    """

    def __init__(self, d_emb: int = 64, d_res: int = 128):
        self.d_emb = d_emb
        self.d_res = d_res

        # Projection matrices to map reservoir size to semantic embedding size
        # Maps ESN state (d_res) to Keys (K) and Values (V)
        self.W_k = np.random.uniform(-0.2, 0.2, (self.d_emb, self.d_res))
        self.W_v = np.random.uniform(-0.2, 0.2, (self.d_emb, self.d_res))

        # Scale factor for scaled dot-product attention
        self.scale = 1.0 / np.sqrt(self.d_emb)

    def calculate_cross_attention(self, active_embeddings: np.ndarray, reservoir_state: np.ndarray) -> np.ndarray:
        """
        Calculates cross-attention matching:
        Queries (Q) come from active GPU sentence tokens.
        Keys (K) & Values (V) come from projected CPU reservoir states.
        """
        # 1. Project ESN state into semantic Key/Value space (CPU to GPU boundary)
        K = np.dot(self.W_k, reservoir_state)  # Shape: (d_emb,)
        V = np.dot(self.W_v, reservoir_state)  # Shape: (d_emb,)

        # 2. Extract Queries from active sentence embeddings
        # active_embeddings shape: (sequence_length, d_emb)
        Q = active_embeddings  # Shape: (L, d_emb)

        # 3. Calculate Scaled Dot-Product Attention weights
        # Compute similarity between active tokens (Q) and history key (K)
        scores = np.dot(Q, K) * self.scale  # Shape: (L,)

        # Stable Softmax calculation
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        # 4. Multiply attention weights by values (V) to get the context vector
        context_vector = np.outer(attention_weights, V)  # Shape: (L, d_emb)

        # 5. Fuse the active representations with the history context
        fused_representation = active_embeddings + context_vector
        return fused_representation


# --- LOCAL COMPONENT ORCHESTRATOR & DEMO ---
class ResFormerMemoryEngine:
    def __init__(self, d_emb: int = 64, d_res: int = 128):
        self.d_emb = d_emb
        self.reservoir = EchoStateReservoir(d_emb=d_emb, d_res=d_res)
        self.readout = CrossAttentionReadout(d_emb=d_emb, d_res=d_res)

        # Simple local mock embedding dictionary for demo words
        self.vocabulary: Dict[str, np.ndarray] = {}
        self._initialize_mock_vocab()

    def _initialize_mock_vocab(self) -> None:
        """Generates mock embeddings for dialogue simulation."""
        words = ["hello", "system", "user", "remember", "yesterday", "today", "tomorrow", "history", "exocortex"]
        for word in words:
            # Generate a pseudo-random embedding vector for each vocabulary token
            self.vocabulary[word] = np.random.uniform(-1.0, 1.0, self.d_emb)

    def get_sentence_embeddings(self, sentence: str) -> np.ndarray:
        """Converts raw string sentences into an embedding matrix."""
        tokens = re.sub(r"[^\w\s]", "", sentence).lower().split()
        embeddings = []

        for token in tokens:
            if token in self.vocabulary:
                embeddings.append(self.vocabulary[token])
            else:
                # Out of vocabulary default
                embeddings.append(np.random.uniform(-0.1, 0.1, self.d_emb))

        return np.array(embeddings) if embeddings else np.zeros((1, self.d_emb))

    def process_conversation_turn(self, raw_sentence: str) -> Tuple[np.ndarray, np.ndarray]:
        """Processes an incoming dialogue turn through the hybrid cascade."""
        logger.info(f"\n--- Processing Conversation Turn: '{raw_sentence}' ---")

        # Convert sentence string to vector representations
        active_embeddings = self.get_sentence_embeddings(raw_sentence)

        # Step 1: Update the ESN CPU Reservoir for each word token sequentially
        # Represents O(n) linear complexity parsing on CPU
        start_time = time.perf_counter()
        for token_vector in active_embeddings:
            current_reservoir_state = self.reservoir.step(token_vector)
        duration = (time.perf_counter() - start_time) * 1000

        logger.info(f"CPU ESN Reservoir updated sequentially in {duration:.4f} ms")

        # Step 2: Compute GPU-simulated Cross-Attention to fuse past history with present tokens
        fused_states = self.readout.calculate_cross_attention(
            active_embeddings,
            current_reservoir_state
        )
        logger.info("GPU Cross-Attention Handshake successfully completed.")

        return current_reservoir_state, fused_states


import re

if __name__ == "__main__":
    # 1. Initialize Memory Engine
    engine = ResFormerMemoryEngine(d_emb=64, d_res=128)

    # 2. Simulate multi-turn dialogue
    turns = [
        "Hello Exocortex system",
        "Remember history today",
        "Today yesterday tomorrow"
    ]

    # Track reservoir updates over time to see changing contextual memory vectors
    for i, turn in enumerate(turns):
        res_state, fused_state = engine.process_conversation_turn(turn)

        # Sample statistical features of the state
        mean_activation = np.mean(res_state)
        std_activation = np.std(res_state)

        print(f"[TURN {i + 1} RESULTS]:")
        print(f" -> Active Sentence Matrix Shape: {fused_state.shape}")
        print(f" -> Compressed ESN State Mean : {mean_activation:.4f} (SD: {std_activation:.4f})")