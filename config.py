"""Global configurations and hyperparameter settings for the Hillock."""

# Version & File/Model Paths
HILLOCK_VERSION = "1.2.0-dev"
DB_FILE = "hillock_kg.db"
OLLAMA_MODEL = "qwen2:1.5b"
LLM_BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"

# Schema Constraints (v0.9)
SINGLE_VALUED_PREDICATES = {"born_in", "died_in", "place_of_birth", "place_of_death", "capital_of"}

# HDC Hyperparameters (v0.4.1 Recalibrated for SimHash Baseline)
HDC_DIMENSION = 10000
HDC_DECAY = 0.95            # Fading memory decay rate
HDC_THRESHOLD = 0.55        # Recalibrated gating threshold to eliminate hallucination leaks

# GloVe & SimHash Continuous Vector Settings (v0.3 - Neuro-Symbolic Expansion)
GLOVE_PATH = "glove.6B.50d.txt"
GLOVE_MAX_VOCAB = 50000     # Trimmed vocabulary for ~10MB RAM footprint
GLOVE_DIM = 50             # Continuous GloVe vector dimension

# Hebbian Plasticity Hyperparameters
HEBBIAN_ETA = 0.15          # Synaptic learning rate
HEBBIAN_DECAY = 0.01        # Synaptic decay rate

# Parallel Ingestion Settings
BLOCK_SIZE = 5             # Sentences per block
BLOCK_OVERLAP = 2           # Overlapping sentences between blocks
MAX_WORKERS = 1             # Parallel extraction threads (GTX 1070 optimized)