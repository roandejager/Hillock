"""Global configurations and hyperparameter settings for the Hillock."""

# File and Model Paths
DB_FILE = "hillock_kg.db"
OLLAMA_MODEL = "qwen3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# HDC Hyperparameters
HDC_DIMENSION = 10000
HDC_DECAY = 0.95            # Fading memory decay rate
HDC_THRESHOLD = 0.42        # Calibrated gating threshold

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