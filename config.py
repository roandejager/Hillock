"""Global configurations and hyperparameter settings for the Exocortex."""

# File and Model Paths
DB_FILE = "exocortex_kg.db"
OLLAMA_MODEL = "qwen2:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# HDC Hyperparameters [28]
HDC_DIMENSION = 10000
HDC_DECAY = 0.95            # Problem 2: Fading memory decay rate
HDC_THRESHOLD = 0.40        # Calibrated gating threshold

# Hebbian Plasticity Hyperparameters [3]
HEBBIAN_ETA = 0.15          # Synaptic learning rate
HEBBIAN_DECAY = 0.01        # Synaptic decay rate

# Parallel Ingestion Settings [1]
BLOCK_SIZE = 15             # Sentences per block
BLOCK_OVERLAP = 2           # Overlapping sentences between blocks
MAX_WORKERS = 4             # Parallel extraction threads (GTX 1070 optimized)