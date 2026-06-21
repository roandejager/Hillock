"""Global configurations and hyperparameter settings for the Hillock."""

# File and Model Paths
DB_FILE = "hillock_kg.db"
OLLAMA_MODEL = "qwen2:1.5b" # models i have: "qwen3:latest", and "qwen2:1.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# HDC Hyperparameters
HDC_DIMENSION = 10000
HDC_DECAY = 0.95            # Fading memory decay rate
HDC_THRESHOLD = 0.48        # Calibrated gating threshold

# Hebbian Plasticity Hyperparameters
HEBBIAN_ETA = 0.15          # Synaptic learning rate
HEBBIAN_DECAY = 0.01        # Synaptic decay rate

# Parallel Ingestion Settings
BLOCK_SIZE = 3             # Sentences per block
BLOCK_OVERLAP = 1           # Overlapping sentences between blocks
MAX_WORKERS = 1             # Parallel extraction threads (GTX 1070 optimized)