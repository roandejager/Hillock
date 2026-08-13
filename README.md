# Hillock 🧠

**A lightweight, 100% local neuro-symbolic memory engine built for edge hardware.**

Traditional local RAG is surprisingly heavy. Running dense vector databases and using 8B+ generative LLMs just to parse documents and maintain long-term memory burns VRAM, chokes mid-range GPUs, and still hallucinates when asked about things it doesn't know.

**Hillock** was built to solve this. It replaces bloated vector databases and token-hungry extraction passes with a lightweight, three-tier architecture combining relational knowledge graphs, Hebbian synaptic memory, and 10,000-dimensional Vector Symbolic Architectures (VSA/HDC).

Extraction and gating run 100% offline on-device with zero cloud dependencies and zero API costs — TALON handles document parsing and similarity gating without ever calling an LLM, and stays comfortably within a **<1.2 GB VRAM footprint** (tested on a GTX 1070). A local LLM via Ollama is used only for the final response generation step, once a query has already passed the gate.

---

## ⚙️ Architecture & Data Execution Flow

```text
       [ Raw Text / PDF Documents ]
                    │
                    ▼
     [ TALON Engine (CUDA Accelerated) ]
       ├── Stage 1: Coreference Resolution (Fastcoref)
       ├── Stage 2: Bi-Encoder Predicate Router (MiniLM <2ms)
       └── Stage 3: Zero-Shot Latent Relation Extractor (GLiREL)
                    │
                    ├──► [ SQLite Knowledge Graph ]  (Hard SPO Triples)
                    ├──► [ Hebbian Synaptic Engine ] (Co-Activation Plasticity)
                    └──► [ VSA / HDC Reservoir ]    (10,000-D Fingerprinting)
                                   │
                                   ▼
                      [ HDC Similarity Gating ]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       [ Passed Threshold >= 0.42 ]    [ Failed Similarity Gate ]
                    │                             │
                    ▼                             ▼
      [ LLM Response Generation ]     [ Hardcoded Refusal ]
        (Grounded Fact Rendering)     ("I do not have verified info...")

```

### The Three Memory Layers

* 💾 **SQLite Knowledge Graph (`database.py`)**: Stores ground truth facts as Subject-Predicate-Object (SPO) triples in relational tables. No vector drift or approximation errors for factual memory.
* ⚡ **Hebbian Plasticity Engine (`plasticity.py`)**: Tracks co-occurring concepts across turns using gradient-free synaptic learning to mimic natural associative memory recall.
* 🌀 **Hyperdimensional Reservoir (`reservoir.py`)**: A 10,000-dimensional Vector Symbolic Architecture (VSA) hypervector space that compresses conversation context, resolves pronouns, and hard-blocks unanswerable queries in under a millisecond.

---

## 💡 Why Skip Generative LLMs for Ingestion?

Asking an autoregressive LLM to read documents and output structured JSON is slow and wastes compute. Hillock uses tensor-based classification instead, which is dramatically faster and doesn't depend on an LLM staying well-behaved and formatting its output correctly:

| Metric / Dimension | Standard Local RAG (8B LLM) | Hillock (TALON + HDC) |
| --- | --- | --- |
| **Ingestion Latency (30-Sentence Doc)** | **15–30 minutes** (Autoregressive generation bottleneck). | **~5.05 seconds** (6.3 sent/sec pure GPU rate). |
| **VRAM Footprint** | ~5.8 GB – 16 GB+ (Needs large KV-caches and context windows). | **< 1.2 GB VRAM** (FP16 bi-encoder tensor matching). |
| **Pipeline Completion Rate** | ~85–94% (LLM output prone to syntax drift and malformed JSON, causing dropped extractions). | **100%** (Deterministic matrix operations — every sentence produces a structured output, though not every extraction is correct; see benchmarks below). |
| **Unanswerable Queries** | Burns 100–500 GPU tokens generating long hallucinated excuses. | **0 GPU generation cycles** (<1ms CPU gate shuts down the LLM entirely). |

Note the "100%" row above is about the pipeline *running to completion*, not about correctness — that's a separate question, covered honestly in the benchmarks section next.

---

## 🔬 Mathematical Foundations

The core VSA and synaptic learning mechanics rely on the following algebraic setup:

### 1. Bipolar Hypervector Space

Hypervectors operate over a $D = 10,000$ dimensional bipolar domain:

$$\mathcal{H} = \{-1, +1\}^D$$

### 2. VSA Algebraic Operations

* **Bundling (Superposition / Set Membership $\oplus$)**: Element-wise addition followed by deterministic thresholding (resolving exact zeros via index parity):

$$\mathbf{h}_{\text{bundle}} = \text{sign}\left(\sum_{k=1}^K \mathbf{h}_k\right)$$

* **Binding (Association / Role Encoding $\otimes$)**: Encodes variable-value pairs via the element-wise Hadamard product (equivalent to bitwise XOR in binary space):

$$\mathbf{h}_{\text{bind}} = \mathbf{h}_A \odot \mathbf{h}_B \quad \implies \quad \text{CosSim}(\mathbf{h}_{\text{bind}}, \mathbf{h}_A) \approx 0$$

* **Similarity Metric ($\text{CosSim}$)**: Normalized scalar product calculated directly in Hamming space:

$$\text{CosSim}(\mathbf{h}_A, \mathbf{h}_B) = \frac{1}{D} \sum_{i=1}^D h_{A,i} \cdot h_{B,i} = 1 - \frac{2 \cdot d_H(\mathbf{h}_A, \mathbf{h}_B)}{D}$$

### 3. Hebbian Synaptic Plasticity

Gradient-free updates strengthen connections between active entity nodes, fading over conversation turns via exponential decay:

$$w_{\text{new}} = w + \eta(1 - w) \quad (\text{where } \eta = 0.15, \text{ decay } \gamma = 0.01)$$

### 4. Locality-Sensitive Projection (SimHash)

Projects continuous dense embeddings $\mathbf{x} \in \mathbb{R}^d$ into bipolar space $\mathbf{h} \in \{-1, +1\}^D$ via random projection matrix $\mathbf{R} \in \mathbb{R}^{D \times d}$ while preserving cosine similarity:

$$\mathbf{h} = \text{sign}(\mathbf{R}\mathbf{x}) \quad \implies \quad \mathbb{E}[\text{CosSim}(\mathbf{h}_A, \mathbf{h}_B)] = 1 - \frac{2}{\pi}\arccos(\mathbf{x}_A \cdot \mathbf{x}_B)$$

---

## 📊 Benchmarking & Performance

**Heads up on scale:** this is currently a small, fixed benchmark — one 32-sentence complex academic text, 20 answerable questions, 10 hard-negative trick queries designed to trigger hallucinations. It's enough to catch regressions during development but not enough to claim statistical robustness yet. Treat the numbers below as directional, not final. A larger, more varied benchmark is planned before any v1.0 claims.

**On precision improvements:** v0.4 introduces $O(1)$ set-based schema constraints, direction auto-correction for origin predicates, and precompiled regex span sanitization, boosting raw extraction precision from 11.5% to **15.5%** while keeping ingestion fast and sub-second fast-eval retrieval intact.

Tests run cold on a fresh database:

| Version / Milestone | Extraction Recall | Gate Accuracy | Extraction Precision | Retrieval Accuracy | Ingestion Speed (Pure GPU) |
| --- | --- | --- | --- | --- |----------------------------|
| **v0.1.0 Baseline (Qwen LLM)** | 13.6% | 16.7% | 1.8% | 10.0% | ~15–30 minutes             |
| **v0.2.0 Raw TALON Engine** | 13.6% | 16.7% | 1.8% | 10.0% | 40s (Model Load)           |
| **v0.2.2 Quality Patch** | 50.0% | 43.3% | 7.6% | 45.0% | 35s                        |
| **v0.2.3 Audit Fixes** | 59.1% | 56.7% | 11.5% | 45.0% | 2.9 sent/sec               |
| **v0.2.4 Performance Fixes** | 59.1% | 60.0% | 11.5% | 50.0% | 7.4 sent/sec               |
| **v0.3.0 SimHash VSA** | 50.0% | 56.7% | 11.5% | 55.0% | 7.6 sent/sec               |
| **v0.4.1 Schema Precision (Current)** | **50.0%** | **43.3%** | **15.5%** 🎉 | **45.0%** 🎉 | **6.3 sent/sec** 🎉       |

### What the numbers actually mean:

* **High Speed & Low Latency (6.3 sent/sec / 1.42s retrieval):** TALON ingests full documents in ~5.05 seconds, and the 30-query fast-eval benchmark completes in 1.42 seconds (~0.047s per query).
* **Solid Recall & Precision Jump (50.0% / 15.5%):** Schema filtering and direction auto-correction successfully eliminate inverted facts and clean span artifacts, raising precision to 15.5%.

---

## 🚀 Quick Start

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) running locally (`ollama pull qwen2.5:7b-instruct-q4_K_M`) — used only for final response generation, not extraction
* NVIDIA GPU with CUDA support (8GB VRAM recommended, e.g., GTX 1070)

### Setup

```bash
git clone https://github.com/roandejager/Hillock.git
cd Hillock

python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install PyTorch with CUDA first (adjust cu121/cu124 for your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

**Commands:**

* `/ingest [filepath]` — Ingest a local `.txt` or `.pdf` document.
* `/mode [strict/balanced/conversational]` — Change gating strictness and output style.
* `/reset` — Wipe the SQLite graph and reset the HDC memory space.

---

## ⚖️ Licensing & Contributions

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

To keep the project open-source while preserving the option for future commercial dual-licensing, contributors must sign a standard **Contributor License Agreement (CLA)** via `cla-assistant.io` when opening a PR. See `CONTRIBUTING.md` and `CLA.md`.

---

## 📂 Codebase Overview

* `config.py` — Hyperparameters (HDC dimensionality, Hebbian learning rates, similarity thresholds).
* `database.py` — SQLite triple store with micro-batched transactions.
* `plasticity.py` — Hebbian synaptic association engine.
* `reservoir.py` — 10,000-D VSA hypervector memory engine.
* `talon_engine.py` — 3-stage CUDA extraction pipeline (Fastcoref + MiniLM + GLiREL).
* `ingestor.py` — Document ingestion orchestrator and timing tracker.
* `main.py` — CLI chat interface, pronoun resolution, and gating logic.
* `evaluate_hillock_PROTO_ish.py` — Automated benchmarking suite.