<p align="center">
  <img src="assets/hillock_banner.jpg" alt="Hillock Banner" width="100%">
</p>

# Hillock

**A lightweight, 100% local neuro-symbolic memory engine built for edge hardware.**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![VRAM](https://img.shields.io/badge/VRAM-%3C1.2GB-brightgreen)
![Status](https://img.shields.io/badge/status-v1.0.2_Stable-orange)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.gg/BGUPNBcVdp)

### TL;DR
Traditional local RAG setups are often bloated and unreliable. Setting up heavy vector databases like Pinecone or Milvus eats up gigabytes of RAM, and when an answer does not exist in your files, nearest-neighbor vector search still returns irrelevant text chunks that force your local LLM to hallucinate.

Hillock replaces that entire setup with a fast SQLite knowledge graph and CPU bit math. It checks whether a factual relationship actually exists before calling the model. If the fact is missing or unsupported, it refuses to guess. It runs completely offline under 1.2 GB VRAM or purely on a normal laptop CPU.

---

### What Makes It Different?

**Factual Gating Before Generation**  
Most guardrails try to catch hallucinations after the model speaks. Hillock puts a mathematical bouncer in front of the model. If a question asks about an entity or relation that is not verified in memory, the gate blocks the prompt so the LLM never gets the chance to make up a story.

**Fast Local Extraction**  
Instead of burning your GPU by running slow 8B generative models to read documents, Hillock uses lightweight encoder models (GLiREL Large and an ONNX-runtime MiniLM router) to extract factual links directly into SQLite. Ingestion runs at dozens of sentences per second on ordinary hardware.

**Built for Potato Hardware**  
The entire memory and retrieval engine runs comfortably on consumer laptop CPUs without needing an NVIDIA card. Memory structures are bit-packed so similarity checks run in microseconds directly inside CPU cache.

**Drop-in OpenAI API**  
Includes a built-in FastAPI server that exposes standard `/v1/chat/completions` endpoints with streaming support. You can plug it straight into AnythingLLM, Open-WebUI, or custom agent scripts without changing your application code.

---

## What It Looks Like in Practice

Here is an actual interactive session running locally in the CLI (`python main.py`):

```text
User > Where was Marie Curie born?

[HDC Gate]: Concept match passed (Cosine: 0.82 | Predicate: born_in)
[Hebbian Priming]: Associated concepts activated: Radioactivity (0.15), Nobel_Prize (0.15)
Hillock (Renderer) > Marie Curie was born in Poland (Source: science_history.txt).

User > What was the name of Marie Curie's dog?

[HDC Gate]: Predicate check failed (No matching 'has_pet' relation in graph)
Hillock > I do not have verified information about that.

User > /inspect Marie_Curie

[INSPECTING ENTITY]: Marie_Curie
  Stored SPO Facts:
   * [Marie_Curie] -[born_in]-> [Poland]
   * [Marie_Curie] -[discovered]-> [Radioactivity]
  Hebbian Synaptic Associations:
   * Associated Concept: 'Radioactivity' Strength: 0.1500
   * Associated Concept: 'Nobel_Prize'    Strength: 0.1500
```

Notice what happened on the second question: instead of pulling general text about Marie Curie's life and letting the LLM invent a pet, the gate stopped the context from being sent and triggered a clean refusal.

---

## Quick Start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/) running locally (optional if running in headless mode).

### Option A: Install via PyPI
```bash
pip install hillock
python -m spacy download en_core_web_sm
```

### Option B: 1-Click Launchers (From Repo)
The setup scripts create a clean virtual environment, install dependencies, and start the console automatically.

**Windows:**
```bat
run.bat
```

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

### Option C: Manual Setup (From Source)
```bash
git clone https://github.com/roandejager/Hillock.git
cd Hillock

python -m venv .venv
source .venv/bin/activate  # on Windows run: .venv\Scripts\activate

pip install -e .
python -m spacy download en_core_web_sm
```

---

## How to Use Hillock

### 1. The Local API Server (For Web UIs & Desktop Apps)
Start the background server to connect external apps:
```bash
python api.py
```
This runs locally at `http://127.0.0.1:8000`. In your UI (like AnythingLLM or Open-WebUI), set the API type to OpenAI and point the base URL to `http://127.0.0.1:8000/v1`.

### 2. The Terminal Console
If you want to test ingestion and chat directly in your shell:
```bash
python main.py
```
**Key Commands:**
* `/ingest [file.txt or .pdf]` : Extract facts from a document into SQLite.
* `/alias [target] [canonical]` : Merge two names pointing to the same node (e.g. `@john` into `john`).
* `/inspect [entity]` : View stored facts and Hebbian weights for a concept.
* `/mode [strict | balanced | conversational]` : Adjust how the model renders answers.
* `/model [name]` : Switch your local Ollama model on the fly.

### 3. Headless / Agent Mode (Pure Python, No LLM Needed)
If you are building an autonomous agent or custom script and just want a fast, deterministic fact database without running an LLM:

```python
from engine import IntegratedHillock

# Connect to memory engine (sub-200MB RAM, zero GPU)
hillock = IntegratedHillock("my_database.db")

# Query verified facts directly
query = "Where was Marie Curie born?"
entities = hillock.link_entities(query)
facts = hillock.kg.get_all_facts_for_entities(entities)
verified = hillock.select_answering_facts(query, facts)

for subject, predicate, target, doc, score in verified:
    print(f"Verified Fact: [{subject}] -[{predicate}]-> [{target}] (Score: {score:.2f})")
```

---

## How It Works Under the Hood

Hillock is built on three cooperating layers rather than a single massive vector index:

**1. The Symbolic Graph (`database.py`)**  
Facts are stored as structured subject-predicate-object triples in SQLite using Write-Ahead Logging (WAL) for fast concurrent access. Single-valued relationships (like birthplaces or capitals) have strict overwrite safeguards, and entity aliasing prevents graph fracturing.

**2. Associative Synaptic Memory (`plasticity.py`)**  
Instead of training neural weights with backpropagation, Hillock tracks concept co-occurrence using biological Hebbian learning. Concepts that are referenced together strengthen their connection, while inactive connections naturally decay over time.

**3. Vector Symbolic Gating (`reservoir.py`)**  
Concepts and relations are mapped into 10,000-bit hypervectors packed into 157 `uint64` words. When a question is asked, Hillock calculates late-interaction MaxSim scores using CPU bitwise XOR and hardware `popcount` instructions in under a millisecond. If the similarity and predicate alignment fail the threshold, the gate shuts down the generation path completely.

---

## Verification & Benchmarks

The repository includes reproducible evaluation tools you can run on your own machine:

* **`python verify_hillock.py`**  
Runs a 21-point CPU-only test suite validating database schema constraints, single-valued overwrites, Hebbian weight strengthening and decay, bit-packing math, and zero hallucination leaks on negative test controls.

* **`python evaluate_hillock_PROTO_ish.py`**  
Runs an unseeded end-to-end evaluation testing 22 answerable factual queries and 10 unanswerable negative controls, measuring retrieval precision, recall, and hard-negative block rates.

* **`python benchmark_labyrinth.py`**  
A synthetic stress harness testing multi-hop relational retrieval across 1 to 7 graph hops on local hardware, recording per-hop accuracy, refusal rates, and P95 latency.

**Standardized Public Benchmarks:**  
We are currently setting up full evaluation runs against recognized industry harnesses (including Open RAG Bench and Enterprise RAG benchmarks) to publish comparative latency, recall, and memory ablation charts directly here on the README ASAP.

---

## Roadmap: Moving Toward v2.0

With **v1.0.2** stable, the baseline memory engine, aliasing layer, and OpenAI API are locked. Our active development on the `rc/v*` branches is building toward **v2.0**:

* **Continuous Wave Math (FHRR):** Moving from discrete bipolar vectors to continuous complex unit phasors ($D=2048$) in `reservoir.py` for smoother semantic composition and L1 CPU cache alignment.
* **The Resonator Cortex:** Implementing iterative wave resonance loops so multi-hop reasoning problems can be solved algebraically in superposition without exponential graph search.
* **The Labyrinth Benchmark:** Expanding our synthetic stress tests to evaluate 5-hop to 20-hop reasoning chains on consumer laptop CPUs against local and cloud LLMs.

---

## Commercial Licensing

Hillock is dual-licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** and a commercial license.

If you want to use Hillock inside a closed-source product, proprietary SaaS, or commercial setup without releasing your application source code under AGPL-3.0, you will need a commercial license.

### Startup Commercial License (Via Polar)
* **AGPL-3.0 Exemption:** Keep your proprietary code, models, and business logic private.
* **Commercial Indemnity:** Legal protection to distribute and run Hillock in commercial products.
* **Direct Support:** Direct access for architecture review and deployment guidance.

👉 **[Get a Commercial License on Polar ($49/month)](https://buy.polar.sh/polar_cl_yhlaXkyNGFvrco9bYSMvFt6xTRgPkSHpI8cLh30tGLG)**

### Enterprise & Custom Deployments
For custom SLAs, high-volume deployments, or invoice billing:  
✉️ **[Email directly for enterprise licensing](mailto:contact.roandejager@gmail.com?subject=Hillock%20Enterprise%20Licensing)**

---

## License & Contributing

Open-source code is released under the **AGPL-3.0 License**.

To keep the project sustainable while supporting commercial dual-licensing, contributors sign a standard Contributor License Agreement (CLA) via `cla-assistant.io` when opening a pull request. Check out `CONTRIBUTING.md` and `CLA.md` for details.
