<p align="center">
  <img src="assets/hillock_banner.jpg" alt="Hillock Banner" width="100%">
</p>

# Hillock

**A lightweight, 100% local neuro-symbolic memory engine built for edge hardware.**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![VRAM](https://img.shields.io/badge/VRAM-%3C1.2GB-brightgreen)
![Status](https://img.shields.io/badge/status-v1.0.0_Stable-orange)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white)](https://discord.gg/BGUPNBcVdp)

Traditional local RAG setups are pretty bloated. Running massive vector databases and burning an 8B model just to read documents eats up your VRAM, slows down your machine, and still makes things up when it does not know the answer.

Hillock was built to fix that. Instead of dumping raw text into vector chunks, it uses a simple three-part setup: a local SQLite Knowledge Graph, Hebbian synaptic associations, and Hyperdimensional Computing (HDC) math.

It extracts clean facts, checks if it actually knows the answer using vector math before calling the LLM, and runs comfortably on normal consumer hardware with under 1.2 GB of VRAM.

### Why use Hillock?
* **Zero Guessing:** A hard similarity gate stops questions the database cannot answer. If the data is not there, it says so directly instead of hallucinating.
* **Fast Ingestion:** It uses small, specialized bi-encoders instead of a slow generative model to extract facts. A 30-sentence text file takes roughly 5 seconds.
* **Runs on Low-End Hardware:** Fits under 1.2 GB VRAM and can run purely on your CPU if you do not have an NVIDIA card.
* **OpenAI-Compatible API:** Comes with a built-in FastAPI server so you can hook it right into Open-WebUI, AnythingLLM, or Obsidian.

---

## Quick Start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/) running on your machine.

### Option A: Install via PyPI
```bash
pip install hillock
python -m spacy download en_core_web_sm
```

### Option B: 1-Click Launchers (From Repo)
The setup scripts create a virtual environment, grab the dependencies, and start the CLI automatically.

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

# Install the package locally
pip install -e .

# Download the small SpaCy model for entity parsing
python -m spacy download en_core_web_sm
```

---

## How to Use Hillock

You can interact with Hillock in three different ways depending on what you are building.

### 1. The Local API Server
Start the background server to connect external apps or custom frontends:
```bash
python api.py
```
This runs locally at `http://127.0.0.1:8000`. Point your UI (like AnythingLLM or Open-WebUI) to `http://127.0.0.1:8000/v1` as an OpenAI endpoint.

### 2. The Terminal Console
If you just want to test things out and chat directly with your memory graph:
```bash
python main.py
```
**Built-in Commands:**
* `/ingest [file.txt or .pdf]` : Extract facts from a local file and store them.
* `/alias [target] [canonical]` : Merge two names pointing to the same entity (e.g. `@john` and `john`).
* `/model [name]` : Switch your active Ollama model on the fly.
* `/inspect [entity]` : Check stored facts and synaptic weights for a specific node.
* `/mode [strict | balanced | conversational]` : Adjust how the LLM phrases its answers.

### 3. The Python Library
Use it directly inside your own Python projects:
```python
from engine import IntegratedHillock

# Connect to the local database
engine = IntegratedHillock("my_database.db")

# Query memory directly
answer, primed, traces, mode = engine.execute_chat_turn("Where was Marie Curie born?")
print(answer)
```

---

## Running the Tests

Hillock has a quick CPU-only test suite that validates the database constraints, Hebbian math, and HDC gating without needing a GPU or downloading large model weights:

```bash
python verify_hillock.py
```

## Running the Labyrinth Benchmark

Run the reproducible v1.0.0 multi-hop baseline with an isolated temporary database:

```bash
python benchmark_labyrinth.py
```

The raw results are written to `benchmark_results/labyrinth_v1.0.0.json`. To plot comparisons from saved `labyrinth_v*.json` results (requires optional `matplotlib`):

```bash
python benchmark_labyrinth.py --plot
```

---

## Roadmap: Moving Toward v2.0

With **v1.0.0**, the core engine is locked down and stable: the SQLite graph, the aliasing layer, and the local API are fully set up for real-world projects.

Our next research phase focuses on **v2.0**, moving beyond basic vector matching into deeper logic:

* **Continuous Wave Math (FHRR):** Moving from bipolar vectors to complex-valued phase vectors so we can map soft, continuous relationships much more cleanly.
* **Resonator Networks:** Using closed-loop matrix factorization so multi-hop questions (like "Who is the mother of the founder of company X?") can be solved directly by letting wrong paths cancel out mathematically.
* **The Labyrinth Benchmark:** A tough, synthetic multi-hop test suite to prove that complex logical chains can be solved quickly on a laptop CPU without hallucinations.

---

## Commercial Licensing

Hillock is dual-licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** and a commercial license.

If you want to use Hillock inside a closed-source app, proprietary SaaS, or on-premise setup without open-sourcing your own code under AGPL-3.0, you will need a commercial license.

### Startup Commercial License (Via Polar)
* **AGPL-3.0 Exemption:** Keep your own application code, custom logic, and models private.
* **Commercial Indemnity:** Legal protection to distribute and run Hillock in your products.
* **Direct Support:** Direct contact for setup help, tuning, and architecture advice.

👉 **[Get a Commercial License on Polar ($49/month)](https://buy.polar.sh/polar_cl_yhlaXkyNGFvrco9bYSMvFt6xTRgPkSHpI8cLh30tGLG)**

### Enterprise & Custom Deployments
For custom SLAs, large-scale multi-node setups, or invoice payments:  
✉️ **[Email directly for enterprise licensing](mailto:contact.roandejager@gmail.com?subject=Hillock%20Enterprise%20Licensing)**

---

## License & Contributing

Open-source code is released under the **AGPL-3.0 License**.

To keep the project sustainable while supporting commercial dual-licensing, contributors sign a standard Contributor License Agreement (CLA) via `cla-assistant.io` when opening a pull request. Check out `CONTRIBUTING.md` and `CLA.md` for details.
