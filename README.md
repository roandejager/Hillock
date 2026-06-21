# Hillock 🧠

Hi! This is **Hillock**, a local, personal memory system that integrates symbolic data structures with high-dimensional vector computing. I started hacking on this because standard vector databases always felt way too heavy, expensive, and over-engineered just to run a quick, offline chatbot on my own computer. 

⚠️ **Heads up:** This project is very much a work in progress and honestly, it isn't all that yet. Right now it's a personal, highly experimental research prototype. However, the ultimate ambition is to build a mathematically sound, completely gradient-free cognitive layer for secure, privacy-first local applications.

---

## ⚙️ How It Works (The General Flow)

Here is a quick look at how data moves through the system:

```text
       [Raw Text / PDFs]
               │
               ▼  (Parallel Ingestor)
       [ Ollama (Qwen3) ]
         │            │
         ▼            ▼
    [SQLite Graph]  [Hebbian Memory]
         │            │
         └─────┬──────┘
               ▼
       [VSA/HDC Reservoir] ──► [Gating Controller (Hillock)]
```
*(Note: This ASCII diagram was made with AI, so it might not be 100% correct or perfectly aligned, but it shows the general idea of how things connect.)*

Basically, it splits the work into a few different layers:
* 💾 **SQLite Graph**: Stores the permanent, hard facts as simple triples (like `Marie_Curie` -> `born_in` -> `Poland`) so the system has a solid ground truth.
* ⚡ **Hebbian Plasticity**: Dynamically tracks which entities are being talked about in the chat and strengthens the connections between them, like a simple digital synapse.
* 🌀 **Hyperdimensional Computing (HDC)**: Uses a 10,000-dimensional vector that constantly updates with conversational history, which helps the system resolve pronouns (like "he" or "she") and decide when to block a query to prevent hallucinations.

---

## 📹 Live Interactive Demo (Terminal Cast)

Here is a short terminal recording showing Hillock ingesting a document in real-time, mapping associations, and aggressively blocking an unanswerable question to prevent an LLM hallucination:

![Hillock Gating Demo](demo.gif)

---

## 📊 Scientific Benchmarking Baselines

We evaluated Hillock under our **Long-Form Research Benchmark**, consisting of a highly complex, 30-sentence historical/scientific text and 30 diverse queries (including hard negatives designed to bait hallucinations). 

To ensure absolute scientific honesty and avoid the "evaluation inflation" common in modern AI projects, we run our benchmarks cold on a completely wiped, fresh database. Using a local, heavy **Qwen 3 (5.2B)** model on consumer hardware, here are our exact baseline metrics:

| Metric | Score | Diagnostic Meaning |
| :--- | :---: | :--- |
| **Extraction Precision** | **10.6%** | Percentage of extracted database triples that were perfectly structured. |
| **Extraction Recall** | **22.7%** | Completeness of automatically indexed relations over the 30 complex sentences. |
| **Retrieval Accuracy** | **30.0%** | Exact-string match accuracy on answerable historical queries. |
| **Gate Accuracy** | **30.0%** | Gating success rate (blocking unanswerable queries/hard negatives). |

### The "Qwen 3" Paradox (Why the scores are what they are):
* **The Ingestion Bottleneck**: A 30-sentence dense academic text is highly complex. A local model easily gets confused by multi-clause grammar. It extracted noisy relations like `[Grace_Hopper] -[became_a_pioneer]-> [developed_the_first_compiler]` instead of a clean `[Grace_Hopper] -[developed]-> [compiler]`.
* **The expressiveness penalty**: Interestingly, Qwen 3 actually performed *worse* on paper than Qwen 2 (1.5B). This is because Qwen 3 is *too* smart and expressive. Instead of extracting rigid, simple triples like `[Marie_Curie] -[born_in]-> [Poland]`, it extracted beautifully natural, historically accurate triples like `[Marie_Curie] -[spent_childhood_in]-> [Poland]`. The strict, exact-string evaluation harness penalized this, proving how rigid standard AI benchmarks are, and why we need flexible semantic path matching in future versions.
* **Stable Vector Normalization**: Despite the small model extraction noise, the HDC Semantic Matcher itself is mathematically highly stable. By keeping all candidate facts strictly bound to exactly 3 unique components (Subject, Object, and best-matching Predicate word), we prevent shorter facts from having artificially higher similarity scores.

---

## 🚀 Quick Start (How to run it)

If you want to try running this prototype, it is highly recommended to set up a clean Python virtual environment so you do not mess up your global packages. You will also need [Ollama](https://ollama.com/) installed and running locally.

### 1. Clone and Navigate
```bash
git clone https://github.com/roandejager/Hillock.git
cd Hillock
```

### 2. Set Up Virtual Environment
```bash
# Create the environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies & Pull Model
```bash
pip install -r requirements.txt
ollama pull qwen3:latest
```

### 4. Start the Chat Console
```bash
python main.py
```

Inside the console, you can use these commands:
* `/ingest [filepath]` — Index a local `.txt` or `.pdf` file.
* `/mode [strict/balanced/conversational]` — Change how conversational the AI is.
* `/reset` — Wipe the SQLite database and reset the HDC memory space.

---

## ⚖️ Licensing & Contributions

Hillock is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

To preserve our ability to dual-license the project commercially in the future (for companies who cannot open-source their code under AGPL restrictions) while keeping the core project open and free for hobbyists, all contributors must sign our **Contributor License Agreement (CLA)**.

Our automated bot (via `cla-assistant.io`) will automatically guide you through signing the CLA when you open a Pull Request. For details, see `CONTRIBUTING.md` and `CLA.md`.

---

## 📂 File Reference

* `config.py` — Holds all the hyperparameters (HDC dimensions, decay rates, etc.).
* `database.py` — The SQLite interface for symbolic fact storage.
* `ingestor.py` — Spawns parallel worker threads to chunk and parse documents.
* `plasticity.py` — Tracks Hebbian co-activation weights between concepts.
* `reservoir.py` — The vector symbolic architecture context math.
* `main.py` — Orchestrates the console loop, pronoun resolution, and gating.
* `evaluate_hillock_PROTO_ish.py` — The automated long-form evaluation script.