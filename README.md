# Hillock 🧠

Hi! This is **Hillock**, which is basically a local, personal memory system I've been hacking on because standard vector databases always felt way too heavy and complicated just to run a quick, offline chatbot on my own computer. 

⚠️ **Heads up:** This project is very much a work in progress, and honestly, it isn't all that. It's just a fun personal experiment I'm working on to see if we can use brain-inspired math to make local AI memory better. It is definitely not a finished, production-ready product, so expect some clunky parts and weird bugs.

---

## 📊 Quick Performance Baseline

I put this prototype through a really tough, multi-clause scientific benchmark with complex sentence structures and hard negatives. Running a tiny local Qwen 1.5B model, here is how it did:

* **Retrieval Accuracy**: **50.0%** (It retrieved the correct facts for half of the complex queries, while others got blocked or tripped up).
* **Gate Accuracy**: **50.0%** (It successfully blocked half of the unanswerable/hallucinatory queries).

*(For a more detailed technical breakdown of these metrics and why running a tiny 1.5B model on complex grammar is actually quite hard, check out the Benchmark section at the bottom.)*

---

## ⚙️ How It Works (The General Flow)

Here is a quick look at how data moves through the system:

```text
       [Raw Text / PDFs]
               │
               ▼  (Parallel Ingestor)
       [ Ollama (Qwen2) ]
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

## 🚀 Quick Start (How to run it)

If you actually want to try running this clunky prototype, it is highly recommended to set up a clean Python virtual environment so you do not mess up your global packages. You will also need [Ollama](https://ollama.com/) installed and running locally.

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
ollama pull qwen2:1.5b
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

## 📊 Detailed Technical Benchmarks

Here is the exact diagnostic output from the upgraded, highly rigorous evaluation script (`evaluate_hillock_PROTO_ish.py`):

```text
--------------------------------------------------
  * Extraction Precision : 25.0%  (Correctly structured factual nodes)
  * Extraction Recall    : 42.9%  (Completeness of indexed relations)
  * Retrieval Accuracy   : 50.0%  (Factual accuracy on answerable queries)
  * Gate Accuracy        : 50.0%  (Hallucination defense rate)
--------------------------------------------------
```

### Why the scores are what they are:
* **The 25% Extraction Precision & 42.9% Recall**: We moved the evaluation set to highly complex sentences (e.g., *"Einstein, famous for his theoretical work, also worked alongside..."*). A tiny 1.5B parameter model really struggles with this. It extracted weird relations like `[Einstein] -[discovered]-> [theoretical_work]` because it doesn't have the attention span to separate nouns from actions in complex clauses.
* **The "Isaac Newton" Hallucination Leak**: Interestingly, the system successfully extracted `[Isaac_Newton] -[born_in]-> [Woolsthorpe,England]` from the text and answered `"Where was Isaac Newton born?"` correctly. However, because our strict evaluation script expected this query to be blocked (since it wasn't in the original seed database), the script flagged it as a "leak" even though the system technically parsed and retrieved it perfectly!
* **The "Albert Einstein" Leak**: Because the 1.5B model hallucinated that Einstein discovered `"theoretical_work"`, when asked *"What did Albert Einstein discover?"*, the HDC matcher opened the gate and returned that fact.
* **Vector Normalization**: The retriever matching itself is mathematically highly stable. By keeping all candidate facts strictly bound to exactly 3 unique components (Subject, Object, and best-matching Predicate word), we prevent shorter facts from having artificially higher similarity scores.

---

## 📂 File Reference

* `config.py` — Holds all the hyperparameters (HDC dimensions, decay rates, etc.).
* `database.py` — The SQLite interface for symbolic fact storage.
* `ingestor.py` — Spawns parallel worker threads to chunk and parse documents.
* `plasticity.py` — Tracks Hebbian co-activation weights between concepts.
* `reservoir.py` — The vector symbolic architecture context math.
* `main.py` — Orchestrates the console loop, pronoun resolution, and gating.
* `evaluate_hillock_PROTO_ish.py` — The automated evaluation script.

---

## 🤯 Small Fun (ish) Fact
The project is named after the biological *Axon Hillock*—the exact gatekeeper region of a human neuron that sums up electrical signals and decides whether to fire (open the gate) or remain silent (block).