# Hillock 🧠

Hi! This is **Hillock**, which is basically a local, personal memory system I've been hacking on because standard vector databases always felt way too heavy and complicated just to run a quick, offline chatbot on my own computer. 

⚠️ **Heads up:** This project is very much a work in progress, and honestly, it isn't all that. It's just a fun personal experiment I'm working on to see if we can use brain-inspired math to make local AI memory better. It is definitely not a finished, production-ready product, so expect some clunky parts and weird bugs.

---

## 📊 Quick Performance Baseline

I put this prototype through a massive, highly rigorous 30-sentence scientific benchmark with complex sentence structures, deep distractors, and tricky "hard negative" queries. Running a tiny local Qwen 1.5B model, here is how it did:

* **Retrieval Accuracy**: **30.0%** (It retrieved the correct facts for some of the highly complex queries, but the tiny model missed others during extraction).
* **Gate Accuracy**: **30.0%** (It successfully blocked many unanswerable/hallucinatory queries, though some leaks occurred due to tiny model extraction errors).

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
  * Extraction Precision : 10.6%  (Correctly structured factual nodes)
  * Extraction Recall    : 22.7%  (Completeness of indexed relations)
  * Retrieval Accuracy   : 30.0%  (Factual accuracy on answerable queries)
  * Gate Accuracy        : 30.0%  (Hallucination defense rate)
--------------------------------------------------
```

### Why the scores are what they are:
* **The 10.6% Extraction Precision & 22.7% Recall**: We pushed the evaluation set to a massive **30 complex, multi-subject sentences** spanning Quantum Physics, Computer Science, Space Exploration, and Philosophy. A tiny 1.5B parameter model (`qwen2:1.5b`) is simply too small to parse this much dense text without getting confused. It hallucinated relationships like `[James_Watson] -[discovered]-> [double-helix_model_of_DNA]` or `[Grace_Hopper] -[became_a_pioneer]-> [developed_the_first_compiler]`.
* **The "Newton / Galileo / Aristotle" Blocks**: Because the 1.5B model failed to parse their clean relations during the parallel ingestion phase, those questions were safely blocked during step 2 (resulting in correct blocks for unanswerable ones but false blocks for answerable ones).
* **The "Edison / Feynman" Leaks**: Because the 1.5B model extracted noisy relations during ingestion (like `[Heinrich_Hertz] -[born_in]-> [Hamburg,_Germany]`), when asked about unmentioned things (like who Hertz collaborated with), the gate opened on the birth fact, resulting in "leaks" under the strict test suite.
* **Vector Normalization**: The retriever matching itself is mathematically highly stable. By keeping all candidate facts strictly bound to exactly 3 unique components (Subject, Object, and best-matching Predicate word), we prevent shorter facts from having artificially higher similarity scores.

---

## 📂 File Reference

* `config.py` — Holds all the hyperparameters (HDC dimensions, decay rates, etc.).
* `database.py` — The SQLite interface for symbolic fact storage.
* `ingestor.py` — Spawns parallel worker threads to chunk and parse documents.
* `plasticity.py` — Tracks Hebbian co-activation weights betwee