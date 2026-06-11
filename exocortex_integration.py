"""
Phase 3: Unified Exocortex Orchestrator (Fully Patched - Safe Gate Edition)
Integrates SQLite Knowledge Graphs, Hebbian Co-activation,
and CPU-based Reservoir Context Compression.

Structural Safety:
  - Fixed (Bug 1): Removed noisy random ESN text projections. ESN runs purely as a mathematical trace.
  - Fixed (Bug 2 & 3): Extractor dynamically registers new entities in the DB and Hebbian table.
  - Architectural Gate: Deterministically blocks LLM rendering if no database facts are found.
  - Autonomous Learning: Swaps hallucination failures into a JSON extraction learning loop.
"""

import sqlite3
import re
import numpy as np
import logging
import os
import json
import urllib.request
import urllib.error
from typing import List, Tuple, Dict, Set, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Exocortex")


# ==========================================
# PHASE 1: SYMBOLIC GRAPH & PLASTICITY
# ==========================================

class SQLiteKnowledgeGraph:
    """Manages the decoupled symbolic database (Filing Cabinet) [1]."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")  # Enforce integrity constraints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    source_id TEXT,
                    predicate TEXT,
                    target_id TEXT,
                    PRIMARY KEY (source_id, predicate, target_id),
                    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hebbian_weights (
                    entity_a TEXT,
                    entity_b TEXT,
                    weight REAL DEFAULT 0.0,
                    PRIMARY KEY (entity_a, entity_b),
                    FOREIGN KEY (entity_a) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_b) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def seed_initial_knowledge(self) -> None:
        entities = [
            ("France", "France", "Country"),
            ("Paris", "Paris", "City"),
            ("London", "London", "City"),
            ("UK", "United Kingdom", "Country"),
            ("Marie_Curie", "Marie Curie", "Person"),
            ("Poland", "Poland", "Country"),
            ("Radioactivity", "Radioactivity", "Scientific Field"),
            ("Nobel_Prize", "Nobel Prize", "Award"),
            ("Alan_Turing", "Alan Turing", "Person"),
            ("Enigma", "Enigma Machine", "Machine")
        ]
        relations = [
            ("Paris", "capital_of", "France"),
            ("London", "capital_of", "UK"),
            ("Marie_Curie", "born_in", "Poland"),
            ("Marie_Curie", "discovered", "Radioactivity"),
            ("Marie_Curie", "won", "Nobel_Prize"),
            ("Alan_Turing", "born_in", "London"),
            ("Alan_Turing", "cracked", "Enigma")
        ]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?)", entities)
            cursor.executemany("INSERT OR IGNORE INTO relations VALUES (?, ?, ?)", relations)
            conn.commit()

    def update_relation(self, source_id: str, predicate: str, new_target_id: str,
                        source_type: str = "Generic", target_type: str = "Generic") -> None:
        """Updates relations, registering entities first to satisfy foreign key rules."""
        # Convert names to clean DB keys
        src_key = source_id.strip().replace(" ", "_")
        tgt_key = new_target_id.strip().replace(" ", "_")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Register missing entities first
            cursor.execute(
                "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, ?)",
                (src_key, src_key.replace("_", " "), source_type)
            )
            cursor.execute(
                "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, ?)",
                (tgt_key, tgt_key.replace("_", " "), target_type)
            )

            # Update relation
            cursor.execute("DELETE FROM relations WHERE source_id = ? AND predicate = ?", (src_key, predicate))
            cursor.execute("INSERT OR REPLACE INTO relations VALUES (?, ?, ?)", (src_key, predicate, tgt_key))
            conn.commit()

    def query_relation(self, source_id: str, predicate: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT target_id FROM relations WHERE source_id = ? AND predicate = ?", (source_id, predicate))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_all_entity_ids(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entities")
            return [row[0] for row in cursor.fetchall()]


class HebbianPlasticityEngine:
    """Manages gradient-free co-activation associations (Synaptic Memory) [3]."""
    def __init__(self, db_path: str, eta: float = 0.15, decay: float = 0.01):
        self.db_path = db_path
        self.eta = eta
        self.decay = decay

    def update_associations(self, active_entities: Set[str]) -> None:
        if len(active_entities) < 2:
            self._apply_global_decay()
            return

        sorted_entities = sorted(list(active_entities))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i in range(len(sorted_entities)):
                for j in range(i + 1, len(sorted_entities)):
                    ent_a, ent_b = sorted_entities[i], sorted_entities[j]

                    # Ensure entities exist in Hebbian context before weight writing
                    cursor.execute("SELECT id FROM entities WHERE id = ?", (ent_a,))
                    if not cursor.fetchone():
                        continue
                    cursor.execute("SELECT id FROM entities WHERE id = ?", (ent_b,))
                    if not cursor.fetchone():
                        continue

                    cursor.execute("SELECT weight FROM hebbian_weights WHERE entity_a = ? AND entity_b = ?", (ent_a, ent_b))
                    row = cursor.fetchone()
                    current_w = row[0] if row else 0.0
                    new_w = current_w + self.eta * (1.0 - current_w)
                    cursor.execute("""
                        INSERT INTO hebbian_weights (entity_a, entity_b, weight)
                        VALUES (?, ?, ?)
                        ON CONFLICT(entity_a, entity_b) DO UPDATE SET weight = excluded.weight
                    """, (ent_a, ent_b, new_w))

            # Decay inactive associations
            cursor.execute("SELECT entity_a, entity_b, weight FROM hebbian_weights")
            for ent_a, ent_b, weight in cursor.fetchall():
                if ent_a not in active_entities or ent_b not in active_entities:
                    cursor.execute("UPDATE hebbian_weights SET weight = ? WHERE entity_a = ? AND entity_b = ?",
                                   (weight * (1.0 - self.decay), ent_a, ent_b))
            conn.commit()

    def _apply_global_decay(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE hebbian_weights SET weight = weight * ?", (1.0 - self.decay,))
            conn.commit()

    def get_associated_priming_context(self, entity: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_b, weight FROM hebbian_weights WHERE entity_a = ? AND weight > ?
                UNION
                SELECT entity_a, weight FROM hebbian_weights WHERE entity_b = ? AND weight > ?
                ORDER BY weight DESC
            """, (entity, threshold, entity, threshold))
            return cursor.fetchall()


# ==========================================
# PHASE 2: CONTEXT SCALING (ESN ONLY)
# ==========================================

class EchoStateReservoir:
    """CPU-bound Context Compressor (Maintains continuous state history mathematically) [48]."""
    def __init__(self, d_emb: int, d_res: int, leak_rate: float = 0.3):
        self.d_emb = d_emb
        self.d_res = d_res
        self.leak_rate = leak_rate
        self.state = np.zeros(self.d_res)
        self.W_in = np.random.uniform(-0.1, 0.1, (self.d_res, self.d_emb))
        self.W_res = np.random.uniform(-0.5, 0.5, (self.d_res, self.d_res))
        self._scale_spectral_radius(0.9)

    def _scale_spectral_radius(self, target: float) -> None:
        radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
        if radius > 0:
            self.W_res = self.W_res * (target / radius)

    def step(self, input_vector: np.ndarray) -> np.ndarray:
        raw = np.dot(self.W_in, input_vector) + np.dot(self.W_res, self.state)
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * np.tanh(raw)
        return self.state


# ==========================================
# PHASE 3: SECURE GATE & EXTRACTION ORCHESTRATOR
# ==========================================

class IntegratedExocortex:
    def __init__(self, db_path: str, d_emb: int = 64, d_res: int = 128, ollama_model: str = "qwen2:1.5b"):
        self.d_emb = d_emb
        self.ollama_model = ollama_model
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)
        self.reservoir = EchoStateReservoir(d_emb, d_res)

        # Mapping rules representing structural templates (DecID reasoning paths)
        self.templates = [
            (r"where was (.+?) born", "born_in"),
            (r"what is the capital of (.+)", "capital_of"),
            (r"what did (.+?) discover", "discovered"),
            (r"what did (.+?) win", "won"),
            (r"what did (.+?) crack", "cracked")
        ]

        # Word embedding vocabulary
        self.vocab: Dict[str, np.ndarray] = {}
        np.random.seed(101)
        core_words = ["hello", "exocortex", "where", "what", "is", "the", "capital", "born",
                      "radioactivity", "france", "poland", "london", "uk", "enigma"]
        for w in core_words:
            self.vocab[w] = np.random.uniform(-1.0, 1.0, self.d_emb)

    def _get_sentence_representation(self, sentence: str) -> np.ndarray:
        tokens = re.sub(r"[^\w\s]", "", sentence).lower().split()
        vectors = [self.vocab.get(t, np.random.uniform(-0.1, 0.1, self.d_emb)) for t in tokens]
        return np.array(vectors) if vectors else np.zeros((1, self.d_emb))

    def link_entities(self, query: str) -> Set[str]:
        detected = set()
        normalized_query = re.sub(r"[^\w\s]", "", query).lower()
        for entity_id in self.kg.get_all_entity_ids():
            readable = entity_id.replace("_", " ").lower()
            if re.search(r"\b" + re.escape(readable) + r"\b", normalized_query):
                detected.add(entity_id)
        return detected

    def resolve_logical_path(self, query: str, active_entities: Set[str]) -> Optional[Tuple[str, str, str]]:
        for pattern, predicate in self.templates:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_name = match.group(1).strip()
                extracted_name = re.sub(r"[^\w\s]", "", extracted_name).lower().replace(" ", "_")

                matched_entity = next((e for e in active_entities if e.lower() == extracted_name), None)
                if matched_entity:
                    return (matched_entity, predicate, "PLACEHOLDER_1")
        return None

    def query_ollama(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Calls local Ollama API deterministic endpoint."""
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception:
            return None

    def extract_factual_declaration(self, user_message: str) -> Optional[Dict[str, str]]:
        """
        Uses local LLM to extract factual declarations in strict JSON format.
        Fails back gracefully if no declaration is found or parsing fails.
        """
        system_prompt = (
            "You are a precise factual information extractor.\n"
            "Analyze the user's message. If the user is declaring or stating a new fact (e.g. 'I was born in London' or 'Einstein cracked Enigma'), "
            "extract the fact as a clean JSON block.\n\n"
            "Rules:\n"
            "1. Output ONLY valid JSON, do not include any other conversational filler.\n"
            "2. Fields required:\n"
            "   - 'is_factual_declaration': boolean (true if user makes a direct factual claim, false if asking a question or greeting)\n"
            "   - 'subject': string (normalized snake_case entity name, e.g. 'Albert_Einstein')\n"
            "   - 'predicate': string (relationship predicate, e.g. 'born_in', 'discovered', 'cracked')\n"
            "   - 'object': string (normalized snake_case target entity, e.g. 'Germany')\n"
        )

        response = self.query_ollama(user_message, system_prompt)
        if not response:
            return None

        # Parse JSON safely out of markdown formatting
        try:
            cleaned = re.sub(r"```json|```", "", response).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start:end+1]
                data = json.loads(cleaned)
                if data.get("is_factual_declaration") is True:
                    return {
                        "subject": data.get("subject", "").strip().replace(" ", "_"),
                        "predicate": data.get("predicate", "").strip(),
                        "object": data.get("object", "").strip().replace(" ", "_")
                    }
        except Exception:
            pass
        return None

    def execute_chat_turn(self, query: str) -> Tuple[str, List[Tuple[str, float]], str]:
        """
        Executes an orchestrator turn.
        Deterministically gates LLM generation, only invoking the Renderer when database facts exist [1].
        """
        # 1. Update CPU Reservoir Compression
        embeddings = self._get_sentence_representation(query)
        for vec in embeddings:
            res_state = self.reservoir.step(vec)

        # 2. Entity Linker Lookup
        active_entities = self.link_entities(query)

        # 3. Decoupled Path Resolution
        path = self.resolve_logical_path(query, active_entities)

        resolved_value = None
        if path:
            source_id, predicate, _ = path
            resolved_value = self.kg.query_relation(source_id, predicate)

        # 4. GATED RENDERER LOOP: Only call LLM if verified database facts exist [1]
        if resolved_value:
            # Retrieve Hebbian Priming Context
            primed_info = []
            for ent in active_entities:
                primed_info.extend(self.plasticity.get_associated_priming_context(ent))

            # Update Hebbian co-activation weights for all active elements + target
            all_active = active_entities.copy()
            all_active.add(resolved_value)
            self.plasticity.update_associations(all_active)

            # Format Prompt for the Renderer
            facts_str = f"[{source_id.replace('_', ' ')} {predicate} {resolved_value.replace('_', ' ')}]"
            priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"

            system_prompt = (
                "You are a language renderer. You know nothing about the conversation except what is provided here.\n"
                f"Verified Facts: {facts_str}\n"
                f"Context Priming: {priming_str}\n"
                "Task: Render a single, grammatically correct response answering the user's question using ONLY the provided facts."
            )

            llm_response = self.query_ollama(query, system_prompt)
            if llm_response:
                return f"Exocortex (Ollama-Renderer) > {llm_response}", primed_info, "RENDER_SUCCESS"
            else:
                # Local renderer fallback
                return f"Exocortex (Simulated) > {source_id.replace('_', ' ')} was resolved to {resolved_value.replace('_', ' ')}.", primed_info, "RENDER_FALLBACK"

        # 5. DETERMINISTIC EXTRACTION GATE: If no facts are resolved, do not call the Renderer [1].
        # Attempt to run the Factual Extractor instead [23].
        extracted_fact = self.extract_factual_declaration(query)
        if extracted_fact:
            sub = extracted_fact["subject"]
            pred = extracted_fact["predicate"]
            obj = extracted_fact["object"]

            # Auto-register newly discovered entities/relation
            self.kg.update_relation(sub, pred, obj)

            # Update Hebbian associations for the new entities so they wire together
            self.plasticity.update_associations({sub, obj})

            return f"Exocortex (Autonomous Learner) > I have recorded a new factual declaration: [{sub.replace('_', ' ')}] -[{pred}]-> [{obj.replace('_', ' ')}].", [], "EXTRACT_SUCCESS"

        # 6. Safe Deterministic Fallback: We have no facts, and the query is not a declaration.
        return "Exocortex > I do not have verified information about that.", [], "DETERMINISTIC_GATED_FALLBACK"


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================

if __name__ == "__main__":
    db_file = "exocortex_kg.db"

    # Persistent Database setup
    is_new = not os.path.exists(db_file)
    exocortex = IntegratedExocortex(db_file)
    if is_new:
         logger.info("Initializing persistent Exocortex Database.")

    print("\n========================================================")
    print("      EXOCORTEX NEURO-SYMBOLIC CHATBOT INITIALIZED      ")
    print("========================================================")
    print("Ask me factual questions! (e.g. 'Where was Marie Curie born?')")
    print("To teach me new things, just speak factually to me!")
    print("  example: 'Einstein was born in Germany' or 'Newton discovered Gravity'")
    print("Type 'exit' or 'quit' to shut down.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Safely shutting down local exocortex.")
                break

            # Execute chat turn (Safe Gated Architecture)
            reply, primed, mode = exocortex.execute_chat_turn(user_input)

            # Print response
            print(reply)

            # Display current memory activations (Hebbian associations)
            if primed:
                print("  [Memory Priming Node Activations]:")
                for node, weight in primed[:3]:
                    print(f"    * Associated Concept: '{node:<13}'  Synaptic Connection Strength: {weight:.4f}")
            print()

        except KeyboardInterrupt:
            print("\nSafely shutting down local exocortex.")
            break