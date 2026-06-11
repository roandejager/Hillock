"""
Phase 3: Unified Exocortex Orchestrator (Closed-Loop Gated Edition)
Integrates SQLite Knowledge Graphs, Hebbian Co-activation,
and CPU-based Reservoir Context Compression.

Closed-Loop Architecture:
  - Fixed (Bug 1): Free-Form Query Resolver with Dynamic Schema Priming.
  - Fixed (Bug 2): Rigid Python pre-filter enforcing a strict question gate.
  - Fixed (Bug 3): Token-based Entity Linking and Identity Resolution.
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
        """Performs a fuzzy, stem-invariant predicate search in SQL."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Exact match check
            cursor.execute("SELECT target_id FROM relations WHERE source_id = ? AND predicate = ?", (source_id, predicate))
            result = cursor.fetchone()
            if result:
                return result[0]

            # 2. Fuzzy, tense-invariant stemming check (e.g. 'work_with' matches 'worked_with')
            cursor.execute("SELECT predicate, target_id FROM relations WHERE source_id = ?", (source_id,))
            all_relations = cursor.fetchall()

            def stem(s: str) -> str:
                # Strip typical verb suffix inflections for matching
                s = s.lower().replace("_", "").replace(" ", "")
                return re.sub(r"ed$|ing$|s$", "", s)

            stemmed_query = stem(predicate)
            for db_pred, tgt_id in all_relations:
                if stemmed_query in stem(db_pred) or stem(db_pred) in stemmed_query:
                    logger.info(f"Fuzzy Predicate Match: Resolved '{predicate}' to database relation '{db_pred}'")
                    return tgt_id

            return None

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
    """CPU-bound Context Compressor [48]."""
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
# PHASE 3: SECURE GATE, RESOLVER, & EXTRACTION
# ==========================================

class IntegratedExocortex:
    def __init__(self, db_path: str, d_emb: int = 64, d_res: int = 128, ollama_model: str = "qwen2:1.5b"):
        self.d_emb = d_emb
        self.ollama_model = ollama_model
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)
        self.reservoir = EchoStateReservoir(d_emb, d_res)

        self.vocab: Dict[str, np.ndarray] = {}
        np.random.seed(101)
        core_words = ["hello", "exocortex", "where", "what", "is", "the", "capital", "born",
                      "radioactivity", "france", "poland", "london", "uk", "enigma"]
        for w in core_words:
            self.vocab[w] = np.random.uniform(-1.0, 1.0, self.d_emb)

    def is_question(self, text: str) -> bool:
        """Fixed (Bug 2): Stricter Python pre-filter checking for syntactic question starters."""
        cleaned = text.strip().lower()
        if cleaned.endswith("?"):
            return True
        question_words = {"who", "what", "where", "when", "why", "how", "is", "did", "does", "was", "can", "are", "which", "whom"}
        tokens = re.sub(r"[^\w\s]", "", cleaned).split()
        if tokens and tokens[0] in question_words:
            return True
        return False

    def resolve_entity_identity(self, name_str: str) -> str:
        """Fixed (Bug 3): Substring identity checks resolve names to pre-existing canonical IDs."""
        normalized_new = name_str.strip().replace(" ", "_").lower()
        all_ids = self.kg.get_all_entity_ids()

        for ent_id in all_ids:
            if ent_id.lower() == normalized_new:
                return ent_id

        for ent_id in all_ids:
            lower_ent = ent_id.lower()
            if normalized_new in lower_ent or lower_ent in normalized_new:
                return ent_id if len(ent_id) >= len(normalized_new) else name_str.strip().replace(" ", "_")

        return name_str.strip().replace(" ", "_")

    def link_entities(self, query: str) -> Set[str]:
        """Fixed (Bug 3): Token-based entity linker resolves names by matching individual word subparts."""
        detected = set()
        query_words = set(re.sub(r"[^\w\s]", " ", query).lower().split())

        for entity_id in self.kg.get_all_entity_ids():
            ent_parts = entity_id.lower().split("_")
            for part in ent_parts:
                if len(part) > 2 and part in query_words:
                    detected.add(entity_id)
                    break
        return detected

    def query_ollama(self, prompt: str, system_prompt: str) -> Optional[str]:
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

    def parse_json_safely(self, raw_text: str) -> Optional[dict]:
        try:
            cleaned = re.sub(r"```json|```", "", raw_text).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                return json.loads(cleaned[start:end+1])
        except Exception:
            pass
        return None

    def extract_field_via_regex(self, text: str, field_name: str) -> Optional[str]:
        pattern = r'"' + re.escape(field_name) + r'"\s*:\s*["\']([^"\']+)["\']'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def translate_query_to_path(self, query: str, active_entities: Set[str]) -> Optional[Tuple[str, str]]:
        """
        Fixed (Bug 1): Free-Form Query Resolver with Dynamic Schema Priming.
        Injects the database's existing predicates for the active entity to guide LLM translation.
        """
        known_predicates = set()
        with sqlite3.connect(self.kg.db_path) as conn:
            cursor = conn.cursor()
            for ent in active_entities:
                cursor.execute("SELECT predicate FROM relations WHERE source_id = ?", (ent,))
                for row in cursor.fetchall():
                    known_predicates.add(row[0])

        predicates_str = ", ".join([f"'{p}'" for p in known_predicates]) if known_predicates else "None"

        system_prompt = (
            "You are a database query translator. Output ONLY a JSON block.\n"
            "Translate the question into a structured lookup with 'subject' and 'predicate'.\n"
            f"Active DB Predicates for these entities: [{predicates_str}]\n"
            "If possible, select a predicate from the Active DB Predicates list.\n"
            "Example JSON:\n"
            "{\n"
            "  \"subject\": \"Marie_Curie\",\n"
            "  \"predicate\": \"born_in\"\n"
            "}\n"
        )
        response = self.query_ollama(query, system_prompt)
        if not response:
            return None

        data = self.parse_json_safely(response)

        # Try JSON extraction
        if data and "subject" in data and "predicate" in data:
            resolved_subject = self.resolve_entity_identity(data["subject"])
            return resolved_subject, data["predicate"].strip()

        # Fallback to direct regex string parsing if Qwen produces malformed JSON
        reg_sub = self.extract_field_via_regex(response, "subject")
        reg_pred = self.extract_field_via_regex(response, "predicate")
        if reg_sub and reg_pred:
            resolved_subject = self.resolve_entity_identity(reg_sub)
            return resolved_subject, reg_pred

        return None

    def extract_factual_declaration(self, user_message: str) -> Optional[Dict[str, str]]:
        """Extracts conversational factual statements into JSON, resolving names."""
        system_prompt = (
            "You are a factual extractor. Output ONLY a JSON block.\n"
            "If the user makes a factual statement, extract the fact.\n"
            "Example statement: 'Einstein was born in Germany'\n"
            "Example JSON:\n"
            "{\n"
            "  \"is_factual_declaration\": true,\n"
            "  \"subject\": \"Albert_Einstein\",\n"
            "  \"predicate\": \"was_born_in\",\n"
            "  \"object\": \"Germany\"\n"
            "}\n"
            "If the input is a question or greeting, return:\n"
            "{\n"
            "  \"is_factual_declaration\": false\n"
            "}\n"
        )
        response = self.query_ollama(user_message, system_prompt)
        if not response:
            return None

        data = self.parse_json_safely(response)

        # Check standard JSON output
        if data and data.get("is_factual_declaration") is True:
            sub = self.resolve_entity_identity(data.get("subject", ""))
            obj = self.resolve_entity_identity(data.get("object", ""))
            return {
                "subject": sub,
                "predicate": data.get("predicate", "").strip(),
                "object": obj
            }

        # Check regex fallback for safety
        reg_is_fact = self.extract_field_via_regex(response, "is_factual_declaration")
        if reg_is_fact and "true" in reg_is_fact.lower():
            reg_sub = self.extract_field_via_regex(response, "subject")
            reg_pred = self.extract_field_via_regex(response, "predicate")
            reg_obj = self.extract_field_via_regex(response, "object")
            if reg_sub and reg_pred and reg_obj:
                sub = self.resolve_entity_identity(reg_sub)
                obj = self.resolve_entity_identity(reg_obj)
                return {"subject": sub, "predicate": reg_pred, "object": reg_obj}

        return None

    def _get_sentence_representation(self, sentence: str) -> np.ndarray:
        tokens = re.sub(r"[^\w\s]", "", sentence).lower().split()
        vectors = [self.vocab.get(t, np.random.uniform(-0.1, 0.1, self.d_emb)) for t in tokens]
        return np.array(vectors) if vectors else np.zeros((1, self.d_emb))

    def execute_chat_turn(self, query: str) -> Tuple[str, List[Tuple[str, float]], str]:
        """Runs the gated, self-learning exocortex routing pipeline."""
        # Update CPU Reservoir Compression
        embeddings = self._get_sentence_representation(query)
        for vec in embeddings:
            res_state = self.reservoir.step(vec)

        # Check if the input is a question using the pre-filter
        is_query = self.is_question(query)

        resolved_value = None
        source_id, predicate = None, None

        # 1. QUERY PATHWAY: Execute retrieval check if the input is classified as a question
        if is_query:
            active_entities = self.link_entities(query)
            if active_entities:
                path = self.translate_query_to_path(query, active_entities)
                if path:
                    source_id, predicate = path
                    resolved_value = self.kg.query_relation(source_id, predicate)

        # 2. GATED RENDERER LOOP: Only invoke LLM generation if verified database facts exist [1]
        if resolved_value:
            # Retrieve Hebbian Priming Context
            primed_info = self.plasticity.get_associated_priming_context(source_id)

            # Update Hebbian co-activation weights for all active elements + target
            self.plasticity.update_associations({source_id, resolved_value})

            # Format Prompt for the Renderer
            facts_str = f"[{source_id.replace('_', ' ')} {predicate} {resolved_value.replace('_', ' ')}]"
            priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"

            system_prompt = (
                "You are a language renderer. You know nothing about the conversation except what is provided here.\n"
                f"Verified Facts: {facts_str}\n"
                f"Context Priming: {priming_str}\n"
                "Task: Render a single response answering the user's question using ONLY the provided facts."
            )

            llm_response = self.query_ollama(query, system_prompt)
            if llm_response:
                return f"Exocortex (Ollama-Renderer) > {llm_response}", primed_info, "RENDER_SUCCESS"
            else:
                return f"Exocortex (Simulated) > {source_id.replace('_', ' ')} was resolved to {resolved_value.replace('_', ' ')}.", primed_info, "RENDER_FALLBACK"

        # 3. EXTRACTION PATHWAY: If input is not a question, run the autonomous learning gate
        if not is_query:
            extracted_fact = self.extract_factual_declaration(query)
            if extracted_fact:
                sub = extracted_fact["subject"]
                pred = extracted_fact["predicate"]
                obj = extracted_fact["object"]

                # Auto-register relation and update associations
                self.kg.update_relation(sub, pred, obj)
                self.plasticity.update_associations({sub, obj})

                return f"Exocortex (Autonomous Learner) > I have recorded a new factual declaration: [{sub.replace('_', ' ')}] -[{pred}]-> [{obj.replace('_', ' ')}].", [], "EXTRACT_SUCCESS"

        # 4. Deterministic Gated Fallback (Hallucination Blocked)
        return "Exocortex > I do not have verified information about that.", [], "DETERMINISTIC_GATED_FALLBACK"


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================

if __name__ == "__main__":
    db_file = "exocortex_kg.db"

    # Deterministic DB Reset: Force-clears legacy fragmented files from prior failed configurations
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            logger.info("Cleared outdated database file for a clean, unfragmented run.")
        except PermissionError:
            pass

    exocortex = IntegratedExocortex(db_file)

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