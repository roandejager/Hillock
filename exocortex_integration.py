"""
Phase 3: Unified Exocortex Orchestrator (HDC Reservoir & Persistent Brain Edition)
Integrates SQLite Knowledge Graphs, Hebbian Co-activation,
and CPU-bound Hyperdimensional Computing (HDC) Reservoirs.

Upgrades:
  - Fixed (Bug 1): Database persistence enabled across sessions.
  - Fixed (Bug 2): Rigid question-filtering restricted to interrogatives.
  - Upgrade (HDC Reservoir): Replaced ESN with 10,000-D Bipolar HDC vector loop.
  - Upgrade (Wired Readout): Computes cosine similarity to inject top-3 active concepts.
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
# PHASE 2: HYPERDIMENSIONAL COMPUTING (HDC)
# ==========================================

class HyperdimensionalReservoir:
    """
    CPU-bound Context Compressor using Vector Symbolic Architectures (VSA) [1, 28].
    Replaces ESN continuous states with 10,000-dimensional bipolar hypervectors.
    """
    def __init__(self, d: int = 10000):
        self.d = d
        self.state = np.zeros(self.d, dtype=np.int32)

        # Hypervector codebook maps known entity IDs to random bipolar vectors
        self.codebook: Dict[str, np.ndarray] = {}

        # Self-contained vocab mappings for non-entity sentence tokens
        self.vocab_book: Dict[str, np.ndarray] = {}

    def get_or_allocate_hypervector(self, name_id: str, is_vocab_token: bool = False) -> np.ndarray:
        """Dynamically retrieves or allocates a unique bipolar hypervector (+1/-1) [28]."""
        book = self.vocab_book if is_vocab_token else self.codebook

        if name_id not in book:
            # Generate random bipolar hypervector
            vector = np.random.choice([-1, 1], size=self.d).astype(np.int32)
            book[name_id] = vector
        return book[name_id]

    def step(self, token_hv: np.ndarray) -> np.ndarray:
        """
        Updates the context hypervector by:
        1. Permutation: rolling/shifting the existing state to track temporal order.
        2. Binding: multiplying element-wise by the new token.
        3. Bundling: adding the bound vector to the running history sum.
        """
        # 1. Permutation (Cyclic shift)
        permuted_state = np.roll(self.state, shift=1)

        # 2. Binding (Element-wise multiplication)
        bound = permuted_state * token_hv

        # 3. Bundling (Addition / Accumulation)
        self.state = self.state + bound
        return self.state

    def get_context_fingerprint(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Wired Readout: Computes cosine similarity between the current context
        hypervector and all hypervectors in the entity codebook.
        """
        scores = []
        ctx_norm = np.linalg.norm(self.state)
        if ctx_norm == 0:
            return []

        for entity_id, hv in self.codebook.items():
            hv_norm = np.linalg.norm(hv)
            if hv_norm == 0:
                continue
            similarity = np.dot(self.state, hv) / (ctx_norm * hv_norm)
            scores.append((entity_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==========================================
# PHASE 3: SECURE GATE, RESOLVER, & EXTRACTION
# ==========================================

class IntegratedExocortex:
    def __init__(self, db_path: str, d_res: int = 10000, ollama_model: str = "qwen2:1.5b"):
        self.ollama_model = ollama_model
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)
        self.hdc = HyperdimensionalReservoir(d=d_res)

        # Pre-seed hypervectors for all initial database entities
        for ent_id in self.kg.get_all_entity_ids():
            self.hdc.get_or_allocate_hypervector(ent_id)

    def is_question(self, text: str) -> bool:
        """
        Fixed (Bug 2): Rigid question-filtering restricted to interrogatives.
        Prevents declarative sentences starting with 'is' from skipping extraction.
        """
        cleaned = text.strip().lower()
        if cleaned.endswith("?"):
            return True
        # Rigid set containing purely interrogative pronouns/adverbs
        question_words = {"who", "what", "where", "when", "why", "how", "which", "whom"}
        tokens = re.sub(r"[^\w\s]", "", cleaned).split()
        if tokens and tokens[0] in question_words:
            return True
        return False

    def resolve_entity_identity(self, name_str: str) -> str:
        """Resolves identity variations (e.g. 'Turing' matching 'Alan_Turing')."""
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
        """Token-based entity linker resolves names by matching individual word subparts."""
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
        """Translates natural questions into a structured database lookup using Dynamic Schema Priming."""
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
        if data and "subject" in data and "predicate" in data:
            resolved_subject = self.resolve_entity_identity(data["subject"])
            return resolved_subject, data["predicate"].strip()

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
        if data and data.get("is_factual_declaration") is True:
            sub = self.resolve_entity_identity(data.get("subject", ""))
            obj = self.resolve_entity_identity(data.get("object", ""))
            return {
                "subject": sub,
                "predicate": data.get("predicate", "").strip(),
                "object": obj
            }

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

    def process_hdc_context(self, text: str, active_entities: Set[str]) -> List[Tuple[str, float]]:
        """
        Passes sentence tokens through the HDC reservoir.
        Integrates newly resolved entities into the hyperdimensional codebook.
        """
        # Ensure any newly discovered entities are allocated a stable hypervector in the codebook
        for ent in active_entities:
            self.hdc.get_or_allocate_hypervector(ent, is_vocab_token=False)

        # Segment input text into simple tokens
        tokens = re.sub(r"[^\w\s]", "", text).lower().split()

        # Step each token sequentially through the HDC permutation-binding loop
        for token in tokens:
            # Map word to either the entity codebook or the vocab codebook
            resolved_id = self.resolve_entity_identity(token)
            if resolved_id in self.hdc.codebook:
                token_hv = self.hdc.get_or_allocate_hypervector(resolved_id, is_vocab_token=False)
            else:
                token_hv = self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)

            self.hdc.step(token_hv)

        # Return the top-3 closest semantic active concepts matching our conversational fingerprint
        return self.hdc.get_context_fingerprint(top_k=3)

    def execute_chat_turn(self, query: str) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Runs the gated, self-learning exocortex routing pipeline."""
        # Check if the input is a question using the pre-filter
        is_query = self.is_question(query)

        # Extract active entities present in the query
        active_entities = self.link_entities(query)

        # 1. Update CPU HDC Reservoir Sequence Tracking (Context Scaling)
        # Returns the top-3 closest matching semantic entities (Wired Readout)
        hdc_fingerprint = self.process_hdc_context(query, active_entities)

        resolved_value = None
        source_id, predicate = None, None

        # 2. QUERY PATHWAY: Execute retrieval check if the input is classified as a question
        if is_query:
            if active_entities:
                path = self.translate_query_to_path(query, active_entities)
                if path:
                    source_id, predicate = path
                    resolved_value = self.kg.query_relation(source_id, predicate)

        # 3. GATED RENDERER LOOP: Only invoke LLM generation if verified database facts exist [1]
        if resolved_value:
            # Retrieve Hebbian Priming Context
            primed_info = self.plasticity.get_associated_priming_context(source_id)

            # Update Hebbian co-activation weights for all active elements + target
            self.plasticity.update_associations({source_id, resolved_value})

            # Format Prompt for the Renderer
            facts_str = f"[{source_id.replace('_', ' ')} {predicate} {resolved_value.replace('_', ' ')}]"
            priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"

            # Inject top HDC active traces directly into system prompt
            fingerprint_str = ", ".join([f"{node} (match {sim:.2f})" for node, sim in hdc_fingerprint]) if hdc_fingerprint else "None"

            system_prompt = (
                "You are a language renderer. You know nothing about the conversation except what is provided here.\n"
                f"Verified Facts: {facts_str}\n"
                f"Context Priming: {priming_str}\n"
                f"HDC Conversational Fingerprint: {fingerprint_str}\n"
                "Task: Render a single response answering the user's question using ONLY the provided facts."
            )

            llm_response = self.query_ollama(query, system_prompt)
            if llm_response:
                return f"Exocortex (Ollama-Renderer) > {llm_response}", primed_info, hdc_fingerprint, "RENDER_SUCCESS"
            else:
                return f"Exocortex (Simulated) > {source_id.replace('_', ' ')} was resolved to {resolved_value.replace('_', ' ')}.", primed_info, hdc_fingerprint, "RENDER_FALLBACK"

        # 4. EXTRACTION PATHWAY: If input is not a question, run the autonomous learning gate
        if not is_query:
            extracted_fact = self.extract_factual_declaration(query)
            if extracted_fact:
                sub = extracted_fact["subject"]
                pred = extracted_fact["predicate"]
                obj = extracted_fact["object"]

                # Auto-register relation and update associations
                self.kg.update_relation(sub, pred, obj)
                self.plasticity.update_associations({sub, obj})

                # Update ESN/HDC codebook to allocate hypervector for the newly taught entities
                self.hdc.get_or_allocate_hypervector(sub)
                self.hdc.get_or_allocate_hypervector(obj)

                return f"Exocortex (Autonomous Learner) > I have recorded a new factual declaration: [{sub.replace('_', ' ')}] -[{pred}]-> [{obj.replace('_', ' ')}].", [], hdc_fingerprint, "EXTRACT_SUCCESS"

        # 5. Deterministic Gated Fallback (Hallucination Blocked)
        return "Exocortex > I do not have verified information about that.", [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================

if __name__ == "__main__":
    db_file = "exocortex_kg.db"

    # Fixed (Bug 1): Database persistence enabled across sessions.
    # We no longer delete the exocortex_kg.db file on restart.
    is_new = not os.path.exists(db_file)
    exocortex = IntegratedExocortex(db_file)
    if is_new:
         logger.info("Initializing persistent Exocortex Database.")
    else:
         logger.info("Persistent Exocortex Database loaded successfully.")

    print("\n========================================================")
    print("      EXOCORTEX NEURO-SYMBOLIC CHATBOT INITIALIZED      ")
    print("========================================================")
    print("Ask me factual questions! (e.g. 'Where was Marie Curie born?')")
    print("To teach me new things, just speak factually to me!")
    print("  example: 'Einstein was born in Germany' or 'Paris is the capital of France'")
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
            reply, primed, fingerprint, mode = exocortex.execute_chat_turn(user_input)

            # Print response
            print(reply)

            # Display current memory activations (Hebbian associations)
            if primed:
                print("  [Memory Priming Node Activations]:")
                for node, weight in primed[:3]:
                    print(f"    * Associated Concept: '{node:<13}'  Synaptic Connection Strength: {weight:.4f}")

            # Display HDC active traces (Conversational Context Fingerprint)
            if fingerprint and mode == "RENDER_SUCCESS":
                print("  [HDC Conversational Fingerprint Traces]:")
                for node, sim in fingerprint[:3]:
                    print(f"    * Active Semantic Echo: '{node:<13}'  Vector Cosine Similarity: {sim:.4f}")
            print()

        except KeyboardInterrupt:
            print("\nSafely shutting down local exocortex.")
            break