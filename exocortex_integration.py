"""
Phase 3: Unified Exocortex Orchestrator (Optimized Ingestion Edition)
Integrates SQLite Knowledge Graphs, Hebbian Co-activation,
and CPU-bound Leaky Hyperdimensional Computing (HDC) Reservoirs.

Upgrades:
  - Fixed (Ingestion Parser): Properly splits filename and mode in /ingest.
  - Fixed (Verb-Filter): Restored and corrected the SpaCy-based verb presence filter [1].
  - Fixed (Fast Mode): Added 'fast_mode' toggle to bypass LLM calls during ingestion [1].
"""

import sqlite3
import re
import numpy as np
import logging
import os
import json
import urllib.request
import urllib.error
import sys
import subprocess
from typing import List, Tuple, Dict, Set, Optional

# Guarantee 'nlp' is always defined globally to prevent runtime NameErrors [2]
nlp = None

# Try loading SpaCy for ultra-fast grammatical dependency parsing [1]
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

# Optional import for PDF processing [25]
try:
    import pypdf
except ImportError:
    pypdf = None

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

    def clear_and_reinitialize(self) -> None:
        """Fixed (Problem 3): Safe SQL-level reset."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = OFF;")
            cursor.execute("DROP TABLE IF EXISTS hebbian_weights;")
            cursor.execute("DROP TABLE IF EXISTS relations;")
            cursor.execute("DROP TABLE IF EXISTS entities;")
            conn.commit()

        self._initialize_db()
        self.seed_initial_knowledge()

    def get_entity_count(self) -> int:
        """Fixed (Problem 3): Queries exact unique entity count in SQL [1]."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM entities")
            return cursor.fetchone()[0]

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

            # 2. Fuzzy, tense-invariant stemming check
            cursor.execute("SELECT predicate, target_id FROM relations WHERE source_id = ?", (source_id,))
            all_relations = cursor.fetchall()

            def stem(s: str) -> str:
                s = s.lower().replace("_", "").replace(" ", "")
                return re.sub(r"ed$|ing$|s$", "", s)

            stemmed_query = stem(predicate)
            for db_pred, tgt_id in all_relations:
                if stemmed_query in stem(db_pred) or stem(db_pred) in stemmed_query:
                    logger.info(f"Fuzzy Predicate Match: Resolved '{predicate}' to database relation '{db_pred}'")
                    return tgt_id

            return None

    def get_all_facts_for_entities(self, active_entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Retrieves all registered facts involving active entities as subject or object."""
        if not active_entities:
            return []
        placeholders = ", ".join(["?"] * len(active_entities))
        query = f"SELECT source_id, predicate, target_id FROM relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
        params = list(active_entities) + list(active_entities)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

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
    """CPU-bound Context Compressor using Vector Symbolic Architectures (VSA) [1, 28]."""
    def __init__(self, d: int = 10000):
        self.d = d
        self.state = np.zeros(self.d, dtype=np.float64)
        self.codebook: Dict[str, np.ndarray] = {}
        self.vocab_book: Dict[str, np.ndarray] = {}

    def get_or_allocate_hypervector(self, name_id: str, is_vocab_token: bool = False) -> np.ndarray:
        """Retrieves or allocates a unique, static random bipolar vector [28]."""
        book = self.vocab_book if is_vocab_token else self.codebook
        if name_id not in book:
            vector = np.random.choice([-1, 1], size=self.d).astype(np.int32)
            book[name_id] = vector
        return book[name_id]

    def step(self, token_hv: np.ndarray, decay: float = 0.95) -> np.ndarray:
        """
        Leaky reservoir update.
        Decays the original accumulated state before adding the newly bound sequence trace [28, 48].
        """
        bound_token = np.roll(self.state, shift=1) * token_hv
        self.state = (decay * self.state) + bound_token
        return self.state

    def get_context_fingerprint(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Wired Readout: Computes cosine similarity between running context state
        and static entity codebook [28, 48].
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
# PHASE 3: SECURE GATE, SELECTOR, & EXTRACTION
# ==========================================

class IntegratedExocortex:
    def __init__(self, db_path: str, d_res: int = 10000, ollama_model: str = "qwen2:1.5b"):
        self.d_emb = 64
        self.ollama_model = ollama_model
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)
        self.hdc = HyperdimensionalReservoir(d=d_res)

        # Configurable verbosity mode [1]
        self.verbosity_mode = "BALANCED"  # Options: STRICT, BALANCED, CONVERSATIONAL

        # Predicate Normalization Map (Fixes SQL Clutter & Mapped SpaCy Lemmas) [1]
        self.predicate_map = {
            "was_born_in": "born_in",
            "was born in": "born_in",
            "was_born": "born_in",
            "was born": "born_in",
            "bear": "born_in",                  # Maps SpaCy 'born' root lemma to born_in
            "born": "born_in",
            "came_from": "born_in",
            "work": "collaborated_with",        # Maps SpaCy 'worked' root lemma
            "work_with": "collaborated_with",
            "worked_with": "collaborated_with",
            "worked with": "collaborated_with",
            "partnered_with": "collaborated_with",
            "partnered with": "collaborated_with",
            "co_invented": "discovered",
            "discovered": "discovered",
            "found": "discovered",
            "uncovered": "discovered",
            "crack": "cracked",                 # Maps SpaCy 'cracked' root lemma
            "cracked": "cracked",
            "broke": "cracked"
        }

        # Pre-seed hypervectors for all initial database entities
        for ent_id in self.kg.get_all_entity_ids():
            self.hdc.get_or_allocate_hypervector(ent_id)

    def is_question(self, text: str) -> bool:
        """Fixed (Bug 2): Rigid question-filtering restricted strictly to interrogative words."""
        cleaned = text.strip().lower()
        if cleaned.endswith("?"):
            return True
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
            with urllib.request.urlopen(req, timeout=30) as response:
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
        """Fixed (Bug 1): Fully restored regex-based JSON extraction fallback."""
        pattern = r'"' + re.escape(field_name) + r'"\s*:\s*["\']?([^"\'}\n]+)["\']?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def select_answering_facts(self, query: str, facts: List[Tuple[str, str, str]], threshold: float = 0.40) -> List[Tuple[str, str, str, float]]:
        """
        Fixed (HDC Multi-Fact Gating): Computes cosine similarities and returns
        ALL candidate facts that score above the threshold [1, 28].
        """
        if not facts:
            return []

        # Clean query tokens
        query_tokens = set(re.sub(r"[^\w\s]", "", query).lower().split())
        query_hv = np.zeros(self.hdc.d, dtype=np.int32)

        for token in query_tokens:
            resolved = self.resolve_entity_identity(token)
            if resolved in self.hdc.codebook:
                query_hv += self.hdc.get_or_allocate_hypervector(resolved, is_vocab_token=False)
            else:
                query_hv += self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)

        scored_facts = []

        # Match each candidate fact against the query
        for s, p, o in facts:
            # Expand fact into elements, including known relation tenses/synonyms
            fact_words = [s.lower(), o.lower()]
            fact_words.extend(p.lower().replace("_", " ").split())

            # Map predicate tenses into the semantic vector footprint
            if p == "collaborated_with" or p == "work":
                fact_words.extend(["work", "worked", "with", "partner", "partnered", "collaborated"])
            elif p == "born_in" or p == "bear":
                fact_words.extend(["born", "in", "birth", "came", "from", "bear", "was"])
            elif p == "discovered" or p == "discover":
                fact_words.extend(["discover", "discovered", "found", "find", "science"])
            elif p == "cracked" or p == "crack":
                fact_words.extend(["crack", "cracked", "broke", "break", "decrypted"])

            # Bundle candidate fact word hypervectors together
            fact_hv = np.zeros(self.hdc.d, dtype=np.int32)
            for word in set(fact_words):
                resolved_word = self.resolve_entity_identity(word)
                if resolved_word in self.hdc.codebook:
                    fact_hv += self.hdc.get_or_allocate_hypervector(resolved_word, is_vocab_token=False)
                else:
                    fact_hv += self.hdc.get_or_allocate_hypervector(word, is_vocab_token=True)

            # Compute Cosine Similarity between Query Vector and Fact Vector
            q_norm = np.linalg.norm(query_hv)
            f_norm = np.linalg.norm(fact_hv)
            similarity = np.dot(query_hv, fact_hv) / (q_norm * f_norm) if (q_norm > 0 and f_norm > 0) else 0.0

            logger.info(f"HDC Semantic Matcher: Candidate Fact [{s} {p} {o}] Cosine Similarity: {similarity:.4f}")

            if similarity >= threshold:
                scored_facts.append((s, p, o, similarity))

        # Sort by score descending
        scored_facts.sort(key=lambda x: x[3], reverse=True)
        return scored_facts

    def extract_factual_declaration_two_pass(self, sentence: str) -> Optional[Dict[str, str]]:
        """
        Solved (Problem 1): Two-Pass Relation Ingestor [1, 23].
        1. Entity Anchoring Pass: Identifies related subjects.
        2. Constrained Predicate Pass: Extracts verb while avoiding object bleeding.
        """
        # Pass 1: Entity Anchoring (Determine Subject and Object first)
        system_1 = (
            "You are an entity extractor. Output ONLY a JSON block.\n"
            "Identify and extract the two primary entities being related in the sentence.\n"
            "Normalize them using snake_case (e.g., 'Marie_Curie').\n"
            "Example JSON:\n"
            "{\n"
            "  \"entity_a\": \"Marie_Curie\",\n"
            "  \"entity_b\": \"Radioactivity\"\n"
            "}\n"
        )
        response_1 = self.query_ollama(sentence, system_1)
        logger.info(f"DEBUG: Two-Pass - Pass 1 Raw Output:\n{response_1}")
        if not response_1:
            return None

        data_1 = self.parse_json_safely(response_1)
        ent_a, ent_b = None, None
        if data_1 and "entity_a" in data_1 and "entity_b" in data_1:
            ent_a = data_1["entity_a"]
            ent_b = data_1["entity_b"]
        else:
            ent_a = self.extract_field_via_regex(response_1, "entity_a")
            ent_b = self.extract_field_via_regex(response_1, "entity_b")

        if not ent_a or not ent_b:
            return None

        # Resolve entity identities against database schema to prevent fragmentation
        resolved_a = self.resolve_entity_identity(ent_a)
        resolved_b = self.resolve_entity_identity(ent_b)

        # Pass 2: Predicate Extraction under Anchored Constraints
        system_2 = (
            "You are a predicate extractor. Output ONLY a JSON block.\n"
            "Extract ONLY the single verb or verb phrase that connects Entity A to Entity B in the sentence.\n"
            "Rules:\n"
            "1. Return ONLY the relationship verb, no nouns, subjects, or objects.\n"
            f"2. Do NOT include '{resolved_a}' or '{resolved_b}' or any words from them in the predicate.\n"
            "Example JSON:\n"
            "{\n"
            "  \"predicate\": \"discovered\"\n"
            "}\n"
        )
        prompt_2 = (
            f"Sentence: '{sentence}'\n"
            f"Entity A: '{resolved_a}'\n"
            f"Entity B: '{resolved_b}'"
        )
        response_2 = self.query_ollama(prompt_2, system_2)
        logger.info(f"DEBUG: Two-Pass - Pass 2 Raw Output:\n{response_2}")
        if not response_2:
            return None

        data_2 = self.parse_json_safely(response_2)
        pred = None
        if data_2 and "predicate" in data_2:
            pred = data_2["predicate"]
        else:
            pred = self.extract_field_via_regex(response_2, "predicate")

        if not pred:
            return None

        # Validation checks: Strip the target if bleeding still occurred
        clean_pred = pred.strip().lower().replace(" ", "_")
        normalized_b = resolved_b.lower().replace("_", "")

        if normalized_b in clean_pred.replace("_", ""):
            logger.warning(f"Validation: Bleeding detected! Stripped '{resolved_b}' from predicate '{pred}'")
            clean_pred = clean_pred.replace(normalized_b, "").strip("_")

        if not clean_pred or clean_pred in ["", "_"]:
            clean_pred = "related_to"

        return {
            "subject": resolved_a,
            "predicate": clean_pred,
            "object": resolved_b
        }

    def extract_fact_spacy(self, sentence: str) -> Optional[Dict[str, str]]:
        """
        Ingestion Speed Upgrade (Step 1): Classical NLP parsing via SpaCy [1].
        Extracts facts in under 1ms on CPU, bypassing API calls entirely.
        """
        global nlp
        if nlp is None:
            return None
        try:
            doc = nlp(sentence)

            # Find subject, verb, and object using grammatical dependency parsing [1]
            subject, verb_token, obj = None, None, None
            for token in doc:
                if token.dep_ in ("nsubj", "nsubjpass") and subject is None:
                    subject = token
                if token.dep_ == "ROOT":
                    verb_token = token
                if token.dep_ in ("dobj", "attr", "pobj") and obj is None:
                    obj = token

            if not subject or not verb_token or not obj:
                return None

            # Expand subject and object to full noun phrases (excluding determiners like 'the', 'a')
            subject_text = " ".join([t.text for t in subject.subtree if t.dep_ not in ("det",)])
            obj_text = " ".join([t.text for t in obj.subtree if t.dep_ not in ("det",)])
            predicate = verb_token.lemma_  # base form (lemma)

            # Handle auxiliary/copula structures ("Paris is capital of France" -> "capital_of")
            if verb_token.lemma_ == "be":
                for child in verb_token.children:
                    if child.dep_ in ("acomp", "xcomp", "attr"):
                        predicate = child.lemma_
                        break
                # Special check for "was born" patterns
                for token in doc:
                    if token.dep_ == "ROOT" and token.lemma_ in ("born", "bear"):
                        predicate = "born_in"
                        break

            # Normalize names using existing entity codes
            sub_resolved = self.resolve_entity_identity(subject_text.replace(" ", "_"))
            obj_resolved = self.resolve_entity_identity(obj_text.replace(" ", "_"))

            return {
                "subject": sub_resolved,
                "predicate": predicate,
                "object": obj_resolved
            }
        except Exception as e:
            logger.error(f"SpaCy extraction failed: {e}")
            return None

    def process_hdc_context(self, text: str, active_entities: Set[str]) -> List[Tuple[str, float]]:
        """Passes sentence tokens through the HDC reservoir and returns context traces [28]."""
        for ent in active_entities:
            self.hdc.get_or_allocate_hypervector(ent, is_vocab_token=False)

        tokens = re.sub(r"[^\w\s]", "", text).lower().split()
        for token in tokens:
            resolved_id = self.resolve_entity_identity(token)
            if resolved_id in self.hdc.codebook:
                token_hv = self.hdc.get_or_allocate_hypervector(resolved_id, is_vocab_token=False)
            else:
                token_hv = self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)
            self.hdc.step(token_hv)

        return self.hdc.get_context_fingerprint(top_k=3)

    def is_worth_extracting(self, sentence: str) -> bool:
        """
        Speed Booster: Syntactic pre-filter [1].
        Skips sentences during ingestion that do not contain relational claims.
        """
        cleaned = sentence.strip()
        if len(cleaned.split()) < 3:
            return False

        global nlp
        if nlp is not None:
            # Use SpaCy to check if a verb or auxiliary verb exists (takes < 1ms)
            doc = nlp(sentence)
            has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
            if not has_verb:
                return False
        else:
            # If it doesn't mention capitalized nouns or known entities, skip it
            capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", cleaned)
            active_ents = self.link_entities(cleaned)

            if not active_ents and len(capitalized_words) < 2:
                return False

        return True

    def ingest_document(self, file_path: str, fast_mode: bool = True) -> str:
        """
        Indexes bulk factual claims from a local text or PDF document into SQL and HDC [23, 25].
        Fixed: Respects fast_mode parameter to completely bypass LLM fallback on CPU [1].
        """
        if not os.path.exists(file_path):
            return f"Ingestion Error: Local file '{file_path}' does not exist."

        ext = os.path.splitext(file_path)[1].lower()
        raw_text = ""

        # Native PDF Ingestor [25]
        if ext == ".pdf":
            if pypdf is None:
                return "Ingestion Error: 'pypdf' package is required for PDFs. Run 'pip install pypdf'."
            try:
                reader = pypdf.PdfReader(file_path)
                pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                raw_text = "\n".join(pages_text)
                logger.info(f"Successfully extracted {len(pages_text)} pages from PDF: {file_path}")
            except Exception as e:
                return f"Ingestion Error: Failed to parse PDF file. Details: {e}"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

        sentences = [s.strip() for s in re.split(r"[.!?\n]", raw_text) if s.strip()]

        extracted_count = 0
        skipped_count = 0

        for i, sentence in enumerate(sentences):
            # Speed Booster Filter
            if not self.is_worth_extracting(sentence):
                skipped_count += 1
                continue

            logger.info(f"Ingesting [{i+1}/{len(sentences)}]: '{sentence}'")
            active_ents = self.link_entities(sentence)
            self.process_hdc_context(sentence, active_ents)

            # Step 1: Attempt ultra-fast classical NLP extraction via SpaCy first [1]
            fact = self.extract_fact_spacy(sentence)

            # Step 1 Fallback: Call secure Two-Pass LLM extractor only if fast_mode is disabled [1]
            if not fact and not fast_mode:
                logger.info(" -> SpaCy returned None. Falling back to Two-Pass LLM extraction...")
                fact = self.extract_factual_declaration_two_pass(sentence)

            if fact:
                sub = fact["subject"]
                pred = fact["predicate"]
                obj = fact["object"]

                # Apply predicate normalization map
                norm_pred = self.predicate_map.get(pred.strip().lower(), pred.strip().lower().replace(" ", "_"))

                self.kg.update_relation(sub, norm_pred, obj)
                self.plasticity.update_associations({sub, obj})
                self.hdc.get_or_allocate_hypervector(sub)
                self.hdc.get_or_allocate_hypervector(obj)

                extracted_count += 1
                logger.info(f" -> [INDEXED RELATION]: [{sub}] -[{norm_pred}]-> [{obj}]")

        return f"Autonomous Ingestion complete. Successfully indexed {extracted_count} facts from {len(sentences)} sentences (Skipped {skipped_count} non-factual lines)."

    def _get_mode_prompts(self, query: str, facts_str: str, primed_info: list, hdc_fingerprint: list) -> Tuple[str, str]:
        """Fixed: Dynamically updates the rendering prompt based on active verbosity settings [1]."""
        priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"
        fingerprint_str = ", ".join([f"{node} (match {sim:.2f})" for node, sim in hdc_fingerprint]) if hdc_fingerprint else "None"

        if self.verbosity_mode == "STRICT":
            system_prompt = (
                "You are a professional fact-to-text renderer. Translate ONLY the provided fact into one sentence. "
                "Do not add any extra context, historical assumptions, or details."
            )
            render_prompt = f"Fact: {facts_str}"

        elif self.verbosity_mode == "BALANCED":
            system_prompt = (
                "You are a knowledgeable assistant. Answer the question using the verified facts provided. "
                "You may add one short sentence of natural conversational context if it flows naturally, "
                "but do NOT invent specific facts, dates, or claims not in the verified data."
            )
            render_prompt = (
                f"Verified fact: {facts_str}\n"
                f"Related context from memory: {priming_str}\n"
                f"Question: {query}"
            )

        else:  # CONVERSATIONAL
            system_prompt = (
                "You are a curious, warm assistant with access to a verified knowledge base. "
                "Answer naturally and conversationally. The verified fact you must include is provided. "
                "You may expand slightly using the memory context provided, but always be clear "
                "that the verified fact is the grounded answer. Never invent specific data."
            )
            render_prompt = (
                f"Verified fact: {facts_str}\n"
                f"Memory associations: {priming_str}\n"
                f"HDC context traces: {fingerprint_str}\n"
                f"Answer this question naturally: {query}"
            )

        return system_prompt, render_prompt

    def execute_chat_turn(self, query: str) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]], str]:
        """Runs the gated, self-learning exocortex routing pipeline."""
        is_query = self.is_question(query)

        # 1. GREETING/FILLER PRE-FILTER: Blocks conversational filler from entering extraction pathways [1]
        greetings = {"hello", "hi", "hey", "greetings", "thanks", "thank you", "bye", "goodbye"}
        query_clean = re.sub(r"[^\w\s]", "", query).strip().lower()

        if query_clean in greetings or len(query_clean.split()) < 2:
            dummy_primed = []
            dummy_fingerprint = []
            if self.verbosity_mode == "CONVERSATIONAL":
                return "Exocortex > Hello! I am your conversational exocortex. Ask me any factual questions about my indexed knowledge.", dummy_primed, dummy_fingerprint, "GREETING"
            elif self.verbosity_mode == "BALANCED":
                return "Exocortex > Hello. Ready for factual questions.", dummy_primed, dummy_fingerprint, "GREETING"
            else:
                return "Exocortex > I do not have verified information about that.", dummy_primed, dummy_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        # Resume standard token-linker and context building
        active_entities = self.link_entities(query)

        # HDC-based Pronoun Coreference Resolution [1, 28]
        if not active_entities:
            pronouns = {"he", "she", "his", "her", "him", "they", "them", "it"}
            query_words = set(re.sub(r"[^\w\s]", "", query).lower().split())
            if query_words.intersection(pronouns):
                # Check top active context concept currently echoing in HDC memory
                fingerprint = self.hdc.get_context_fingerprint(top_k=1)
                if fingerprint:
                    closest_entity, similarity = fingerprint[0]
                    logger.info(f"HDC Coreference Resolution: Resolved pronoun to active context concept '{closest_entity}' (Similarity: {similarity:.4f})")
                    active_entities.add(closest_entity)

        hdc_fingerprint = self.process_hdc_context(query, active_entities)
        resolved_value = None
        source_id, predicate = None, None

        # 2. QUERY PATHWAY: If input is classified as a question, execute retrieval and return early!
        if is_query:
            if active_entities:
                # Retrieve ALL facts involving linked entities from SQL
                candidate_facts = self.kg.get_all_facts_for_entities(active_entities)

                # Execute Multiple-Choice Fact Selector to retrieve ALL qualified facts [1]
                matched_facts = self.select_answering_facts(query, candidate_facts)
                if matched_facts:
                    # Update Hebbian associations for all active elements + resolved targets [3]
                    active_update_set = active_entities.copy()
                    for s, p, o, _ in matched_facts:
                        active_update_set.add(s)
                        active_update_set.add(o)
                    self.plasticity.update_associations(active_update_set)

                    # Format all matching facts above the threshold into a unified string [1]
                    if len(matched_facts) == 1:
                        s, p, o, _ = matched_facts[0]
                        facts_str = f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}]"
                        source_id = s
                    else:
                        facts_str = " | ".join([f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}]" for s, p, o, _ in matched_facts])
                        source_id = matched_facts[0][0]

                    primed_info = self.plasticity.get_associated_priming_context(source_id)

                    # Retrieve prompt templates based on verbosity mode [1]
                    system_prompt, render_prompt = self._get_mode_prompts(query, facts_str, primed_info, hdc_fingerprint)

                    llm_response = self.query_ollama(render_prompt, system_prompt)
                    if llm_response:
                        return f"Exocortex (Ollama-Renderer) > {llm_response}", primed_info, hdc_fingerprint, "RENDER_SUCCESS"
                    else:
                        # Fallback compiler
                        return f"Exocortex (Simulated) > Handshake resolved: {facts_str}.", primed_info, hdc_fingerprint, "RENDER_FALLBACK"

            # Strict Question Gate: It's a query but no facts resolved. Return early to block fallback to extraction!
            return "Exocortex > I do not have verified information about that.", [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        # 3. EXTRACTION PATHWAY: If input is not a question, run the autonomous learning gate
        # Fixed (Bug 3): Unified conversational turn to use the secure Two-Pass extraction [1, 23]
        extracted_fact = self.extract_factual_declaration_two_pass(query)
        if extracted_fact:
            sub = extracted_fact["subject"]
            pred = extracted_fact["predicate"]
            obj = extracted_fact["object"]

            # Normalize predicate to prevent SQL schema clumping
            norm_pred = self.predicate_map.get(pred, pred)

            # Auto-register relation and update associations
            self.kg.update_relation(sub, norm_pred, obj)
            self.plasticity.update_associations({sub, obj})
            self.hdc.get_or_allocate_hypervector(sub)
            self.hdc.get_or_allocate_hypervector(obj)

            return f"Exocortex (Autonomous Learner) > I have recorded a new factual declaration: [{sub.replace('_', ' ')}] -[{norm_pred}]-> [{obj.replace('_', ' ')}].", [], hdc_fingerprint, "EXTRACT_SUCCESS"

        # 4. Safe Gated Fallback for non-factual declaratives
        return "Exocortex > I do not have verified information about that.", [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"


# ==========================================
# INTERACTIVE TERMINAL LOOP
# ==========================================

if __name__ == "__main__":
    db_file = "exocortex_kg.db"

    # Check if database already exists for persistent tracking [1]
    db_exists = os.path.exists(db_file)
    exocortex = IntegratedExocortex(db_file)

    if db_exists:
         # Query and print current entity count to verify persistent brain is working [1]
         count = exocortex.kg.get_entity_count()
         logger.info(f"Persistent Exocortex Database loaded successfully. Resuming from existing knowledge base with {count} unique entities.")
    else:
         logger.info("Initializing new persistent Exocortex Database.")

    print("\n========================================================")
    print("      EXOCORTEX NEURO-SYMBOLIC CHATBOT INITIALIZED      ")
    print("========================================================")
    print("Ask me factual questions! (e.g. 'Where was Marie Curie born?')")
    print("To teach me new things, just speak factually to me!")
    print("  example: 'Einstein was born in Germany' or 'Paris is the capital of France'")
    print("To ingest a local document (TXT or PDF) in bulk, type:")
    print("  /ingest [filename.ext] [fast / thorough]  (default: fast)")
    print("To switch personality modes live, type:")
    print("  /mode [strict / balanced / conversational]")
    print("To deliberately clear and re-initialize the database, type:")
    print("  /reset")
    print("Type 'exit' or 'quit' to shut down.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
                print("Safely shutting down local exocortex.")
                break

            # Intercept Personality Mode Commands (Problem 2)
            if user_input.startswith("/mode"):
                 parts = user_input.split()
                 if len(parts) == 2:
                      mode_name = parts[1].strip().upper()
                      if mode_name in ["STRICT", "BALANCED", "CONVERSATIONAL"]:
                           exocortex.verbosity_mode = mode_name
                           print(f"Exocortex [SYSTEM]: Verbosity mode set to [{mode_name}] successfully.")
                      else:
                           print("Exocortex [SYSTEM]: Error. Modes available: strict, balanced, conversational.")
                 else:
                      print("Exocortex [SYSTEM]: Error. Format is: /mode [strict/balanced/conversational]")
                 continue

            # Intercept Deliberate Database Reset Command (Problem 3)
            if user_input.strip() == "/reset":
                 print("Exocortex [SYSTEM]: Initiating deliberate database reset...")

                 # SQL-level drop-and-reinit completely immune to Windows file lock issues [1]
                 exocortex.kg.clear_and_reinitialize()

                 # Reset the HDC reservoir state and codebook
                 exocortex.hdc.state = np.zeros(exocortex.hdc.d, dtype=np.float64)
                 exocortex.hdc.codebook.clear()
                 exocortex.hdc.vocab_book.clear()

                 # Re-seed HDC codebook with newly seeded SQL entities
                 for ent_id in exocortex.kg.get_all_entity_ids():
                     exocortex.hdc.get_or_allocate_hypervector(ent_id)

                 print("Exocortex [SYSTEM]: Database has been cleanly reset, re-seeded, and HDC state cleared.")
                 continue

            # Intercept Bulk Document Ingestion (TXT or PDF)
            if user_input.startswith("/ingest"):
                parts = user_input.split()
                if len(parts) >= 2:
                    filename = parts[1].strip()
                    mode = "fast"  # default mode
                    if len(parts) >= 3:
                        mode = parts[2].strip().lower()

                    fast_mode = (mode == "fast")
                    print(f"Exocortex [SYSTEM]: Initiating bulk ingestion for '{filename}' (Mode: {mode.upper()})...")
                    result = exocortex.ingest_document(filename, fast_mode=fast_mode)
                    print(f"Exocortex [SYSTEM]: {result}")
                else:
                    print("Exocortex [SYSTEM]: Error. Correct format is: /ingest [filename.ext] [fast/thorough]")
                continue

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