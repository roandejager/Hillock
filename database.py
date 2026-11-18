"""SQLite Database module representing the decoupled fact storage."""

import hashlib
import sqlite3
import re
import numpy as np
from contextlib import contextmanager
from typing import Iterator, List, Tuple, Optional, Sequence, Set
from config import DB_FILE, SINGLE_VALUED_PREDICATES

class SQLiteKnowledgeGraph:
    """Manages the decoupled symbolic database (Filing Cabinet)."""
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def _connect(self, foreign_keys: bool = True) -> Iterator[sqlite3.Connection]:
        """Opens a SQLite connection, runs a transaction, and always closes the handle."""
        conn = sqlite3.connect(self.db_path)
        try:
            if foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA synchronous = NORMAL;")
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
                    source_doc TEXT DEFAULT 'manual_entry',
                    snippet_id TEXT DEFAULT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
            # HYPERGRAPH-HDC: New table for multi-hop vector blobs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hdc_reservoirs (
                    doc_id TEXT PRIMARY KEY,
                    reservoir_vector BLOB NOT NULL,
                    bound_path_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # v0.10: Entity Aliasing Layer
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    alias_id TEXT PRIMARY KEY,
                    canonical_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (canonical_id) REFERENCES entities(id) ON DELETE CASCADE
                )
            """)
            # v1.2: Source snippets for human-readable context in FREE mode
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS source_snippets (
                    id TEXT PRIMARY KEY,
                    source_doc TEXT NOT NULL,
                    page_or_line TEXT DEFAULT '',
                    raw_text TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("PRAGMA table_info(relations)")
            existing_relation_cols = {row[1] for row in cursor.fetchall()}
            if "snippet_id" not in existing_relation_cols:
                cursor.execute("ALTER TABLE relations ADD COLUMN snippet_id TEXT DEFAULT NULL")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_relations_snippet_id ON relations(snippet_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snippets_source_doc ON source_snippets(source_doc)")
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
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?)", entities)
            cursor.executemany("INSERT OR IGNORE INTO relations (source_id, predicate, target_id) VALUES (?, ?, ?)", relations)
            conn.commit()

    def clear_and_reinitialize(self) -> None:
        """Safe SQL-level reset."""
        with self._connect(foreign_keys=False) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = OFF;")
            cursor.execute("DROP TABLE IF EXISTS hebbian_weights;")
            cursor.execute("DROP TABLE IF EXISTS relations;")
            cursor.execute("DROP TABLE IF EXISTS entity_aliases;")
            cursor.execute("DROP TABLE IF EXISTS entities;")
            cursor.execute("DROP TABLE IF EXISTS hdc_reservoirs;")
            cursor.execute("DROP TABLE IF EXISTS source_snippets;")
            conn.commit()
        self._initialize_db()
        self.seed_initial_knowledge()

    def get_entity_count(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM entities")
            return cursor.fetchone()[0]

    def get_relations_count(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM relations")
            return cursor.fetchone()[0]

    def get_synapse_count(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM hebbian_weights")
            return cursor.fetchone()[0]

    def update_relation(self, source_id: str, predicate: str, new_target_id: str,
                        source_type: str = "Generic", target_type: str = "Generic") -> None:
        """Single relation insertion method for conversational turns."""
        self.update_relations_batch([(source_id, predicate, new_target_id)], source_type, target_type)

    def update_relations_batch(
        self,
        triples: List[Tuple[str, str, str]],
        source_type: str = "Generic",
        target_type: str = "Generic",
        source_doc: str = "manual_entry",
        snippet_ids: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        if not triples:
            return

        if snippet_ids is not None and len(snippet_ids) != len(triples):
            raise ValueError("snippet_ids length must match triples length.")

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION;")
                for idx, (source_id, predicate, new_target_id) in enumerate(triples):
                    src_key = source_id.strip().replace(" ", "_")
                    tgt_key = new_target_id.strip().replace(" ", "_")
                    snip_id = snippet_ids[idx] if snippet_ids is not None else None

                    cursor.execute(
                        "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, ?)",
                        (src_key, src_key.replace("_", " "), source_type),
                    )
                    cursor.execute(
                        "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, ?)",
                        (tgt_key, tgt_key.replace("_", " "), target_type),
                    )

                    if predicate in SINGLE_VALUED_PREDICATES:
                        cursor.execute(
                            "DELETE FROM relations WHERE source_id = ? AND predicate = ?",
                            (src_key, predicate),
                        )

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO relations (source_id, predicate, target_id, source_doc, snippet_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (src_key, predicate, tgt_key, source_doc, snip_id),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def query_relation(self, source_id: str, predicate: str) -> Optional[str]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT target_id FROM relations WHERE source_id = ? AND predicate = ?", (source_id, predicate))
            result = cursor.fetchone()
            if result:
                return result[0]

            cursor.execute("SELECT predicate, target_id FROM relations WHERE source_id = ?", (source_id,))
            all_relations = cursor.fetchall()

            def stem(s: str) -> str:
                s = s.lower().replace("_", "").replace(" ", "")
                return re.sub(r"ed$|ing$|s$", "", s)

            stemmed_query = stem(predicate)
            for db_pred, tgt_id in all_relations:
                if stemmed_query in stem(db_pred) or stem(db_pred) in stemmed_query:
                    return tgt_id
            return None

    def get_all_facts_for_entities(self, active_entities: Set[str]) -> List[Tuple[str, str, str, str]]:
        if not active_entities:
            return []
        placeholders = ", ".join(["?"] * len(active_entities))
        query = f"SELECT source_id, predicate, target_id, source_doc FROM relations WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
        params = list(active_entities) + list(active_entities)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_all_entity_ids(self) -> List[str]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entities")
            return [row[0] for row in cursor.fetchall()]

    def save_document_reservoir(self, doc_id: str, vector: np.ndarray, path_count: int) -> None:
        """Stores the binarized multi-hop document reservoir into SQLite as a compact BLOB."""
        # Convert to bipolar int8 to save space, then to bytes for SQLite
        bipolar_vector = np.sign(vector).astype(np.int8)
        bipolar_vector[bipolar_vector == 0] = 1
        vector_bytes = bipolar_vector.tobytes()
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO hdc_reservoirs (doc_id, reservoir_vector, bound_path_count) 
                VALUES (?, ?, ?)
            """, (doc_id, vector_bytes, path_count))
            conn.commit()

    def get_canonical_id(self, entity_id: str) -> Optional[str]:
        """Checks if an entity string is a known alias and returns its canonical root."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT canonical_id FROM entity_aliases WHERE alias_id = ?", (entity_id,))
            res = cursor.fetchone()
            return res[0] if res else None

    def add_alias(self, alias_id: str, canonical_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO entity_aliases (alias_id, canonical_id) VALUES (?, ?)", (alias_id, canonical_id))
            conn.commit()

    def merge_nodes(self, old_id: str, canonical_id: str) -> None:
        """Collapses old_id into canonical_id, migrating all relations and synapses."""
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION;")

                # 1. Migrate Relations (Source)
                cursor.execute("UPDATE OR IGNORE relations SET source_id = ? WHERE source_id = ?", (canonical_id, old_id))
                cursor.execute("DELETE FROM relations WHERE source_id = ?", (old_id,))

                # 2. Migrate Relations (Target)
                cursor.execute("UPDATE OR IGNORE relations SET target_id = ? WHERE target_id = ?", (canonical_id, old_id))
                cursor.execute("DELETE FROM relations WHERE target_id = ?", (old_id,))

                # 3. Migrate Synapses (Hebbian Weights)
                cursor.execute("UPDATE OR IGNORE hebbian_weights SET entity_a = ? WHERE entity_a = ?", (canonical_id, old_id))
                cursor.execute("DELETE FROM hebbian_weights WHERE entity_a = ?", (old_id,))
                cursor.execute("UPDATE OR IGNORE hebbian_weights SET entity_b = ? WHERE entity_b = ?", (canonical_id, old_id))
                cursor.execute("DELETE FROM hebbian_weights WHERE entity_b = ?", (old_id,))

                # 4. Register the Alias and Delete the Old Node
                cursor.execute("INSERT OR REPLACE INTO entity_aliases (alias_id, canonical_id) VALUES (?, ?)", (old_id, canonical_id))
                cursor.execute("DELETE FROM entities WHERE id = ?", (old_id,))

                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def make_snippet_id(source_doc: str, page_or_line: str, raw_text: str) -> str:
        """Generates a deterministic content ID so re-ingesting does not duplicate snippets."""
        material = f"{source_doc}|{page_or_line}|{raw_text.strip()}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:24]

    @staticmethod
    def _escape_like(term: str) -> str:
        return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def save_snippets_batch(self, snippets: List[Tuple[str, str, str, str]]) -> None:
        """Batch-inserts (snippet_id, source_doc, page_or_line, raw_text) rows."""
        cleaned = [
            (sid, doc, loc, text.strip())
            for sid, doc, loc, text in snippets
            if sid and text and text.strip()
        ]
        if not cleaned:
            return

        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN TRANSACTION;")
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO source_snippets (id, source_doc, page_or_line, raw_text)
                    VALUES (?, ?, ?, ?)
                    """,
                    cleaned,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def save_snippet(
        self,
        source_doc: str,
        raw_text: str,
        page_or_line: str = "",
        snippet_id: Optional[str] = None,
    ) -> str:
        """Saves a single snippet and returns its deterministic ID."""
        cleaned_text = raw_text.strip()
        if not cleaned_text:
            raise ValueError("raw_text must not be empty.")
        sid = snippet_id or self.make_snippet_id(source_doc, page_or_line, cleaned_text)
        self.save_snippets_batch([(sid, source_doc, page_or_line, cleaned_text)])
        return sid

    def get_snippet_count(self) -> int:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM source_snippets")
            return cursor.fetchone()[0]

    def get_snippets_for_entities(
        self, active_entities: Set[str], limit: int = 6
    ) -> List[Tuple[str, str, str, str]]:
        """Returns distinct (id, source_doc, page_or_line, raw_text) linked to active entities."""
        if not active_entities or limit <= 0:
            return []
        placeholders = ", ".join(["?"] * len(active_entities))
        query = f"""
            SELECT DISTINCT s.id, s.source_doc, s.page_or_line, s.raw_text
            FROM relations r
            JOIN source_snippets s ON r.snippet_id = s.id
            WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
            ORDER BY s.created_at DESC, s.id ASC
            LIMIT ?
        """
        params = list(active_entities) + list(active_entities) + [limit]
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def get_snippets_for_facts(
        self, fact_triples: Sequence[Tuple[str, str, str]], limit: int = 6
    ) -> List[Tuple[str, str, str, str]]:
        """Returns distinct (id, source_doc, page_or_line, raw_text) linked to specific SPO facts."""
        if not fact_triples or limit <= 0:
            return []
        selected = list(fact_triples)[:25]
        clauses = " OR ".join(
            ["(r.source_id = ? AND r.predicate = ? AND r.target_id = ?)"] * len(selected)
        )
        params: List[object] = []
        for s, p, o in selected:
            params.extend([s, p, o])
        params.append(limit)
        query = f"""
            SELECT DISTINCT s.id, s.source_doc, s.page_or_line, s.raw_text
            FROM relations r
            JOIN source_snippets s ON r.snippet_id = s.id
            WHERE ({clauses})
            ORDER BY s.created_at DESC, s.id ASC
            LIMIT ?
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def search_snippets(
        self, terms: Sequence[str], limit: int = 5
    ) -> List[Tuple[str, str, str, str]]:
        """Keyword fallback search across raw_text ranked by number of matching terms."""
        if limit <= 0:
            return []
        clean_terms = list(
            dict.fromkeys(
                t.strip().lower() for t in terms if isinstance(t, str) and len(t.strip()) >= 3
            )
        )[:6]
        if not clean_terms:
            return []

        like_patterns = [f"%{self._escape_like(t)}%" for t in clean_terms]
        score_expr = " + ".join(
            ["(CASE WHEN LOWER(raw_text) LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"]
            * len(like_patterns)
        )
        where_expr = " OR ".join(
            ["LOWER(raw_text) LIKE ? ESCAPE '\\'"] * len(like_patterns)
        )
        query = f"""
            SELECT id, source_doc, page_or_line, raw_text, ({score_expr}) AS hits
            FROM source_snippets
            WHERE {where_expr}
            ORDER BY hits DESC, created_at DESC, id ASC
            LIMIT ?
        """
        params = like_patterns + like_patterns + [limit]
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [row[:4] for row in cursor.fetchall()]