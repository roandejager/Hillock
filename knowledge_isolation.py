"""
Knowledge Isolation System: Phase 1
An implementation of Decoupled Knowledge Editing & model Deliberation (DecKER)
integrated with a local Hebbian Co-activation Plasticity Engine.

This script operates entirely without global backpropagation or weight training,
optimizing reasoning paths and factual retrieval for local hardware.
"""

import sqlite3
import re
import math
import logging
from typing import List, Dict, Tuple, Set, Optional

# Set up clean logging for step-by-step verification
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KnowledgeIsolation")


class SQLiteKnowledgeGraph:
    """
    Manages the decoupled symbolic factual database.
    This separates static factual knowledge from model parameter spaces.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initializes tables for entities, relations, and Hebbian weights."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Entities: explicit concepts and their types
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL
                )
            """)

            # Relations: directed triples representing factual statements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    source_id TEXT,
                    predicate TEXT,
                    target_id TEXT,
                    PRIMARY KEY (source_id, predicate, target_id),
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            """)

            # Hebbian Association Weights (Plasticity tracking)
            # We enforce entity_a < entity_b alphabetically to avoid duplicate reciprocal rows
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hebbian_weights (
                    entity_a TEXT,
                    entity_b TEXT,
                    weight REAL DEFAULT 0.0,
                    PRIMARY KEY (entity_a, entity_b),
                    FOREIGN KEY (entity_a) REFERENCES entities(id),
                    FOREIGN KEY (entity_b) REFERENCES entities(id)
                )
            """)
            conn.commit()

    def seed_initial_knowledge(self) -> None:
        """Seeds sample entity and relation data for evaluation."""
        entities = [
            ("France", "France", "Country"),
            ("Paris", "Paris", "City"),
            ("London", "London", "City"),
            ("UK", "United Kingdom", "Country"),
            ("Marie_Curie", "Marie Curie", "Person"),
            ("Poland", "Poland", "Country"),
            ("Radioactivity", "Radioactivity", "Scientific Field"),
            ("Nobel_Prize", "Nobel Prize", "Award")
        ]

        relations = [
            ("Paris", "capital_of", "France"),
            ("London", "capital_of", "UK"),
            ("Marie_Curie", "born_in", "Poland"),
            ("Marie_Curie", "discovered", "Radioactivity"),
            ("Marie_Curie", "won", "Nobel_Prize")
        ]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany("INSERT OR IGNORE INTO entities VALUES (?, ?, ?)", entities)
            cursor.executemany("INSERT OR IGNORE INTO relations VALUES (?, ?, ?)", relations)
            conn.commit()
        logger.info("Decoupled Knowledge Graph seeded successfully.")

    def update_relation(self, source_id: str, predicate: str, new_target_id: str) -> None:
        """
        Dynamically edits or inserts a factual record.
        This represents zero-shot 'hot-editing' of model knowledge.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete old matching relations to avoid conflicts during updates
            cursor.execute(
                "DELETE FROM relations WHERE source_id = ? AND predicate = ?",
                (source_id, predicate)
            )
            cursor.execute(
                "INSERT INTO relations (source_id, predicate, target_id) VALUES (?, ?, ?)",
                (source_id, predicate, new_target_id)
            )
            conn.commit()
        logger.warning(f"KNOWLEDGE EDIT: Modified fact [{source_id} -{predicate}-> {new_target_id}]")

    def query_relation(self, source_id: str, predicate: str) -> Optional[str]:
        """Queries a relation target based on source and predicate."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT target_id FROM relations WHERE source_id = ? AND predicate = ?",
                (source_id, predicate)
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_all_entity_ids(self) -> List[str]:
        """Returns all registered entity IDs for the entity linker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM entities")
            return [row[0] for row in cursor.fetchall()]


class HebbianPlasticityEngine:
    """
    Implements local non-connectionist synaptic updates.
    Increments weights for entities that 'fire together' (co-occur in a query/response sequence)
    and decays unused weights to prevent representational drift.
    """

    def __init__(self, db_path: str = ":memory:", eta: float = 0.1, decay: float = 0.01):
        self.db_path = db_path
        self.eta = eta  # Learning rate (plasticity)
        self.decay = decay  # Synaptic decay rate (preventing saturation)

    def update_associations(self, active_entities: Set[str]) -> None:
        """
        Applies a local Hebbian update step.
        For all active pairs: w_new = w_old + eta * (1.0 - w_old).
        For all other pairs in the database: w_new = w_old * (1.0 - decay).
        """
        if len(active_entities) < 2:
            self._apply_global_decay()
            return

        sorted_entities = sorted(list(active_entities))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Step 1: Apply local Hebbian co-activation updates
            for i in range(len(sorted_entities)):
                for j in range(i + 1, len(sorted_entities)):
                    ent_a, ent_b = sorted_entities[i], sorted_entities[j]

                    # Fetch current weight
                    cursor.execute(
                        "SELECT weight FROM hebbian_weights WHERE entity_a = ? AND entity_b = ?",
                        (ent_a, ent_b)
                    )
                    row = cursor.fetchone()
                    current_w = row[0] if row else 0.0

                    # Hebbian rule: shift weight closer to 1.0 based on concurrent activity
                    new_w = current_w + self.eta * (1.0 - current_w)

                    cursor.execute("""
                        INSERT INTO hebbian_weights (entity_a, entity_b, weight)
                        VALUES (?, ?, ?)
                        ON CONFLICT(entity_a, entity_b) DO UPDATE SET weight = excluded.weight
                    """, (ent_a, ent_b, new_w))

            # Step 2: Decay weights of inactive connections to maintain stability
            cursor.execute("SELECT entity_a, entity_b, weight FROM hebbian_weights")
            all_weights = cursor.fetchall()
            for ent_a, ent_b, weight in all_weights:
                if ent_a not in active_entities or ent_b not in active_entities:
                    decayed_w = weight * (1.0 - self.decay)
                    cursor.execute(
                        "UPDATE hebbian_weights SET weight = ? WHERE entity_a = ? AND entity_b = ?",
                        (decayed_w, ent_a, ent_b)
                    )
            conn.commit()
        logger.info(f"Hebbian plasticity update applied for active context: {active_entities}")

    def _apply_global_decay(self) -> None:
        """Applies basic synaptic decay to all weights when active context is empty."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE hebbian_weights SET weight = weight * ?", (1.0 - self.decay,))
            conn.commit()

    def get_associated_priming_context(self, entity: str, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Retrieves highly-associated adjacent concepts to 'prime' the context window.
        Simulates how local Hebbian circuits prepare adjacent nodes for downstream inference.
        """
        associations = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_b, weight FROM hebbian_weights WHERE entity_a = ? AND weight > ?
                UNION
                SELECT entity_a, weight FROM hebbian_weights WHERE entity_b = ? AND weight > ?
                ORDER BY weight DESC
            """, (entity, threshold, entity, threshold))
            associations = cursor.fetchall()
        return associations


class DecKERPipeline:
    """
    Decoupling Knowledge Editing and model Deliberation (DecKER) Engine.
    Coordinates structural reasoning, path parsing, and factual verification.
    """

    def __init__(self, kg: SQLiteKnowledgeGraph, plasticity_engine: HebbianPlasticityEngine):
        self.kg = kg
        self.plasticity_engine = plasticity_engine

        # Simple structural rule mapping to simulate local Mamba/RWKV template output
        self.templates = [
            (r"where was (.+?) born", "born_in"),
            (r"what is the capital of (.+)", "capital_of"),
            (r"who discovered (.+)", "discovered"),  # Backwards relation resolution check
            (r"what did (.+?) discover", "discovered"),
            (r"what awards did (.+?) win", "won"),
            (r"what did (.+?) win", "won")
        ]

    def link_entities(self, query: str) -> Set[str]:
        """Parses the raw context string and links it to known entities in the database."""
        detected = set()
        all_entities = self.kg.get_all_entity_ids()

        # Replace punctuation to ease matching
        normalized_query = re.sub(r"[^\w\s]", " ", query).lower()

        for entity_id in all_entities:
            # Match entities replacing underscore with space
            readable_name = entity_id.replace("_", " ").lower()
            if re.search(r"\b" + re.escape(readable_name) + r"\b", normalized_query):
                detected.add(entity_id)
        return detected

    def generate_masked_path(self, query: str, active_entities: Set[str]) -> Optional[Tuple[str, str, str]]:
        for pattern, predicate in self.templates:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted_name = match.group(1).strip()
                # Clean up trailing punctuation (e.g., 'France?' becomes 'France')
                extracted_name = re.sub(r"[^\w\s]", "", extracted_name)
                extracted_name = extracted_name.lower().replace(" ", "_")

                matched_entity = None
                for ent in active_entities:
                    if ent.lower() == extracted_name:
                        matched_entity = ent
                        break

                if matched_entity:
                    logger.info(
                        f"DecKER Stage 1: Masked path generated -> [{matched_entity}] -[{predicate}]-> [PLACEHOLDER_1]")
                    return (matched_entity, predicate, "PLACEHOLDER_1")
        return None

    def resolve_factual_binding(self, masked_path: Tuple[str, str, str]) -> Optional[str]:
        """
        Stage 2 of DecKER: Retrieve verified values from the external database
        instead of extracting them from neural weights.
        """
        source_id, predicate, placeholder = masked_path
        resolved_value = self.kg.query_relation(source_id, predicate)
        if resolved_value:
            logger.info(f"DecKER Stage 2: Factual placeholder resolved [{placeholder} -> {resolved_value}]")
            return resolved_value
        logger.warning(f"DecKER Stage 2: Failed to resolve factual placeholder for predicate '{predicate}'")
        return None

    def evaluate_and_respond(self, source_entity: str, predicate: str, target_entity: str) -> str:
        """Stage 3 of DecKER: Evaluates logical paths and generates the grounded output response."""
        # Convert internal entity IDs back to human-readable names
        source_name = source_entity.replace("_", " ")
        target_name = target_entity.replace("_", " ")

        # Structure templates based on semantic relation rules
        if predicate == "capital_of":
            return f"The capital of {source_name} is {target_name}."
        elif predicate == "born_in":
            return f"{source_name} was born in {target_name}."
        elif predicate == "discovered":
            return f"{source_name} discovered {target_name}."
        elif predicate == "won":
            return f"{source_name} won the {target_name}."

        return f"Relation verified: {source_name} {predicate} {target_name}."

    def process_query(self, query: str) -> str:
        """Orchestrates the entire DecKER pipeline and triggers Hebbian co-activation."""
        logger.info(f"\n--- Processing Query: '{query}' ---")

        # Step 1: Parse query context to link registered entities
        active_entities = self.link_entities(query)
        logger.info(f"Linked Entities: {active_entities}")

        if not active_entities:
            return "Query analysis completed: No recognizable entities found in local Knowledge Graph."

        # Step 2: Construct abstract reasoning path (DecKER Stage 1)
        masked_path = self.generate_masked_path(query, active_entities)
        if not masked_path:
            return "DecKER Error: Could not compile query into a structured reasoning path."

        # Step 3: Factual Resolution Lookup (DecKER Stage 2)
        resolved_target = self.resolve_factual_binding(masked_path)
        if not resolved_target:
            return f"Resolution failure: No database path found for {masked_path[0]} -> {masked_path[1]}"

        # Step 4: Final Candidate Response Synthesis (DecKER Stage 3)
        source_ent, predicate, _ = masked_path
        response = self.evaluate_and_respond(source_ent, predicate, resolved_target)

        # Step 5: Hebbian Plasticity Step
        # Both the source and resolved targets are activated in local working memory
        transaction_entities = {source_ent, resolved_target}
        self.plasticity_engine.update_associations(transaction_entities)

        return response


# --- EXECUTION DEMONSTRATION LOOP ---
if __name__ == "__main__":
    import os

    # Change from ":memory:" to a physical database file
    db_file = "exocortex_kg.db"

    # Clean up any old database from previous failed runs to ensure a clean start
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            logger.info(f"Removed old database file '{db_file}' for a clean run.")
        except PermissionError:
            logger.warning(f"Could not remove locked database file '{db_file}'. Running with existing database.")

    # 1. Initialize Decoupled Architecture Components
    kg = SQLiteKnowledgeGraph(db_file)
    kg.seed_initial_knowledge()

    # Explicitly pass the same database path to the plasticity engine
    plasticity = HebbianPlasticityEngine(db_file, eta=0.15, decay=0.01)
    pipeline = DecKERPipeline(kg, plasticity)

    # 2. Test DecID-DecKER Processing
    # Marie Curie birthplace query
    ans_1 = pipeline.process_query("Where was Marie Curie born?")
    print(f"\n[SYSTEM RESPONSE]: {ans_1}")

    # Capital of France query
    ans_2 = pipeline.process_query("What is the capital of France?")
    print(f"\n[SYSTEM RESPONSE]: {ans_2}")

    # 3. Verify Hebbian Priming Mechanisms
    logger.info("\n--- Simulating Local Co-activation Buildup ---")
    for _ in range(3):
        pipeline.process_query("What did Marie Curie discover?")

    logger.info("\n--- Evaluating Contextual Priming Space for 'Marie_Curie' ---")
    primed_nodes = plasticity.get_associated_priming_context("Marie_Curie")
    print(f"Active Memory Priming Nodes for 'Marie_Curie':")
    for node, weight in primed_nodes:
        print(f" -> Concept: {node:<15} Synaptic Weight: {weight:.4f}")

    # 4. Demonstrate Zero-Shot Knowledge Editing (Decoupled Update Proof)
    logger.info("\n--- Performing Zero-Shot Knowledge Edit (Without Parameter Re-training) ---")

    pre_edit = pipeline.process_query("Where was Marie Curie born?")
    print(f"[Before Edit]: {pre_edit}")

    # Update the target in the SQLite database
    kg.update_relation("Marie_Curie", "born_in", "France")

    post_edit = pipeline.process_query("Where was Marie Curie born?")
    print(f"[After Edit]: {post_edit}")

    logger.info("\nExecution complete. Phase 1 structural logic successfully established.")

