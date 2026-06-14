"""Manages gradient-free co-activation associations (Synaptic Memory)."""

import sqlite3
from typing import List, Tuple, Set
from config import DB_FILE, HEBBIAN_ETA, HEBBIAN_DECAY


class HebbianPlasticityEngine:
    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self.eta = HEBBIAN_ETA
        self.decay = HEBBIAN_DECAY

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

                    cursor.execute("SELECT weight FROM hebbian_weights WHERE entity_a = ? AND entity_b = ?",
                                   (ent_a, ent_b))
                    row = cursor.fetchone()
                    current_w = row[0] if row else 0.0
                    new_w = current_w + self.eta * (1.0 - current_w)
                    cursor.execute("""
                                   INSERT INTO hebbian_weights (entity_a, entity_b, weight)
                                   VALUES (?, ?, ?) ON CONFLICT(entity_a, entity_b) DO
                                   UPDATE SET weight = excluded.weight
                                   """, (ent_a, ent_b, new_w))

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
                           SELECT entity_b, weight
                           FROM hebbian_weights
                           WHERE entity_a = ?
                             AND weight > ?
                           UNION
                           SELECT entity_a, weight
                           FROM hebbian_weights
                           WHERE entity_b = ?
                             AND weight > ?
                           ORDER BY weight DESC
                           """, (entity, threshold, entity, threshold))
            return cursor.fetchall()