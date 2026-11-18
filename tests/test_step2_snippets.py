import gc
import os
import sqlite3
import sys
import tempfile
import time
import unittest
from contextlib import closing
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import ingestor
from database import SQLiteKnowledgeGraph
from engine import IntegratedHillock
from talon_engine import TalonEngine


class Step2SnippetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "step2_test.db")
        self.repo_temp_files = []

    def tearDown(self):
        for p in self.repo_temp_files:
            if os.path.exists(p):
                os.remove(p)
        gc.collect()
        self.temp_dir.cleanup()

    def test_non_destructive_migration_on_legacy_db(self):
        legacy_db = os.path.join(self.temp_dir.name, "legacy_v1_1.db")
        with closing(sqlite3.connect(legacy_db)) as conn, conn:
            conn.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL)")
            conn.execute("""
                CREATE TABLE relations (
                    source_id TEXT,
                    predicate TEXT,
                    target_id TEXT,
                    source_doc TEXT DEFAULT 'manual_entry',
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, predicate, target_id)
                )
            """)
            conn.execute("INSERT INTO entities VALUES ('Paris', 'Paris', 'City')")
            conn.execute("INSERT INTO entities VALUES ('France', 'France', 'Country')")
            conn.execute("INSERT INTO relations (source_id, predicate, target_id) VALUES ('Paris', 'capital_of', 'France')")

        kg = SQLiteKnowledgeGraph(legacy_db)
        self.assertEqual(kg.get_snippet_count(), 0)
        facts = kg.get_all_facts_for_entities({"Paris"})
        self.assertEqual(facts, [("Paris", "capital_of", "France", "manual_entry")])

        with closing(sqlite3.connect(legacy_db)) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(relations)").fetchall()}
        self.assertIn("snippet_id", cols)

    def test_snippet_dedup_linking_and_4tuple_backward_compatibility(self):
        kg = SQLiteKnowledgeGraph(self.db_path)
        sid1 = kg.save_snippet("notes.txt", "Resistor limits current in a series loop.", "page 1")
        sid1_dup = kg.save_snippet("notes.txt", "Resistor limits current in a series loop.", "page 1")
        sid2 = kg.save_snippet("notes.txt", "Resistor limits current in a series loop.", "page 2")

        self.assertEqual(sid1, sid1_dup)
        self.assertNotEqual(sid1, sid2)
        self.assertEqual(kg.get_snippet_count(), 2)

        with self.assertRaises(ValueError):
            kg.update_relations_batch([("Resistor", "limits", "Current")], snippet_ids=[])

        kg.update_relations_batch(
            [("Resistor", "limits", "Current")],
            source_doc="notes.txt",
            snippet_ids=[sid1],
        )

        # Must remain exact 4-tuple for benchmark_labyrinth / Aimee compatibility
        facts = kg.get_all_facts_for_entities({"Resistor"})
        self.assertEqual(facts, [("Resistor", "limits", "Current", "notes.txt")])

        ent_snips = kg.get_snippets_for_entities({"Resistor"})
        self.assertEqual(len(ent_snips), 1)
        self.assertEqual(ent_snips[0], (sid1, "notes.txt", "page 1", "Resistor limits current in a series loop."))

        fact_snips = kg.get_snippets_for_facts([("Resistor", "limits", "Current")])
        self.assertEqual(fact_snips, ent_snips)

    def test_search_snippets_ranks_multi_term_hits_and_escapes_wildcards(self):
        kg = SQLiteKnowledgeGraph(self.db_path)
        kg.save_snippet("elec.pdf", "Voltage is measured in volts.", "page 1")
        kg.save_snippet("elec.pdf", "Ohm law relates voltage, current, and resistance directly.", "page 2")
        kg.save_snippet("misc.txt", "100% pass_rate achieved today.", "lines 1-1")

        ranked = kg.search_snippets(["voltage", "resistance", "current"], limit=5)
        self.assertEqual(len(ranked), 2)
        self.assertIn("resistance", ranked[0][3].lower())

        escaped = kg.search_snippets(["100%", "pass_rate"], limit=5)
        self.assertEqual(len(escaped), 1)
        self.assertIn("100% pass_rate", escaped[0][3])

    def test_clear_and_reinitialize_wipes_snippets(self):
        kg = SQLiteKnowledgeGraph(self.db_path)
        kg.save_snippet("doc.txt", "Temporary snippet content.", "lines 1-1")
        self.assertEqual(kg.get_snippet_count(), 1)
        kg.clear_and_reinitialize()
        self.assertEqual(kg.get_snippet_count(), 0)

    def test_talon_process_document_attaches_context_snippet(self):
        talon = object.__new__(TalonEngine)
        talon.device = "cpu"
        talon.t_first_triple = None
        talon.t_last_triple = None
        talon.coref = MagicMock()
        talon.coref.resolve_text.side_effect = lambda text: text
        talon.router = MagicMock()
        talon.router.select_top_predicates_batch.return_value = [["born_in"], ["discovered"]]
        talon.extractor = MagicMock()
        talon.extractor.extract_relations.side_effect = [
            [],
            [{"subject": "Marie Curie", "predicate": "discovered", "object": "Radioactivity"}],
        ]

        triples = talon.process_document("Marie Curie lived in Paris. Marie Curie discovered Radioactivity.")
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0]["sentence_index"], 1)
        self.assertEqual(triples[0]["source_sentence"], "Marie Curie discovered Radioactivity")
        self.assertIn("Marie Curie lived in Paris", triples[0]["context_snippet"])
        self.assertIn("Marie Curie discovered Radioactivity", triples[0]["context_snippet"])

    def test_ingestor_blocks_path_traversal_and_saves_snippets_with_mocked_talon(self):
        with self.assertRaises(ValueError):
            ingestor.get_raw_document_segments("../../outside_secret.txt")

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=REPO_ROOT,
            prefix="_tmp_step2_ingest_",
            suffix=".txt",
            delete=False,
        ) as tf:
            tf.write(
                "Capacitors store electric energy in an electric field.\n"
                "Ada Lovelace wrote the first computer algorithm in London.\n"
            )
            sample_path = tf.name
        self.repo_temp_files.append(sample_path)

        hillock = IntegratedHillock(self.db_path)

        fake_talon = MagicMock()
        fake_talon.device = "cpu"
        fake_talon.extractor.model = object()
        fake_talon.t_first_triple = time.perf_counter()
        fake_talon.t_last_triple = fake_talon.t_first_triple + 0.05
        fake_talon.process_document.return_value = [
            {
                "subject": "Ada Lovelace",
                "predicate": "authored",
                "object": "first computer algorithm",
                "source_sentence": "Ada Lovelace wrote the first computer algorithm in London",
                "context_snippet": "Capacitors store electric energy in an electric field. Ada Lovelace wrote the first computer algorithm in London.",
            }
        ]

        with patch("ingestor.get_talon_engine", return_value=fake_talon):
            summary, stats = ingestor.ingest_document_parallel(sample_path, hillock)

        self.assertIn("Source Snippets Stored", summary)
        self.assertEqual(stats["extracted_triples"], 1)
        self.assertGreaterEqual(stats["snippets_saved"], 1)
        self.assertGreaterEqual(hillock.kg.get_snippet_count(), 1)

        fact_snips = hillock.kg.get_snippets_for_facts(
            [("Ada_Lovelace", "authored", "first_computer_algorithm")]
        )
        self.assertEqual(len(fact_snips), 1)
        self.assertIn("Ada Lovelace wrote the first computer algorithm", fact_snips[0][3])

        # Also verify raw prose without a triple is still searchable via snippet window
        prose_hits = hillock.kg.search_snippets(["capacitors", "electric"])
        self.assertGreaterEqual(len(prose_hits), 1)


if __name__ == "__main__":
    unittest.main()