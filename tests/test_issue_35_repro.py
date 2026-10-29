import gc
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import talon_engine
from database import SQLiteKnowledgeGraph
from engine import IntegratedHillock
from talon_engine import ZeroShotRelationExtractor


class FakeToken:
    def __init__(self, text, lemma="", dep_="", head=None):
        self.text = text
        self.lemma_ = lemma or text.lower()
        self.dep_ = dep_
        self.head = head or self


class FakeSpan:
    def __init__(self, text, start, end, label_):
        self.text = text
        self.start = start
        self.end = end
        self.label_ = label_


class FakeDocument:
    def __init__(self):
        words = ["Ada", "Lovelace", "was", "not", "born", "in", "Berlin"]
        self.tokens = [FakeToken(word) for word in words]
        born = self.tokens[4]
        born.lemma_ = "bear"
        self.tokens[3].dep_ = "neg"
        self.tokens[3].head = born
        self.ents = [
            FakeSpan("Ada Lovelace", 0, 2, "PERSON"),
            FakeSpan("Berlin", 6, 7, "GPE"),
        ]
        self.noun_chunks = []

    def __iter__(self):
        return iter(self.tokens)


class FakeRelationModel:
    def predict_relations(self, *args, **kwargs):
        return [{"head_text": "Ada Lovelace", "tail_text": "Berlin", "label": "born_in"}]


class Issue35ReproTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "issue_35.db")
        self.engine = IntegratedHillock(self.db_path)

    def tearDown(self):
        self.engine = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed database")
            gc.collect()
        self.temp_dir.cleanup()

    def test_negated_birth_is_not_stored(self):
        extractor = ZeroShotRelationExtractor()
        extractor.model = FakeRelationModel()
        with patch.object(talon_engine, "nlp", return_value=FakeDocument()):
            triples = extractor.extract_relations(
                "Ada Lovelace was not born in Berlin",
                ["born_in"],
            )

        graph = SQLiteKnowledgeGraph(self.db_path)
        graph.update_relations_batch([
            (triple["subject"], triple["predicate"], triple["object"])
            for triple in triples
        ])
        self.assertIsNone(graph.query_relation("Ada_Lovelace", "born_in"))

    def test_when_rejects_location_only_fact(self):
        self.engine.kg.update_relation("Zorbel_Nax", "born_in", "Veloria")
        entities = self.engine.link_entities("When was Zorbel Nax born?")
        facts = self.engine.kg.get_all_facts_for_entities(entities)

        self.assertEqual(self.engine.select_answering_facts("When was Zorbel Nax born?", facts), [])

    def test_where_accepts_location_fact(self):
        self.engine.kg.update_relation("Zorbel_Nax", "born_in", "Veloria")
        entities = self.engine.link_entities("Where was Zorbel Nax born?")
        facts = self.engine.kg.get_all_facts_for_entities(entities)

        self.assertTrue(self.engine.select_answering_facts("Where was Zorbel Nax born?", facts))

    def test_who_action_question_accepts_matching_fact(self):
        query = "Who cracked Enigma?"
        entities = self.engine.link_entities(query)
        facts = self.engine.kg.get_all_facts_for_entities(entities)

        matched_facts = self.engine.select_answering_facts(query, facts)
        self.assertTrue(any(
            (subject, predicate, obj) == ("Alan_Turing", "cracked", "Enigma")
            for subject, predicate, obj, _, _ in matched_facts
        ))

    def test_strict_prompt_includes_question(self):
        self.engine.verbosity_mode = "STRICT"
        _, prompt = self.engine._get_mode_prompts("When was Ada born?", "Ada born_in London", [], [])

        self.assertEqual(prompt, "Fact: Ada born_in London\nQuestion: When was Ada born?")


if __name__ == "__main__":
    unittest.main()
