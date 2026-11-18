import contextlib
import gc
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from engine import IntegratedHillock
from main import handle_cli_input


class Step3TwoGearTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "step3_test.db")
        self.engine = IntegratedHillock(self.db_path)

    def tearDown(self):
        self.engine = None
        gc.collect()
        self.temp_dir.cleanup()

    def test_default_mode_is_free_and_invalid_modes_rejected(self):
        self.assertEqual(self.engine.verbosity_mode, "FREE")
        self.engine.verbosity_mode = "strict"
        self.assertEqual(self.engine.verbosity_mode, "STRICT")
        with self.assertRaises(ValueError):
            self.engine.verbosity_mode = "BALANCED"
        with self.assertRaises(ValueError):
            self.engine.verbosity_mode = "CONVERSATIONAL"

    def test_strict_mode_preserves_gate_and_refusal_contract(self):
        self.engine.verbosity_mode = "STRICT"

        # Non-question statement blocked in STRICT
        reply_stmt, _, _, mode_stmt = self.engine.execute_chat_turn(
            "Help me brainstorm a workout plan."
        )
        self.assertEqual(mode_stmt, "DETERMINISTIC_GATED_FALLBACK")
        self.assertIn("do not have verified information", reply_stmt)
        self.assertFalse(self.engine.last_receipt["vault_used"])
        self.assertFalse(self.engine.last_receipt["general_ai_used"])

        # Unverified question blocked in STRICT
        reply_q, _, _, mode_q = self.engine.execute_chat_turn(
            "What is Marie Curie's favorite color?"
        )
        self.assertEqual(mode_q, "DETERMINISTIC_GATED_FALLBACK")

        # Verified question passes STRICT and issues deterministic receipt
        with patch.object(
            self.engine,
            "query_ollama_stream",
            return_value="Marie Curie was born in Poland.",
        ):
            reply_ok, _, _, mode_ok = self.engine.execute_chat_turn(
                "Where was Marie Curie born?"
            )
        self.assertEqual(mode_ok, "RENDER_SUCCESS")
        self.assertIn("Poland", reply_ok)
        self.assertTrue(self.engine.last_receipt["vault_used"])
        self.assertFalse(self.engine.last_receipt["general_ai_used"])

    def test_free_mode_answers_general_statements_and_passes_history(self):
        self.assertEqual(self.engine.verbosity_mode, "FREE")
        captured = {}

        def fake_ollama(prompt, system_prompt, history=None):
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            captured["history"] = history
            return "Use a Python list comprehension for clean filtering."

        history = [
            {"role": "user", "content": "I am writing a Python script."},
            {"role": "assistant", "content": "Happy to help with Python."},
        ]

        with patch.object(self.engine, "query_ollama_stream", side_effect=fake_ollama):
            reply, _, _, mode = self.engine.execute_chat_turn(
                "Help me filter even numbers cleanly",
                history=history,
            )

        self.assertEqual(mode, "FREE_GENERAL")
        self.assertIn("list comprehension", reply)
        self.assertEqual(captured["history"], history)
        self.assertFalse(self.engine.last_receipt["vault_used"])
        self.assertTrue(self.engine.last_receipt["general_ai_used"])

    def test_free_mode_soft_sweep_uses_hebbian_neighbors_and_snippets(self):
        # Link a snippet where TALON extracted no triple (pure prose search fallback)
        self.engine.kg.save_snippet(
            "elsys_notes.pdf",
            "A resistor limits electrical current according to Ohm's law V = I * R.",
            "page 4",
        )
        # Strengthen Hebbian connection Marie_Curie <-> Radioactivity <-> Nobel_Prize
        self.engine.plasticity.update_associations({"Marie_Curie", "Radioactivity", "Nobel_Prize"})

        # 1. Broad statement about Marie Curie (would be blocked by STRICT)
        captured_curie = {}

        def fake_curie(prompt, system_prompt):
            captured_curie["prompt"] = prompt
            return "Marie Curie discovered Radioactivity and won the Nobel Prize."

        with patch.object(self.engine, "query_ollama_stream", side_effect=fake_curie):
            reply, primed, _, mode = self.engine.execute_chat_turn(
                "Tell me everything interesting about Marie Curie"
            )

        self.assertEqual(mode, "FREE_GROUNDED")
        self.assertIn("[VERIFIED FACTS]:", captured_curie["prompt"])
        self.assertIn("Radioactivity", captured_curie["prompt"])
        self.assertTrue(self.engine.last_receipt["vault_used"])
        self.assertGreaterEqual(len(self.engine.last_receipt["facts_used"]), 2)

        # 2. Broad prose snippet lookup without any entity ID match
        captured_ohm = {}

        def fake_ohm(prompt, system_prompt):
            captured_ohm["prompt"] = prompt
            return "Per elsys_notes.pdf, a resistor limits current via V = I * R."

        with patch.object(self.engine, "query_ollama_stream", side_effect=fake_ohm):
            _, _, _, mode_ohm = self.engine.execute_chat_turn(
                "How does a resistor limit electrical current?"
            )

        self.assertEqual(mode_ohm, "FREE_GROUNDED")
        self.assertIn("elsys_notes.pdf", captured_ohm["prompt"])
        self.assertEqual(len(self.engine.last_receipt["snippets_used"]), 1)
        self.assertEqual(self.engine.last_receipt["snippets_used"][0]["source_doc"], "elsys_notes.pdf")

    def test_cue_phrase_and_cli_remember_save_notes_without_saving_questions(self):
        # Question containing 'remember' must NOT trigger save
        with patch.object(self.engine, "query_ollama_stream", return_value="Checking memory..."):
            self.engine.execute_chat_turn("Can you remember where Marie Curie was born?")
        self.assertIsNone(self.engine.last_receipt["saved_memory"])
        self.assertEqual(self.engine.kg.get_snippet_count(), 0)

        # Explicit cue phrase MUST save to source_snippets and appear in receipt
        with patch.object(self.engine, "query_ollama_stream", return_value="Got it, saved!"):
            self.engine.execute_chat_turn("Remember that my Elsys lab is on Tuesday at 10:00.")
        self.assertIsNotNone(self.engine.last_receipt["saved_memory"])
        self.assertEqual(self.engine.kg.get_snippet_count(), 1)
        hits = self.engine.kg.search_snippets(["elsys", "tuesday"])
        self.assertEqual(len(hits), 1)
        self.assertIn("Tuesday at 10:00", hits[0][3])

        # CLI /remember + /mode + receipt display
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            handle_cli_input("/mode strict", self.engine)
            self.assertEqual(self.engine.verbosity_mode, "STRICT")
            handle_cli_input("/mode free", self.engine)
            self.assertEqual(self.engine.verbosity_mode, "FREE")
            handle_cli_input("/remember Capacitors block DC steady-state current.", self.engine)
            with patch.object(self.engine, "query_ollama_stream", return_value="Capacitors block DC."):
                handle_cli_input("What do capacitors block?", self.engine)

        cli_text = out.getvalue()
        self.assertIn("Hillock [MEMORY SAVED]:", cli_text)
        self.assertIn("[Memory Receipt | Mode: FREE", cli_text)
        self.assertIn("General AI: Yes", cli_text)

    def test_free_mode_offline_fallback_when_ollama_unreachable(self):
        self.engine.kg.save_snippet("notes.txt", "Python uses indentation for blocks.", "lines 1-1")
        with patch.object(self.engine, "query_ollama_stream", return_value=""):
            reply, _, _, mode = self.engine.execute_chat_turn("What does Python use for blocks?")
        self.assertEqual(mode, "FREE_OFFLINE_VAULT")
        self.assertIn("indentation", reply)
        self.assertFalse(self.engine.last_receipt["general_ai_used"])

    def test_planning_prompt_pulls_recent_chat_memory_notes(self):
        with patch.object(self.engine, "query_ollama_stream", return_value="Saved."):
            self.engine.execute_chat_turn("Remember that my Elsys lab is on Tuesday at 10:00.")

        captured = {}

        def fake_plan(prompt, system_prompt):
            captured["prompt"] = prompt
            return "Tuesday 10:00: Elsys lab. Add your other tasks in the open slots."

        with patch.object(self.engine, "query_ollama_stream", side_effect=fake_plan):
            _, _, _, mode = self.engine.execute_chat_turn("help me plan my week")

        self.assertEqual(mode, "FREE_GROUNDED")
        self.assertIn("Tuesday at 10:00", captured["prompt"])
        self.assertTrue(self.engine.last_receipt["vault_used"])
        self.assertEqual(self.engine.last_receipt["snippets_used"][0]["source_doc"], "chat_memory")


if __name__ == "__main__":
    unittest.main()