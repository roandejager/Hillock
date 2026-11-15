import concurrent.futures
import contextlib
import gc
import io
import os
import sqlite3
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient

import api
from config import HILLOCK_VERSION
from database import SQLiteKnowledgeGraph
from engine import IntegratedHillock
from main import handle_cli_input


class Step1HygieneTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "step1_test.db")
        self.engine = IntegratedHillock(self.db_path)

    def tearDown(self):
        self.engine = None
        gc.collect()
        self.temp_dir.cleanup()

    def test_sqlite_handles_close_without_resource_warning(self):
        local_db = os.path.join(self.temp_dir.name, "resource_check.db")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            kg = SQLiteKnowledgeGraph(local_db)
            kg.seed_initial_knowledge()
            kg.update_relation("Ada_Lovelace", "born_in", "London")
            kg.add_alias("ada", "Ada_Lovelace")
            _ = kg.get_all_facts_for_entities({"Ada_Lovelace"})
            _ = kg.query_relation("Ada_Lovelace", "born_in")
            kg = None
            gc.collect()
        os.remove(local_db)

    def test_merge_nodes_rolls_back_atomically_on_fk_failure(self):
        self.engine.kg.update_relation("Temp_Source", "discovered", "Temp_Target")
        with self.assertRaises(sqlite3.IntegrityError):
            # 'Missing_Canonical' is not in entities -> triggers FK violation at alias insert
            self.engine.kg.merge_nodes("Temp_Source", "Missing_Canonical")

        facts = self.engine.kg.get_all_facts_for_entities({"Temp_Source"})
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0][:3], ("Temp_Source", "discovered", "Temp_Target"))

    def test_execute_chat_turn_never_prints_to_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # 1. Greeting
            reply_greet, _, _, mode_greet = self.engine.execute_chat_turn("hi")
            # 2. Strict refusal
            self.engine.verbosity_mode = "STRICT"
            reply_refuse, _, _, mode_refuse = self.engine.execute_chat_turn(
                "Where was Unknown_Person born?"
            )
            # 3. Ollama render success
            with patch.object(
                self.engine,
                "query_ollama_stream",
                return_value="Alan Turing cracked the Enigma machine."
            ):
                reply_ok, _, _, mode_ok = self.engine.execute_chat_turn(
                    "Who cracked Enigma?"
                )
            # 4. Ollama render fallback
            with patch.object(self.engine, "query_ollama_stream", return_value=""):
                reply_fb, _, _, mode_fb = self.engine.execute_chat_turn(
                    "Who cracked Enigma?"
                )

        self.assertEqual(out.getvalue(), "")
        self.assertEqual(mode_greet, "GREETING")
        self.assertEqual(mode_refuse, "DETERMINISTIC_GATED_FALLBACK")
        self.assertEqual(mode_ok, "RENDER_SUCCESS")
        self.assertEqual(mode_fb, "RENDER_FALLBACK")
        self.assertIn("Hello", reply_greet)
        self.assertIn("do not have verified information", reply_refuse)
        self.assertIn("cracked the Enigma machine", reply_ok)
        self.assertIn("Handshake resolved", reply_fb)

    def test_cli_inspect_handles_four_tuple_and_missing_arg(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            keep_running_1 = handle_cli_input("/inspect Alan_Turing", self.engine)
            keep_running_2 = handle_cli_input("/inspect", self.engine)

        text = out.getvalue()
        self.assertTrue(keep_running_1)
        self.assertTrue(keep_running_2)
        self.assertIn("Alan_Turing", text)
        self.assertIn("cracked", text)
        self.assertIn("(Source: manual_entry)", text)
        self.assertIn("Format is: /inspect [entity]", text)

    def test_cli_prints_rendered_reply_and_handles_exit(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with patch.object(
                self.engine,
                "query_ollama_stream",
                return_value="Alan Turing cracked Enigma."
            ):
                keep_running = handle_cli_input("Who cracked Enigma?", self.engine)
            should_stop = handle_cli_input("exit", self.engine)

        text = out.getvalue()
        self.assertTrue(keep_running)
        self.assertFalse(should_stop)
        self.assertIn("Hillock (Renderer) > Alan Turing cracked Enigma.", text)

    def test_api_validation_version_streaming_and_concurrency(self):
        original_engine = api.engine
        api.engine = self.engine
        api.engine.verbosity_mode = "STRICT"
        client = TestClient(api.app)
        try:
            # Version sync
            res_health = client.get("/")
            self.assertEqual(res_health.status_code, 200)
            self.assertEqual(res_health.json()["version"], HILLOCK_VERSION)
            self.assertEqual(api.app.version, HILLOCK_VERSION)

            # Empty messages -> 422
            res_empty = client.post(
                "/v1/chat/completions",
                json={"model": "hillock", "messages": []}
            )
            self.assertEqual(res_empty.status_code, 422)

            # Invalid role -> 422
            res_role = client.post(
                "/v1/chat/completions",
                json={"model": "hillock", "messages": [{"role": "admin", "content": "hi"}]}
            )
            self.assertEqual(res_role.status_code, 422)

            # Out-of-range temperature -> 422
            res_temp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "hillock",
                    "messages": [{"role": "user", "content": "Where was Marie Curie born?"}],
                    "temperature": 5.0
                }
            )
            self.assertEqual(res_temp.status_code, 422)

            # System-only or whitespace user message -> 400
            res_sys = client.post(
                "/v1/chat/completions",
                json={"model": "hillock", "messages": [{"role": "system", "content": "be helpful"}]}
            )
            self.assertEqual(res_sys.status_code, 400)

            res_blank = client.post(
                "/v1/chat/completions",
                json={"model": "hillock", "messages": [{"role": "user", "content": "   "}]}
            )
            self.assertEqual(res_blank.status_code, 400)

            # Valid non-streaming completion
            with patch.object(
                api.engine,
                "query_ollama_stream",
                return_value="Marie Curie was born in Poland."
            ):
                res_ok = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hillock",
                        "messages": [{"role": "user", "content": "Where was Marie Curie born?"}]
                    }
                )
            self.assertEqual(res_ok.status_code, 200)
            content = res_ok.json()["choices"][0]["message"]["content"]
            self.assertEqual(content, "Marie Curie was born in Poland.")

            # Valid streaming completion (strict refusal path, no Ollama needed)
            res_stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "hillock",
                    "messages": [{"role": "user", "content": "Where was Unknown_Person born?"}],
                    "stream": True
                }
            )
            self.assertEqual(res_stream.status_code, 200)
            self.assertIn("data: [DONE]", res_stream.text)
            self.assertIn("do not have verified information", res_stream.text)

            # Concurrent burst safety
            def fire_query(idx: int):
                return client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hillock",
                        "messages": [{"role": "user", "content": f"Where was Unknown_{idx} born?"}]
                    }
                ).status_code

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                statuses = list(pool.map(fire_query, range(8)))
            self.assertEqual(statuses, [200] * 8)
        finally:
            api.engine = original_engine


if __name__ == "__main__":
    unittest.main()