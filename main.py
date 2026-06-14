"""The main execution orchestrator for the conversational chat console."""

import os
import re
import numpy as np
import logging
import platform
import multiprocessing
import subprocess
from typing import List, Tuple, Set, Optional, Dict

# Import modular components
from config import DB_FILE, OLLAMA_MODEL, HDC_THRESHOLD, MAX_WORKERS
from database import SQLiteKnowledgeGraph
from plasticity import HebbianPlasticityEngine
from reservoir import HyperdimensionalReservoir
from ingestor import ingest_document_parallel

logger = logging.getLogger("Hillock.Main")


def get_gpu_name() -> str:
    """Uses nvidia-smi utility to dynamically query local GPU model."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return out.strip()
    except Exception:
        return "Non-NVIDIA GPU or nvidia-smi unavailable"


def print_system_dashboard(hillock: "IntegratedHillock") -> None:
    """Displays system specifications and database persistence status."""
    gpu = get_gpu_name()
    cores = multiprocessing.cpu_count()
    os_name = f"{platform.system()} {platform.release()}"
    python_ver = platform.python_version()

    entities = hillock.kg.get_entity_count()
    relations = hillock.kg.get_relations_count()
    synapses = hillock.kg.get_synapse_count()

    print("\n" + "="*60)
    print("               HILLOCK SYSTEM SPECIFICATIONS              ")
    print("="*60)
    print(" [HARDWARE PROFILE]")
    print(f"  * OS Environment : {os_name}")
    print(f"  * CPU Cores      : {cores} Logical Threads")
    print(f"  * GPU Unit       : {gpu}")
    print(f"  * Parallel Workers: {MAX_WORKERS} Threads (GTX 1070 Optimized)")
    print(f"  * Python Host    : {python_ver}")
    print("-"*60)
    print(" [PERSISTENT MEMORY GRAPH STATUS]")
    print(f"  * Database File  : {DB_FILE} ({'Active' if os.path.exists(DB_FILE) else 'Initializing'})")
    print(f"  * Unique Entities: {entities} registered nodes")
    print(f"  * Fact Triples   : {relations} stored relations")
    print(f"  * Synapses       : {synapses} active Hebbian connections")
    print("-"*60)
    print(" [BUILT-IN COMMAND REFERENCE] (Pillar 1)")
    print("  * /ingest [file] [fast/thorough] : Index TXT/PDF files locally")
    print("  * /mode [strict/balanced/convers] : Switch active AI personalities")
    print("  * /reset                         : Clear and re-seed database")
    print("  * exit / quit                    : Safely terminate session")
    print("="*60 + "\n")


class IntegratedHillock:
    def __init__(self, db_path: str = DB_FILE, ollama_model: str = OLLAMA_MODEL):
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)
        self.hdc = HyperdimensionalReservoir()
        self.ollama_model = ollama_model
        self.verbosity_mode = "BALANCED"  # Options: STRICT, BALANCED, CONVERSATIONAL

        # Predicate Normalization Map
        self.predicate_map = {
            "was_born_in": "born_in", "was born in": "born_in", "was_born": "born_in", "was born": "born_in",
            "bear": "born_in", "born": "born_in", "came_from": "born_in",
            "work": "collaborated_with", "work_with": "collaborated_with", "worked_with": "collaborated_with",
            "worked with": "collaborated_with", "partnered_with": "collaborated_with", "partnered with": "collaborated_with",
            "co_invented": "discovered", "discovered": "discovered", "found": "discovered", "uncovered": "discovered",
            "crack": "cracked", "cracked": "cracked", "broke": "cracked"
        }

        # Seed HDC codebook with initial graph entities
        for ent_id in self.kg.get_all_entity_ids():
            self.hdc.get_or_allocate_hypervector(ent_id)

    def is_question(self, text: str) -> bool:
        cleaned = text.strip().lower()
        if cleaned.endswith("?"):
            return True
        question_words = {"who", "what", "where", "when", "why", "how", "which", "whom"}
        tokens = re.sub(r"[^\w\s]", "", cleaned).split()
        if tokens and tokens[0] in question_words:
            return True
        return False

    def resolve_entity_identity(self, name_str: str) -> str:
        normalized_new = name_str.strip().replace(" ", "_").lower()

        # 1. Do not resolve common short prepositions or pronouns to complex entities
        if len(normalized_new) <= 2:
            return name_str.strip().replace(" ", "_")

        all_ids = self.kg.get_all_entity_ids()

        # 2. Perfect Match
        for ent_id in all_ids:
            if ent_id.lower() == normalized_new:
                return ent_id

        # 3. Safe Component Match (checks split parts rather than raw substring)
        for ent_id in all_ids:
            lower_parts = ent_id.lower().split("_")
            if normalized_new in lower_parts:
                return ent_id

        return name_str.strip().replace(" ", "_")

    def link_entities(self, query: str) -> Set[str]:
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
            import urllib.request
            import json
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

    def parse_json_safely(self, raw_text: str) -> Optional[any]:
        try:
            cleaned = re.sub(r"```json|```", "", raw_text).strip()

            # 1. Search for JSON array boundaries first
            start_list = cleaned.find("[")
            end_list = cleaned.rfind("]")

            if start_list != -1 and end_list != -1:
                return json.loads(cleaned[start_list:end_list + 1])

            # 2. Fall back to object boundaries if no array exists
            start_obj = cleaned.find("{")
            end_obj = cleaned.rfind("}")
            if start_obj != -1 and end_obj != -1:
                return json.loads(cleaned[start_obj:end_obj + 1])
        except Exception:
            pass
        return None

    def extract_field_via_regex(self, text: str, field_name: str) -> Optional[str]:
        pattern = r'"' + re.escape(field_name) + r'"\s*:\s*["\']?([^"\'}\n]+)["\']?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def select_answering_facts(self, query: str, facts: List[Tuple[str, str, str]], threshold: float = HDC_THRESHOLD) -> \
    List[Tuple[str, str, str, float]]:
        if not facts:
            return []

        query_tokens = set(re.sub(r"[^\w\s]", "", query).lower().split())
        query_hv = np.zeros(self.hdc.d, dtype=np.int32)

        for token in query_tokens:
            resolved = self.resolve_entity_identity(token)
            if resolved in self.hdc.codebook:
                query_hv += self.hdc.get_or_allocate_hypervector(resolved, is_vocab_token=False)
            else:
                query_hv += self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)

        scored_facts = []
        for s, p, o in facts:
            # Resolve subject and object identities
            s_resolved = self.resolve_entity_identity(s)
            o_resolved = self.resolve_entity_identity(o)

            # Extract keywords for query overlap
            pred_keywords = [p.lower().replace("_", " ")]
            if p in ["collaborated_with", "work"]:
                pred_keywords.extend(["work", "worked", "with", "partner", "collaborated"])
            elif p in ["born_in", "bear"]:
                pred_keywords.extend(["born", "in", "birth"])
            elif p in ["discovered", "discover"]:
                pred_keywords.extend(["discover", "discovered", "found"])
            elif p in ["cracked", "crack"]:
                pred_keywords.extend(["crack", "cracked", "broke"])

            # Dynamically select the single best-matching predicate word
            best_pred_word = p.lower()
            for kw in pred_keywords:
                if kw in query_tokens:
                    best_pred_word = kw
                    break

            # Strictly 3 vector components ensures perfectly balanced norms across all facts
            components = [s_resolved, o_resolved, best_pred_word]

            fact_hv = np.zeros(self.hdc.d, dtype=np.int32)
            for comp in set(components):
                resolved_comp = self.resolve_entity_identity(comp)
                if resolved_comp in self.hdc.codebook:
                    fact_hv += self.hdc.get_or_allocate_hypervector(resolved_comp, is_vocab_token=False)
                else:
                    fact_hv += self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=True)

            q_norm = np.linalg.norm(query_hv)
            f_norm = np.linalg.norm(fact_hv)
            similarity = np.dot(query_hv, fact_hv) / (q_norm * f_norm) if (q_norm > 0 and f_norm > 0) else 0.0

            logger.info(f"HDC Semantic Matcher: Candidate Fact [{s} {p} {o}] Cosine Similarity: {similarity:.4f}")
            if similarity >= threshold:
                scored_facts.append((s, p, o, similarity))

        scored_facts.sort(key=lambda x: x[3], reverse=True)
        return scored_facts

    def extract_factual_declaration_two_pass(self, sentence: str) -> Optional[Dict[str, str]]:
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

        resolved_a = self.resolve_entity_identity(ent_a)
        resolved_b = self.resolve_entity_identity(ent_b)

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

        clean_pred = pred.strip().lower().replace(" ", "_")
        normalized_b = resolved_b.lower().replace("_", "")

        if normalized_b in clean_pred.replace("_", ""):
            clean_pred = clean_pred.replace(normalized_b, "").strip("_")

        if not clean_pred or clean_pred in ["", "_"]:
            clean_pred = "related_to"

        return {
            "subject": resolved_a,
            "predicate": clean_pred,
            "object": resolved_b
        }

    def _get_mode_prompts(self, query: str, facts_str: str, primed_info: list, hdc_fingerprint: list) -> Tuple[str, str]:
        priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"
        fingerprint_str = ", ".join([f"{node} (match {sim:.2f})" for node, sim in hdc_fingerprint]) if hdc_fingerprint else "None"

        if self.verbosity_mode == "STRICT":
            system_prompt = (
                "You are a professional fact renderer. Translate ONLY the provided fact into one sentence. "
                "Do not add any extra_old context, historical assumptions, or details."
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
        """Gating routing controller with pronoun coreference resolution."""
        is_query = self.is_question(query)

        greetings = {"hello", "hi", "hey", "greetings", "thanks", "thank you", "bye", "goodbye"}
        query_clean = re.sub(r"[^\w\s]", "", query).strip().lower()

        if query_clean in greetings or len(query_clean.split()) < 2:
            dummy_primed = []
            dummy_fingerprint = []
            if self.verbosity_mode == "CONVERSATIONAL":
                return "Hillock > Hello! I am your conversational hillock. Ask me any factual questions about my indexed knowledge.", dummy_primed, dummy_fingerprint, "GREETING"
            elif self.verbosity_mode == "BALANCED":
                return "Hillock > Hello. Ready for factual questions.", dummy_primed, dummy_fingerprint, "GREETING"
            else:
                return "Hillock > I do not have verified information about that.", dummy_primed, dummy_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        active_entities = self.link_entities(query)

        # HDC Pronoun Resolution [1, 28]
        if not active_entities:
            pronouns = {"he", "she", "his", "her", "him", "they", "them", "it"}
            query_words = set(re.sub(r"[^\w\s]", "", query).lower().split())
            if query_words.intersection(pronouns):
                fingerprint = self.hdc.get_context_fingerprint(top_k=1)
                if fingerprint:
                    closest_entity, similarity = fingerprint[0]
                    logger.info(f"HDC Coreference: Resolved pronoun to context concept '{closest_entity}' (Similarity: {similarity:.4f})")
                    active_entities.add(closest_entity)

        # Update HDC context state sequentially
        tokens = re.sub(r"[^\w\s]", "", query).lower().split()
        for token in tokens:
            resolved_id = self.resolve_entity_identity(token)
            if resolved_id in self.hdc.codebook:
                token_hv = self.hdc.get_or_allocate_hypervector(resolved_id, is_vocab_token=False)
            else:
                token_hv = self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)
            self.hdc.step(token_hv)

        hdc_fingerprint = self.hdc.get_context_fingerprint(top_k=3)
        resolved_value = None
        source_id, predicate = None, None

        if is_query:
            if active_entities:
                candidate_facts = self.kg.get_all_facts_for_entities(active_entities)
                matched_facts = self.select_answering_facts(query, candidate_facts)
                if matched_facts:
                    # Update associations for all matched targets
                    active_update_set = active_entities.copy()
                    for s, p, o, _ in matched_facts:
                        active_update_set.add(s)
                        active_update_set.add(o)
                    self.plasticity.update_associations(active_update_set)

                    if len(matched_facts) == 1:
                        s, p, o, _ = matched_facts[0]
                        facts_str = f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}]"
                        source_id = s
                    else:
                        facts_str = " | ".join([f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}]" for s, p, o, _ in matched_facts])
                        source_id = matched_facts[0][0]

                    primed_info = self.plasticity.get_associated_priming_context(source_id)
                    system_prompt, render_prompt = self._get_mode_prompts(query, facts_str, primed_info, hdc_fingerprint)

                    llm_response = self.query_ollama(render_prompt, system_prompt)
                    if llm_response:
                        return f"Hillock (Ollama-Renderer) > {llm_response}", primed_info, hdc_fingerprint, "RENDER_SUCCESS"
                    else:
                        return f"Hillock (Simulated) > Handshake resolved: {facts_str}.", primed_info, hdc_fingerprint, "RENDER_FALLBACK"

            return "Hillock > I do not have verified information about that.", [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        # Safe Two-Pass Extraction conversational learning pathway [1, 23]
        extracted_fact = self.extract_factual_declaration_two_pass(query)
        if extracted_fact:
            sub = extracted_fact["subject"]
            pred = extracted_fact["predicate"]
            obj = extracted_fact["object"]
            norm_pred = self.predicate_map.get(pred, pred)
            self.kg.update_relation(sub, norm_pred, obj)
            self.plasticity.update_associations({sub, obj})
            self.hdc.get_or_allocate_hypervector(sub)
            self.hdc.get_or_allocate_hypervector(obj)
            return f"Hillock (Autonomous Learner) > I have recorded a new factual declaration: [{sub.replace('_', ' ')}] -[{norm_pred}]-> [{obj.replace('_', ' ')}].", [], hdc_fingerprint, "EXTRACT_SUCCESS"

        return "Hillock > I do not have verified information about that.", [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"


# Terminal loop orchestrator
if __name__ == "__main__":
    hillock = IntegratedHillock(DB_FILE)

    # Render the System Status Dashboard on Startup [1]
    print_system_dashboard(hillock)

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
                print("Safely shutting down local hillock.")
                break

            if user_input.startswith("/mode"):
                 parts = user_input.split()
                 if len(parts) == 2:
                      mode_name = parts[1].strip().upper()
                      if mode_name in ["STRICT", "BALANCED", "CONVERSATIONAL"]:
                           hillock.verbosity_mode = mode_name
                           print(f"Hillock [SYSTEM]: Verbosity mode set to [{mode_name}] successfully.")
                      else:
                           print("Hillock [SYSTEM]: Error. Modes available: strict, balanced, conversational.")
                 else:
                      print("Hillock [SYSTEM]: Error. Format is: /mode [strict/balanced/conversational]")
                 continue

            if user_input.strip() == "/reset":
                 print("Hillock [SYSTEM]: Initiating deliberate database reset...")
                 hillock.kg.clear_and_reinitialize()
                 hillock.hdc.state = np.zeros(hillock.hdc.d, dtype=np.float64)
                 hillock.hdc.codebook.clear()
                 hillock.hdc.vocab_book.clear()
                 for ent_id in hillock.kg.get_all_entity_ids():
                     hillock.hdc.get_or_allocate_hypervector(ent_id)
                 print("Hillock [SYSTEM]: Database has been cleanly reset, re-seeded, and HDC state cleared.")
                 continue

            if user_input.startswith("/ingest"):
                parts = user_input.split()
                if len(parts) >= 2:
                    filename = parts[1].strip()
                    mode = "fast"
                    if len(parts) >= 3:
                        mode = parts[2].strip().lower()

                    fast_mode = (mode == "fast")
                    print(f"Hillock [SYSTEM]: Initiating bulk ingestion for '{filename}' (Mode: {mode.upper()})...")
                    result = ingest_document_parallel(filename, hillock)
                    print(f"Hillock [SYSTEM]: {result}")
                else:
                    print("Hillock [SYSTEM]: Error. Correct format is: /ingest [filename.ext] [fast/thorough]")
                continue

            reply, primed, fingerprint, mode = hillock.execute_chat_turn(user_input)
            print(reply)

            if primed:
                print("  [Memory Priming Node Activations]:")
                for node, weight in primed[:3]:
                    print(f"    * Associated Concept: '{node:<13}'  Synaptic Connection Strength: {weight:.4f}")

            if fingerprint and mode == "RENDER_SUCCESS":
                print("  [HDC Conversational Fingerprint Traces]:")
                for node, sim in fingerprint[:3]:
                    print(f"    * Active Semantic Echo: '{node:<13}'  Vector Cosine Similarity: {sim:.4f}")
            print()

        except KeyboardInterrupt:
            print("\nSafely shutting down local hillock.")
            break