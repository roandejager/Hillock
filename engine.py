"""Core Hillock Memory Engine."""

import os
import re
import sys
import json
import numpy as np
import logging
import urllib.request
from typing import List, Tuple, Set, Optional

from config import DB_FILE, OLLAMA_MODEL, LLM_BASE_URL, HDC_THRESHOLD
from database import SQLiteKnowledgeGraph
from plasticity import HebbianPlasticityEngine
from reservoir import HyperdimensionalReservoir, load_lightweight_glove


logger = logging.getLogger("Hillock.Engine")

class IntegratedHillock:
    def __init__(self, db_path: str = DB_FILE, ollama_model: str = OLLAMA_MODEL):
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)

        # Load lightweight 10MB GloVe dictionary for continuous SimHash VSA
        self.glove_dict = load_lightweight_glove()
        self.hdc = HyperdimensionalReservoir(glove_dict=self.glove_dict)

        self.ollama_model = ollama_model
        self.verbosity_mode = "BALANCED"  # Options: STRICT, BALANCED, CONVERSATIONAL
        self.debug_level = "OFF"         # Options: OFF, LOW, FULL

        # Predicate Normalization Map
        self.predicate_map = {
            "was_born_in": "born_in", "was born in": "born_in", "was_born": "born_in", "was born": "born_in",
            "bear": "born_in", "born": "born_in", "came_from": "born_in",
            "work": "collaborated_with", "work_with": "collaborated_with", "worked_with": "collaborated_with",
            "worked with": "collaborated_with", "partnered_with": "collaborated_with", "partnered with": "collaborated_with",
            "co_invented": "discovered", "discovered": "discovered", "found": "discovered", "uncovered": "discovered",
            "crack": "cracked", "cracked": "cracked", "broke": "cracked"
        }

        # Seed HDC codebook with initial graph entities using SimHash
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
        # Strip trailing possessive artifacts like _'s or _s
        clean_name = re.sub(r"[_']s$", "", name_str.strip(), flags=re.IGNORECASE)
        normalized_new = clean_name.replace(" ", "_").lower()

        if len(normalized_new) <= 2:
            return clean_name.replace(" ", "_")

        all_ids = self.kg.get_all_entity_ids()

        for ent_id in all_ids:
            if ent_id.lower() == normalized_new:
                return ent_id

        for ent_id in all_ids:
            lower_parts = ent_id.lower().split("_")
            if normalized_new in lower_parts:
                return ent_id

        return clean_name.replace(" ", "_")

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

    def list_local_ollama_models(self) -> List[str]:
        """Queries local Ollama tags API to discover available models on user's PC."""
        url = "http://localhost:11434/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in res_data.get("models", []) if m.get("name")]
                return models
        except Exception:
            return []

    def query_ollama_stream(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Token-streaming generator using the universal OpenAI-compatible format."""
        url = LLM_BASE_URL
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": 0.0
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            full_response = []
            sys.stdout.write("Hillock (Renderer) > ")
            sys.stdout.flush()

            with urllib.request.urlopen(req, timeout=180) as response:
                for line in response:
                    if line:
                        # The OpenAI stream format prefixes lines with "data: "
                        decoded_line = line.decode("utf-8").strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                # Extract token from the OpenAI delta format
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    token = delta.get("content", "")
                                    if token:
                                        sys.stdout.write(token)
                                        sys.stdout.flush()
                                        full_response.append(token)
                            except json.JSONDecodeError:
                                continue
            print()  # Newline after stream finishes
            return "".join(full_response).strip()
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            return None

    def select_answering_facts(self, query: str, facts: List[Tuple[str, str, str, str]], threshold: float = HDC_THRESHOLD) -> List[Tuple[str, str, str, str, float]]:
        if not facts:
            return []

        query_tokens = set(re.sub(r"[^\w\s]", "", query).lower().split())

        query_components = set()
        for token in query_tokens:
            norm_token = self.predicate_map.get(token, token)
            resolved = self.resolve_entity_identity(norm_token)
            if len(token) > 2 or resolved in self.hdc.codebook or norm_token in self.predicate_map.values():
                query_components.add(resolved)
                if norm_token != token:
                    query_components.add(norm_token)

        query_hvs = []
        for comp in query_components:
            if comp in self.hdc.codebook:
                query_hvs.append(self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=False))
            else:
                query_hvs.append(self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=True))

        if not query_hvs:
            return []

        scored_facts = []
        for s, p, o, doc in facts:
            s_resolved = self.resolve_entity_identity(s)
            o_resolved = self.resolve_entity_identity(o)

            p_hv = self.hdc.resolve_predicate_hypervector(p)
            fact_hvs = [p_hv]

            components = [s_resolved, o_resolved]
            for comp in set(components):
                resolved_comp = self.resolve_entity_identity(comp)
                if resolved_comp in self.hdc.codebook:
                    fact_hvs.append(self.hdc.get_or_allocate_hypervector(resolved_comp, is_vocab_token=False))
                else:
                    fact_hvs.append(self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=True))

            similarity = self.hdc.hydra_late_interaction_maxsim(query_hvs, fact_hvs, tau_early=0.20)

            pred_max_align = max(
                (float(np.dot(q_hv.astype(np.float32), p_hv.astype(np.float32)) / self.hdc.D) for q_hv in query_hvs),
                default=0.0
            )

            if self.debug_level in ["LOW", "FULL"]:
                print(f"  [DEBUG HDC HYDRA]: Fact [{s} {p} {o}] MaxSim: {similarity:.4f} | PredAlign: {pred_max_align:.4f}")

            if similarity >= threshold and pred_max_align >= 0.35:
                scored_facts.append((s, p, o, doc, similarity))

        scored_facts.sort(key=lambda x: x[4], reverse=True)
        return scored_facts

    def execute_chat_turn(self, query: str) -> Tuple[str, List[Tuple[str, float]], List[Tuple[str, float]], str]:
        is_query = self.is_question(query)

        greetings = {"hello", "hi", "hey", "greetings", "thanks", "thank you", "bye", "goodbye"}
        query_clean = re.sub(r"[^\w\s]", "", query).strip().lower()

        if query_clean in greetings or len(query_clean.split()) < 2:
            dummy_primed = []
            dummy_fingerprint = []
            if self.verbosity_mode == "CONVERSATIONAL":
                msg = "Hillock > Hello! I am your conversational hillock. Ask me any factual questions about my indexed knowledge."
            elif self.verbosity_mode == "BALANCED":
                msg = "Hillock > Hello. Ready for factual questions."
            else:
                msg = "Hillock > I do not have verified information about that."
            print(msg)
            return msg, dummy_primed, dummy_fingerprint, "GREETING"

        active_entities = self.link_entities(query)

        if not active_entities:
            pronouns = {"he", "she", "his", "her", "him", "they", "them", "it"}
            query_words = set(re.sub(r"[^\w\s]", "", query).lower().split())
            if query_words.intersection(pronouns):
                fingerprint = self.hdc.get_context_fingerprint(top_k=1)
                if fingerprint:
                    closest_entity, similarity = fingerprint[0]
                    if self.debug_level in ["LOW", "FULL"]:
                        print(f"  [DEBUG HDC Coref]: Resolved pronoun to context concept '{closest_entity}' (Similarity: {similarity:.4f})")
                    active_entities.add(closest_entity)

        tokens = re.sub(r"[^\w\s]", "", query).lower().split()
        for token in tokens:
            resolved_id = self.resolve_entity_identity(token)
            if resolved_id in self.hdc.codebook:
                token_hv = self.hdc.get_or_allocate_hypervector(resolved_id, is_vocab_token=False)
            else:
                token_hv = self.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)
            self.hdc.step(token_hv)

        hdc_fingerprint = self.hdc.get_context_fingerprint(top_k=3)

        if is_query:
            if active_entities:
                candidate_facts = self.kg.get_all_facts_for_entities(active_entities)
                matched_facts = self.select_answering_facts(query, candidate_facts)
                if matched_facts:
                    active_update_set = active_entities.copy()
                    for s, p, o, doc, _ in matched_facts:
                        active_update_set.add(s)
                        active_update_set.add(o)
                    self.plasticity.update_associations(active_update_set)

                    if len(matched_facts) == 1:
                        s, p, o, doc, _ = matched_facts[0]
                        facts_str = f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}] (Source: {doc})"
                        source_id = s
                    else:
                        facts_str = " | ".join([f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}] (Source: {doc})" for s, p, o, doc, _ in matched_facts])
                        source_id = matched_facts[0][0]

                    primed_info = self.plasticity.get_associated_priming_context(source_id)
                    system_prompt, render_prompt = self._get_mode_prompts(query, facts_str, primed_info, hdc_fingerprint)

                    llm_response = self.query_ollama_stream(render_prompt, system_prompt)
                    if llm_response:
                        return f"Hillock (Renderer) > {llm_response}", primed_info, hdc_fingerprint, "RENDER_SUCCESS"
                    else:
                        fallback_msg = f"Hillock (Simulated) > Handshake resolved: {facts_str}."
                        print(fallback_msg)
                        return fallback_msg, primed_info, hdc_fingerprint, "RENDER_FALLBACK"

            refusal_msg = "Hillock > I do not have verified information about that."
            print(refusal_msg)
            return refusal_msg, [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        refusal_msg = "Hillock > I do not have verified information about that."
        print(refusal_msg)
        return refusal_msg, [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

    def _get_mode_prompts(self, query: str, facts_str: str, primed_info: list, hdc_fingerprint: list) -> Tuple[str, str]:
        priming_str = ", ".join([f"{node} (strength {w:.2f})" for node, w in primed_info[:2]]) if primed_info else "None"
        fingerprint_str = ", ".join([f"{node} (match {sim:.2f})" for node, sim in hdc_fingerprint]) if hdc_fingerprint else "None"

        if self.verbosity_mode == "STRICT":
            system_prompt = (
                "You are a professional fact renderer. Translate ONLY the provided fact into one sentence. "
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