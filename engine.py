"""Core Hillock Memory Engine."""

import json
import logging
import re
import urllib.request
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import numpy as np

from config import DB_FILE, HDC_THRESHOLD, LLM_BASE_URL, OLLAMA_MODEL
from database import SQLiteKnowledgeGraph
from plasticity import HebbianPlasticityEngine
from reservoir import HyperdimensionalReservoir, load_lightweight_glove

logger = logging.getLogger("Hillock.Engine")

VALID_MODES = {"FREE", "STRICT"}

MEMORY_CUE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"remember\s+that|remember\s*:|"
    r"note\s+that|note\s*:|"
    r"save\s+this\s*:|save\s+to\s+memory\s*:|"
    r"don't\s+forget\s+that|do\s+not\s+forget\s+that|"
    r"keep\s+in\s+mind\s+that"
    r")\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

FREE_STOP_WORDS = {
    "what", "who", "where", "when", "why", "how", "which", "whom",
    "was", "is", "were", "are", "did", "do", "does", "has", "have", "had",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for",
    "with", "by", "about", "tell", "me", "please", "explain", "help",
    "know", "can", "you", "your", "my", "we", "our", "saved", "this", "that",
}


class IntegratedHillock:
    def __init__(
        self,
        db_path: str = DB_FILE,
        ollama_model: str = OLLAMA_MODEL,
        verbosity_mode: str = "FREE",
    ):
        self.kg = SQLiteKnowledgeGraph(db_path)
        self.kg.seed_initial_knowledge()
        self.plasticity = HebbianPlasticityEngine(db_path)

        # Load lightweight 10MB GloVe dictionary for continuous SimHash VSA
        self.glove_dict = load_lightweight_glove()
        self.hdc = HyperdimensionalReservoir(glove_dict=self.glove_dict)

        self.ollama_model = ollama_model
        self._verbosity_mode = "FREE"
        self.verbosity_mode = verbosity_mode  # validated via property setter
        self.debug_level = "OFF"              # Options: OFF, LOW, FULL
        self.last_receipt: Dict[str, Any] = {}

        # Predicate Normalization Map
        self.predicate_map = {
            "was_born_in": "born_in", "was born in": "born_in", "was_born": "born_in", "was born": "born_in",
            "bear": "born_in", "born": "born_in", "came_from": "born_in",
            "work": "collaborated_with", "work_with": "collaborated_with", "worked_with": "collaborated_with",
            "worked with": "collaborated_with", "partnered_with": "collaborated_with", "partnered with": "collaborated_with",
            "co_invented": "discovered", "discovered": "discovered", "found": "discovered", "uncovered": "discovered",
            "crack": "cracked", "cracked": "cracked", "broke": "cracked",
        }

        # Seed HDC codebook with initial graph entities using SimHash
        for ent_id in self.kg.get_all_entity_ids():
            self.hdc.get_or_allocate_hypervector(ent_id)

    @property
    def verbosity_mode(self) -> str:
        return self._verbosity_mode

    @verbosity_mode.setter
    def verbosity_mode(self, mode: str) -> None:
        normalized = str(mode).strip().upper()
        if normalized not in VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}. Valid modes: FREE, STRICT.")
        self._verbosity_mode = normalized

    def get_ambiguous_facts(self) -> List[Tuple[str, str, str, str]]:
        """Scans the Knowledge Graph for unresolved pronouns or vague entities."""
        ambiguous_terms = {"he", "she", "it", "they", "this", "that", "who", "whom", "which", "his", "her"}
        ambiguous_facts = []

        with self.kg._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT source_id, predicate, target_id, source_doc FROM relations")
            for s, p, o, doc in cursor.fetchall():
                if s.lower() in ambiguous_terms or o.lower() in ambiguous_terms:
                    ambiguous_facts.append((s, p, o, doc))

        return ambiguous_facts

    def resolve_ambiguous_fact(self, old_s: str, p: str, old_o: str, new_s: str, new_o: str) -> None:
        """Deletes the ambiguous fact and replaces it with the user-clarified fact."""
        new_s_clean = self.resolve_entity_identity(new_s)
        new_o_clean = self.resolve_entity_identity(new_o)

        with self.kg._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM relations WHERE source_id=? AND predicate=? AND target_id=?",
                (old_s, p, old_o),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, 'Generic')",
                (new_s_clean, new_s_clean.replace("_", " ")),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, 'Generic')",
                (new_o_clean, new_o_clean.replace("_", " ")),
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO relations (source_id, predicate, target_id, source_doc, confidence)
                VALUES (?, ?, ?, 'human_disambiguation', 1.0)
                """,
                (new_s_clean, p, new_o_clean),
            )
            conn.commit()

        self.hdc.get_or_allocate_hypervector(new_s_clean)
        self.hdc.get_or_allocate_hypervector(new_o_clean)

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
        clean_name = re.sub(r"[_']s$", "", name_str.strip(), flags=re.IGNORECASE)
        normalized_new = clean_name.replace(" ", "_").lower()

        canonical = self.kg.get_canonical_id(normalized_new)
        if canonical:
            return canonical

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

    def merge_entities(self, target_str: str, canonical_str: str) -> Tuple[bool, str]:
        """Merges a fractured target node into a canonical root node across DB and HDC space."""
        target_id = self.resolve_entity_identity(target_str)
        canonical_id = self.resolve_entity_identity(canonical_str)

        if target_id == canonical_id:
            return False, "Entities already resolve to the same canonical ID."

        if canonical_id not in self.kg.get_all_entity_ids():
            with self.kg._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO entities (id, name, type) VALUES (?, ?, 'Generic')",
                    (canonical_id, canonical_id.replace("_", " ")),
                )

        self.kg.merge_nodes(target_id, canonical_id)

        if target_id in self.hdc.codebook:
            h_old = self.hdc.codebook[target_id]
            h_can = self.hdc.get_or_allocate_hypervector(canonical_id)

            h_new = np.sign(h_old.astype(np.int32) + h_can.astype(np.int32)).astype(np.int8)
            h_new[h_new == 0] = 1

            self.hdc.codebook[canonical_id] = h_new
            del self.hdc.codebook[target_id]

        return True, f"Successfully merged '{target_id}' into '{canonical_id}'."

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
        url = "http://127.0.0.1:11434/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in res_data.get("models", []) if m.get("name")]
                return models
        except Exception:
            return []

    def query_ollama_generator(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """True token-streaming generator for the FastAPI SSE endpoint."""
        url = LLM_BASE_URL
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-6:]:
                role = turn.get("role", "")
                content = (turn.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content[:1000]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.0 if self.verbosity_mode == "STRICT" else 0.3,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                for line in response:
                    if line:
                        decoded_line = line.decode("utf-8").strip()
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    token = delta.get("content", "")
                                    if token:
                                        yield token
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"LLM streaming generator error: {e}")
            yield ""

    def query_ollama_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Non-streaming collector consuming the Ollama generator."""
        if history:
            return "".join(
                self.query_ollama_generator(prompt, system_prompt, history=history)
            ).strip()
        return "".join(self.query_ollama_generator(prompt, system_prompt)).strip()

    def select_answering_facts(
        self,
        query: str,
        facts: List[Tuple[str, str, str, str]],
        threshold: float = HDC_THRESHOLD,
    ) -> List[Tuple[str, str, str, str, float]]:
        if not facts:
            return []

        stop_words = {
            "what", "how", "which", "whom",
            "was", "is", "were", "are", "did", "do", "does", "has", "had",
            "the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for", "with", "by",
        }

        raw_tokens = set(re.sub(r"[^\w\s]", "", query).lower().split())
        query_tokens = raw_tokens - stop_words

        if re.search(r"\bwhen\b|\bwhat\s+(?:year|date)\b", query.lower()):
            temporal_predicates = {"born_on", "date_of_birth", "birth_date", "occurred_on"}

            def is_temporal_fact(fact: Tuple[str, str, str, str]) -> bool:
                _, predicate, obj, _ = fact
                predicate = predicate.lower()
                return (
                    predicate in temporal_predicates
                    or any(term in predicate.split("_") for term in ("date", "year", "time", "timestamp"))
                    or re.search(r"\b\d{3,4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", obj) is not None
                )

            facts = [fact for fact in facts if is_temporal_fact(fact)]
            if not facts:
                return []

        if re.match(r"\s*who\s+(?:is|was|are|were)\b", query.lower()):
            identity_predicates = {
                "identity", "is_a", "instance_of", "occupation",
                "person_type", "profession", "known_as",
            }
            facts = [fact for fact in facts if fact[1].lower() in identity_predicates]
            if not facts:
                return []

        if not query_tokens:
            query_tokens = raw_tokens

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
                default=0.0,
            )

            if self.debug_level in ["LOW", "FULL"]:
                print(f"  [DEBUG HDC HYDRA]: Fact [{s} {p} {o}] MaxSim: {similarity:.4f} | PredAlign: {pred_max_align:.4f}")

            if similarity >= threshold and pred_max_align >= 0.35:
                scored_facts.append((s, p, o, doc, similarity))

        scored_facts.sort(key=lambda x: x[4], reverse=True)
        return scored_facts

    def _extract_memory_cue(self, query: str) -> Optional[str]:
        match = MEMORY_CUE_RE.match(query.strip())
        if not match:
            return None
        note = match.group(1).strip()
        return note if len(note) >= 3 else None

    def remember_note(self, text: str, source_doc: str = "chat_memory") -> Dict[str, Any]:
        """Stores a user note into source_snippets and links co-occurring known entities."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot save an empty note.")
        sid = self.kg.save_snippet(source_doc, clean_text, page_or_line="chat_note")
        linked = self.link_entities(clean_text)
        if len(linked) >= 2:
            self.plasticity.update_associations(linked)
        return {
            "snippet_id": sid,
            "source_doc": source_doc,
            "page_or_line": "chat_note",
            "text": clean_text,
            "linked_entities": sorted(linked),
        }

    def _score_fact_soft(self, query_hvs: List[np.ndarray], fact: Tuple[str, str, str, str]) -> float:
        if not query_hvs:
            return 0.0
        s, p, o, _ = fact
        p_hv = self.hdc.resolve_predicate_hypervector(p)
        fact_hvs = [p_hv]
        for comp in {self.resolve_entity_identity(s), self.resolve_entity_identity(o)}:
            if comp in self.hdc.codebook:
                fact_hvs.append(self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=False))
            else:
                fact_hvs.append(self.hdc.get_or_allocate_hypervector(comp, is_vocab_token=True))
        return self.hdc.hydra_late_interaction_maxsim(query_hvs, fact_hvs, tau_early=0.15)

    def _retrieve_free_context(
        self, query: str, direct_entities: Set[str]
    ) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]], List[Tuple[str, float]]]:
        """
        Soft associative sweep for FREE mode:
        1) direct entities + 3-turn HDC context entities (sim >= 0.22)
        2) 1-hop Hebbian neighbor expansion
        3) soft fact selection (without predicate veto)
        4) linked + keyword-matched source_snippets
        """
        fp_all = self.hdc.get_context_fingerprint(top_k=3)
        fp_entities = {ent for ent, sim in fp_all if sim >= 0.45}

        seed_entities = set(direct_entities)
        if not seed_entities:
            seed_entities.update(fp_entities)

        neighbor_entities: Set[str] = set()
        primed_info: List[Tuple[str, float]] = []
        seen_primed: Set[str] = set()
        for ent in sorted(seed_entities):
            for neighbor, weight in self.plasticity.get_associated_priming_context(ent, threshold=0.05)[:5]:
                neighbor_entities.add(neighbor)
                if neighbor not in seen_primed and neighbor not in direct_entities:
                    seen_primed.add(neighbor)
                    primed_info.append((neighbor, weight))
        primed_info.sort(key=lambda x: x[1], reverse=True)

        # Build query hypervectors for soft ranking
        raw_tokens = set(re.sub(r"[^\w\s]", "", query).lower().split())
        clean_terms = [t for t in raw_tokens if t not in FREE_STOP_WORDS and len(t) >= 3]
        query_hvs = [
            self.hdc.get_or_allocate_hypervector(
                self.resolve_entity_identity(t),
                is_vocab_token=(self.resolve_entity_identity(t) not in self.hdc.codebook),
            )
            for t in clean_terms
        ]

        selected_facts: List[Tuple[str, str, str, str]] = []
        seen_fact_keys: Set[Tuple[str, str, str]] = set()

        # Pull direct entity facts first (soft ranked, no predicate veto)
        if direct_entities:
            direct_candidates = self.kg.get_all_facts_for_entities(direct_entities)
            scored_direct = sorted(
                direct_candidates,
                key=lambda f: self._score_fact_soft(query_hvs, f),
                reverse=True,
            )
            for fact in scored_direct:
                key = fact[:3]
                if key not in seen_fact_keys:
                    seen_fact_keys.add(key)
                    selected_facts.append(fact)
                if len(selected_facts) >= 6:
                    break

        # Pull neighbor / context facts that clear soft similarity >= 0.22
        assoc_entities = (neighbor_entities | fp_entities) - direct_entities
        if assoc_entities:
            assoc_candidates = self.kg.get_all_facts_for_entities(assoc_entities)
            scored_assoc = [
                (self._score_fact_soft(query_hvs, f), f)
                for f in assoc_candidates
            ]
            scored_assoc.sort(key=lambda x: x[0], reverse=True)
            for sim, fact in scored_assoc:
                if sim < 0.42:
                    continue
                key = fact[:3]
                if key not in seen_fact_keys:
                    seen_fact_keys.add(key)
                    selected_facts.append(fact)
                if len(selected_facts) >= 10:
                    break

        # Gather source snippets (fact-linked + entity-linked + keyword search fallback)
        snippets_by_id: Dict[str, Tuple[str, str, str, str]] = {}
        if selected_facts:
            for row in self.kg.get_snippets_for_facts([f[:3] for f in selected_facts], limit=4):
                snippets_by_id[row[0]] = row
        sweep_entities = direct_entities | neighbor_entities | fp_entities
        if sweep_entities and len(snippets_by_id) < 5:
            for row in self.kg.get_snippets_for_entities(sweep_entities, limit=5):
                snippets_by_id.setdefault(row[0], row)
        if clean_terms and len(snippets_by_id) < 5:
            for row in self.kg.search_snippets(clean_terms, limit=4):
                snippets_by_id.setdefault(row[0], row)
        
        personal_cue_words = {
            "plan", "schedule", "week", "today", "tomorrow", "calendar",
            "day", "note", "notes", "reminder", "reminders", "task", "tasks",
            "class", "classes", "lab", "labs", "exam", "exams", "deadline",
        }
        if raw_tokens & personal_cue_words and len(snippets_by_id) < 5:
            for row in self.kg.get_recent_snippets("chat_memory", limit=5):
                snippets_by_id.setdefault(row[0], row)

        return selected_facts, list(snippets_by_id.values())[:5], primed_info[:5]

    @staticmethod
    def _build_receipt(
        mode: str,
        turn_outcome: str,
        facts: List[Tuple[str, str, str, str]],
        snippets: List[Tuple[str, str, str, str]],
        primed_info: List[Tuple[str, float]],
        saved_memory: Optional[Dict[str, Any]],
        general_ai_used: bool,
    ) -> Dict[str, Any]:
        return {
            "mode": mode,
            "turn_outcome": turn_outcome,
            "vault_used": bool(facts or snippets),
            "facts_used": [
                {"subject": s, "predicate": p, "object": o, "source_doc": doc}
                for s, p, o, doc in facts
            ],
            "snippets_used": [
                {"id": sid, "source_doc": doc, "page_or_line": loc, "raw_text": txt}
                for sid, doc, loc, txt in snippets
            ],
            "primed_topics": [
                {"entity": ent, "weight": round(float(w), 4)}
                for ent, w in primed_info[:5]
            ],
            "saved_memory": saved_memory,
            "general_ai_used": general_ai_used,
        }

    @staticmethod
    def format_receipt_line(receipt: Dict[str, Any]) -> str:
        if not receipt:
            return "[Memory Receipt | Empty]"
        mode = receipt.get("mode", "FREE")
        n_facts = len(receipt.get("facts_used", []))
        n_snips = len(receipt.get("snippets_used", []))
        sources = sorted(
            {
                item["source_doc"]
                for item in (receipt.get("facts_used", []) + receipt.get("snippets_used", []))
                if item.get("source_doc")
            }
        )
        src_str = f" ({', '.join(sources[:3])})" if sources else ""
        parts = [f"Mode: {mode}", f"Vault: {n_facts} facts, {n_snips} snippets{src_str}"]
        if receipt.get("saved_memory"):
            parts.append("Saved Note: Yes")
        parts.append(f"General AI: {'Yes' if receipt.get('general_ai_used') else 'No'}")
        return f"[Memory Receipt | {' | '.join(parts)}]"

    def execute_chat_turn(
        self,
        query: str,
        stream_generator: bool = False,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Any, List[Tuple[str, float]], List[Tuple[str, float]], str]:
        greetings = {"hello", "hi", "hey", "greetings", "thanks", "thank you", "bye", "goodbye"}
        query_clean = re.sub(r"[^\w\s]", "", query).strip().lower()

        if query_clean in greetings:
            if self.verbosity_mode == "FREE":
                msg = "Hillock > Hello! Ready to chat, remember notes, or search your local memory."
            else:
                msg = "Hillock > Hello. Ready for factual questions."
            self.last_receipt = self._build_receipt(
                self.verbosity_mode, "GREETING", [], [], [], None, False
            )
            return msg, [], [], "GREETING"

        # 1. Step HDC context state and link entities
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

        # ==================================================================
        # GEAR 1: STRICT MODE (Locked Verification & Aimee/Issue #35 Contract)
        # ==================================================================
        if self.verbosity_mode == "STRICT":
            is_query = self.is_question(query)
            if is_query and active_entities:
                candidate_facts = self.kg.get_all_facts_for_entities(active_entities)
                matched_facts = self.select_answering_facts(query, candidate_facts)
                if matched_facts:
                    active_update_set = active_entities.copy()
                    for s, p, o, doc, _ in matched_facts:
                        active_update_set.add(s)
                        active_update_set.add(o)
                    self.plasticity.update_associations(active_update_set)

                    used_fact_tuples = [(s, p, o, doc) for s, p, o, doc, _ in matched_facts]
                    if len(matched_facts) == 1:
                        s, p, o, doc, _ = matched_facts[0]
                        facts_str = f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}] (Source: {doc})"
                        source_id = s
                    else:
                        facts_str = " | ".join(
                            f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}] (Source: {doc})"
                            for s, p, o, doc, _ in matched_facts
                        )
                        source_id = matched_facts[0][0]

                    primed_info = self.plasticity.get_associated_priming_context(source_id)
                    system_prompt, render_prompt = self._get_mode_prompts(
                        query, facts_str, primed_info, hdc_fingerprint
                    )
                    self.last_receipt = self._build_receipt(
                        "STRICT", "RENDER_SUCCESS", used_fact_tuples, [], primed_info, None, False
                    )

                    if stream_generator:
                        return (
                            self.query_ollama_generator(render_prompt, system_prompt),
                            primed_info,
                            hdc_fingerprint,
                            "RENDER_SUCCESS",
                        )

                    llm_response = self.query_ollama_stream(render_prompt, system_prompt)
                    if llm_response:
                        return (
                            f"Hillock (Renderer) > {llm_response}",
                            primed_info,
                            hdc_fingerprint,
                            "RENDER_SUCCESS",
                        )
                    self.last_receipt["turn_outcome"] = "RENDER_FALLBACK"
                    fallback_msg = f"Hillock (Simulated) > Handshake resolved: {facts_str}."
                    return fallback_msg, primed_info, hdc_fingerprint, "RENDER_FALLBACK"

            self.last_receipt = self._build_receipt(
                "STRICT", "DETERMINISTIC_GATED_FALLBACK", [], [], [], None, False
            )
            refusal_msg = "Hillock > I do not have verified information about that."
            return refusal_msg, [], hdc_fingerprint, "DETERMINISTIC_GATED_FALLBACK"

        # ==================================================================
        # GEAR 2: FREE MODE (Personal Assistant + Soft Sweep + Snippets)
        # ==================================================================
        saved_memory: Optional[Dict[str, Any]] = None
        cue_note = self._extract_memory_cue(query)
        if cue_note:
            saved_memory = self.remember_note(cue_note)

        matched_facts, matched_snippets, primed_info = self._retrieve_free_context(
            query, active_entities
        )

        if matched_facts or active_entities:
            update_set = set(active_entities)
            for s, p, o, _ in matched_facts:
                update_set.add(s)
                update_set.add(o)
            if len(update_set) >= 2:
                self.plasticity.update_associations(update_set)

        facts_str = " | ".join(
            f"[{s.replace('_', ' ')} {p} {o.replace('_', ' ')}] (Source: {doc})"
            for s, p, o, doc in matched_facts
        )
        snippets_str = "\n".join(
            f"- ({doc}{', ' + loc if loc else ''}): {' '.join(txt.split())[:450]}"
            for _, doc, loc, txt in matched_snippets
        )

        system_prompt, render_prompt = self._build_free_prompts(
            query=query,
            facts_str=facts_str,
            snippets_str=snippets_str,
            primed_info=primed_info,
            saved_memory=saved_memory,
        )

        outcome = "FREE_GROUNDED" if (matched_facts or matched_snippets or saved_memory) else "FREE_GENERAL"
        self.last_receipt = self._build_receipt(
            "FREE",
            outcome,
            matched_facts,
            matched_snippets,
            primed_info,
            saved_memory,
            True,
        )

        if stream_generator:
            gen = (
                self.query_ollama_generator(render_prompt, system_prompt, history=history)
                if history
                else self.query_ollama_generator(render_prompt, system_prompt)
            )
            return gen, primed_info, hdc_fingerprint, outcome

        llm_response = (
            self.query_ollama_stream(render_prompt, system_prompt, history=history)
            if history
            else self.query_ollama_stream(render_prompt, system_prompt)
        )

        if llm_response:
            return f"Hillock (Renderer) > {llm_response}", primed_info, hdc_fingerprint, outcome

        # Offline fallback if Ollama is not running
        self.last_receipt["general_ai_used"] = False
        if saved_memory:
            self.last_receipt["turn_outcome"] = "FREE_OFFLINE_SAVED"
            return (
                f"Hillock (Saved) > Stored in chat_memory: {saved_memory['text']}",
                primed_info,
                hdc_fingerprint,
                "FREE_OFFLINE_SAVED",
            )
        if matched_facts or matched_snippets:
            self.last_receipt["turn_outcome"] = "FREE_OFFLINE_VAULT"
            parts = []
            if facts_str:
                parts.append(f"Facts: {facts_str}")
            if matched_snippets:
                parts.append(f"Snippet ({matched_snippets[0][1]}): {matched_snippets[0][3]}")
            return (
                f"Hillock (Offline Vault) > {' | '.join(parts)}",
                primed_info,
                hdc_fingerprint,
                "FREE_OFFLINE_VAULT",
            )
        self.last_receipt["turn_outcome"] = "FREE_OFFLINE_FALLBACK"
        return (
            "Hillock (Offline) > Ollama is not reachable and no local vault facts matched.",
            [],
            hdc_fingerprint,
            "FREE_OFFLINE_FALLBACK",
        )

    def _build_free_prompts(
        self,
        query: str,
        facts_str: str,
        snippets_str: str,
        primed_info: list,
        saved_memory: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        priming_str = ", ".join(f"{node} ({w:.2f})" for node, w in primed_info[:3]) if primed_info else ""
        has_vault = bool(facts_str or snippets_str)

        if has_vault:
            system_prompt = (
                "You are Hillock, a helpful local personal assistant. Be direct and natural. "
                "Use the provided Verified Facts and Source Snippets when relevant and briefly name the source file. "
                "If vault context does not cover everything, use general knowledge, "
                "but never invent fake meetings, events, or personal facts about the user."
            )
        else:
            system_prompt = (
                "You are Hillock, a helpful local personal assistant. Be direct and natural. "
                "No local vault files matched this prompt, so answer using general knowledge. "
                "Never invent fake personal appointments, events, or local documents; "
                "if planning details are missing, use empty placeholders or ask."
            )

        sections = []
        if saved_memory:
            sections.append(f"[SAVED TO MEMORY THIS TURN]: \"{saved_memory['text']}\" (Source: chat_memory)")
        if facts_str:
            sections.append(f"[VERIFIED FACTS]:\n{facts_str}")
        if snippets_str:
            sections.append(f"[SOURCE SNIPPETS]:\n{snippets_str}")
        if priming_str:
            sections.append(f"[ASSOCIATED TOPICS]: {priming_str}")
        sections.append(f"User: {query}")

        return system_prompt, "\n\n".join(sections)

    def _get_mode_prompts(
        self,
        query: str,
        facts_str: str,
        primed_info: list,
        hdc_fingerprint: list,
        is_refusal: bool = False,
    ) -> Tuple[str, str]:
        """Preserved STRICT prompt formatter (tested by Issue #35 & Aimee contracts)."""
        if is_refusal:
            system_prompt = (
                "You are a factual assistant. No verified facts matched the question. "
                "State clearly that you do not have verified information."
            )
            render_prompt = f"Question: {query}\nMemory: No verified facts found."
            return system_prompt, render_prompt

        if self.verbosity_mode == "STRICT":
            system_prompt = (
                "You are a professional fact renderer. Translate ONLY the provided fact into one sentence. "
                "Do not add any extra context, historical assumptions, or details."
            )
            render_prompt = f"Fact: {facts_str}\nQuestion: {query}"
            return system_prompt, render_prompt

        return self._build_free_prompts(query, facts_str, "", primed_info, None)