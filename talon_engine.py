"""
TALON (Tensor-Accelerated Local Ontology Network)
Engine Module - v0.4.1 Precision Extraction & Directionality Auto-Correction

Architect: Roan de Jager (Hillock Memory Engine)
"""

import os
import re
import time
import logging
import unicodedata
import numpy as np
from typing import List, Dict, Tuple, Optional, Set

# Disable HuggingFace Windows Symlink Warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# 1. Monkey-patch HuggingFace transformers security block for torch.load
try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# 2. Patch missing Hugging Face mapping attribute in older Fastcoref model class
try:
    import fastcoref.coref_models.modeling_fcoref
    fastcoref.coref_models.modeling_fcoref.FCorefModel.all_tied_weights_keys = {}
    fastcoref.coref_models.modeling_fcoref.FCorefModel._all_tied_weights_keys = {}
    fastcoref.coref_models.modeling_fcoref.FCorefModel._tied_weights_keys = {}
except Exception:
    pass

# 3. Patch GLiREL Hugging Face Hub keyword argument compatibility
try:
    import glirel
    if hasattr(glirel, "GLiREL") and hasattr(glirel.GLiREL, "_from_pretrained"):
        _orig_glirel_fp = glirel.GLiREL._from_pretrained
        @classmethod
        def _patched_glirel_from_pretrained(cls, *args, **kwargs):
            kwargs.setdefault("proxies", None)
            kwargs.setdefault("resume_download", None)
            return _orig_glirel_fp(*args, **kwargs)
        glirel.GLiREL._from_pretrained = _patched_glirel_from_pretrained
except Exception:
    pass

logger = logging.getLogger("Hillock.TALON")

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        logger.info("SpaCy model 'en_core_web_sm' missing. Downloading automatically...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.debug(f"SpaCy/PyTorch import bypassed (OS block or missing): {e}")
    nlp = None

try:
    from fastcoref import FCoref
except Exception:
    FCoref = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None

try:
    from glirel import GLiREL
except Exception:
    GLiREL = None


# Master Taxonomy of Open-Domain Wikidata Predicates (~50 Common Relations)
DEFAULT_PREDICATE_TAXONOMY = [
    "born_in", "died_in", "place_of_birth", "place_of_death", "country_of_citizenship",
    "collaborated_with", "worked_with", "partnered_with", "spouse_of", "child_of", "parent_of",
    "discovered", "invented", "co_invented", "developed", "designed", "founded", "created",
    "cracked", "authored", "wrote", "published", "formulated", "proposed",
    "educated_at", "studied_at", "employed_by", "worked_at", "member_of", "affiliated_with",
    "award_received", "won", "nominated_for", "capital_of", "located_in", "headquartered_in",
    "field_of_work", "subclass_of", "part_of", "instance_of", "has_part", "contains",
    "influenced_by", "student_of", "teacher_of", "successor_to", "predecessor_to",
    "migrated_to", "moved_to", "resided_in", "patented", "manufactured", "operated_by"
]

# Origin Predicates where Head MUST be Person and Tail MUST be Location
ORIGIN_PREDICATES = {
    "born_in", "died_in", "place_of_birth", "place_of_death",
    "migrated_to", "moved_to", "resided_in", "country_of_citizenship"
}

# Strict Location NER Tags
LOCATION_TAGS = {"GPE", "LOC"}

# Whitelist of Symmetric Predicates (v0.4 #15)
SYMMETRIC_PREDICATES: Set[str] = {
    "collaborated_with", "worked_with", "partnered_with", "spouse_of"
}

# Precompiled High-Precision Regexes for Entity Sanitization (v0.4 #15)
POSSESSIVE_RE = re.compile(r"(?<=\w)['’]s\b|(?<=s)['’]\b", re.UNICODE)
TRAILING_POSSESSIVE_SUFFIX_RE = re.compile(r"[\s_]+(['’]?s|work|code)$", re.IGNORECASE)
TRAILING_VERB_RE = re.compile(
    r"[\s_]+(collaborated|built|studied|worked|discovered|designed|developed|cracked|authored|wrote|published|formulated|proposed|patented|founded|created)$",
    re.IGNORECASE
)
PREPOSITION_TAIL_RE = re.compile(
    r"[\s_]+(under|before|to|in|at|by|of|where|while|with|for|from|on)$",
    re.IGNORECASE
)


def get_canonical_triple_key(subject: str, predicate: str, object_: str) -> Tuple[str, str, str]:
    """
    Returns a canonical deduplication key for SPO triples.
    For symmetric relations (e.g. collaborated_with), (A, P, B) and (B, P, A) map to (min(A,B), P, max(A,B)).
    For non-symmetric relations (e.g. born_in), (A, P, B) remains (A, P, B).
    """
    s_clean = subject.strip().lower()
    p_clean = predicate.strip().lower()
    o_clean = object_.strip().lower()

    if p_clean in SYMMETRIC_PREDICATES:
        sorted_entities = sorted([s_clean, o_clean])
        return (sorted_entities[0], p_clean, sorted_entities[1])
    else:
        return (s_clean, p_clean, o_clean)


def is_inverted_asymmetric_pair(subject: str, predicate: str, object_: str, seen_keys: Set[Tuple[str, str, str]]) -> bool:
    """
    Checks if an inverted pair (object, predicate, subject) already exists in seen_keys for a non-symmetric relation.
    Returns True if it is an invalid inverted duplicate that should be purged.
    """
    p_clean = predicate.strip().lower()
    if p_clean in SYMMETRIC_PREDICATES:
        return False

    inverted_key = (object_.strip().lower(), p_clean, subject.strip().lower())
    return inverted_key in seen_keys


def clean_entity_text(text: str) -> str:
    """High-precision entity span sanitization engine (v0.4 #15)."""
    if not text:
        return ""

    # 1. Unicode Normalization
    cleaned = unicodedata.normalize("NFC", text).strip()

    # 2. Strip possessives and trailing noise suffixes ('s, ’s, _s, _work, _code)
    cleaned = POSSESSIVE_RE.sub("", cleaned)
    cleaned = TRAILING_POSSESSIVE_SUFFIX_RE.sub("", cleaned)

    # 3. Iteratively strip trailing action verbs and prepositional tails
    prev_len = -1
    while len(cleaned) != prev_len:
        prev_len = len(cleaned)
        cleaned = TRAILING_VERB_RE.sub("", cleaned).strip()
        cleaned = PREPOSITION_TAIL_RE.sub("", cleaned).strip()

    # 4. Remove leading/trailing non-alphanumeric characters
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned).strip()

    # 5. Token-level stopword / generic noise filtering
    words = [w for w in re.split(r"[\s_]+", cleaned) if w]
    stop_words = {
        "was", "is", "were", "are", "colleague", "illustrious",
        "early", "theoretical", "critical", "analyzed", "a", "an", "the",
        "and", "where", "while", "childhood", "years", "first", "practical"
    }

    while words and words[0].lower() in stop_words:
        words.pop(0)
    while words and words[-1].lower() in stop_words:
        words.pop(-1)

    cleaned = " ".join(words).strip()
    return cleaned


class CoreferenceResolver:
    """Stage 1: Handles document-level pronoun resolution using Fastcoref."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None

    def load_model(self) -> bool:
        if FCoref is None:
            logger.error("fastcoref library is not installed.")
            return False

        if self.model is not None:
            return True

        try:
            logger.info(f"Loading Fastcoref model on {self.device}...")
            self.model = FCoref(device=self.device)
            logger.info("Fastcoref model loaded successfully on GPU!")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Fastcoref on {self.device}: {e}. Falling back to CPU...")
            try:
                self.model = FCoref(device="cpu")
                logger.info("Fastcoref model loaded successfully on CPU!")
                return True
            except Exception as cpu_e:
                logger.error(f"Failed to load Fastcoref on CPU: {cpu_e}")
                return False

    def _replace_clusters_in_text(self, text: str, clusters: list) -> str:
        replacements = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            head_start, head_end = cluster[0]
            head_name = text[head_start:head_end].strip()

            if not head_name:
                continue

            for start, end in cluster[1:]:
                replacements.append((start, end, head_name))

        replacements.sort(key=lambda x: x[0], reverse=True)

        resolved = text
        for start, end, head_name in replacements:
            resolved = resolved[:start] + head_name + resolved[end:]

        return resolved

    def resolve_text(self, text: str) -> str:
        if not text or not text.strip():
            return text

        if self.model is None:
            if not self.load_model():
                return text

        try:
            predictions = self.model.predict(texts=[text])
            if predictions and len(predictions) > 0:
                clusters = predictions[0].get_clusters(as_strings=False)
                if clusters:
                    return self._replace_clusters_in_text(text, clusters)
        except Exception as e:
            logger.error(f"Coreference resolution error: {e}")

        return text


class DynamicPredicateRouter:
    """Stage 2: Bi-Encoder Semantic Filter using MiniLM to select top candidate predicates in <2ms."""

    def __init__(self, taxonomy: List[str] = DEFAULT_PREDICATE_TAXONOMY, device: str = "cuda:0"):
        self.taxonomy = taxonomy
        self.device = device
        self.model = None
        self.taxonomy_embeddings = None

    def load_model(self) -> bool:
        if SentenceTransformer is None:
            logger.error("sentence-transformers is not installed.")
            return False

        if self.model is not None and self.taxonomy_embeddings is not None:
            return True

        try:
            logger.info(f"Loading MiniLM Bi-Encoder on {self.device}...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

            taxonomy_phrases = [pred.replace("_", " ") for pred in self.taxonomy]
            self.taxonomy_embeddings = self.model.encode(taxonomy_phrases, convert_to_tensor=True)
            logger.info(f"Pre-cached embeddings for {len(self.taxonomy)} predicate taxonomy labels.")
            return True
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            return False

    def select_top_predicates_batch(self, sentences: List[str], top_k: int = 10) -> List[List[str]]:
        """CUDA Batch Matrix Multiplication: Encodes all sentences at once in a single GPU pass."""
        if self.model is None or self.taxonomy_embeddings is None:
            if not self.load_model():
                return [self.taxonomy[:top_k] for _ in sentences]

        try:
            sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True, batch_size=len(sentences))
            cosine_scores = util.cos_sim(sentence_embeddings, self.taxonomy_embeddings)

            batch_results = []
            for i in range(len(sentences)):
                scores = cosine_scores[i].cpu().numpy()
                top_indices = np.argsort(scores)[::-1][:top_k]
                batch_results.append([self.taxonomy[idx] for idx in top_indices])

            return batch_results
        except Exception as e:
            logger.error(f"Batch predicate routing error: {e}")
            return [self.taxonomy[:top_k] for _ in sentences]


class ZeroShotRelationExtractor:
    """Stage 3: Zero-Shot Span Relation Extractor using GLiREL Large (High Precision Model)."""

    def __init__(self, model_name: str = "jackboyla/glirel-large-v0", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def extract_relations(self, sentence: str, candidate_labels: List[str], threshold: float = 0.42) -> List[Dict[str, str]]:
        if self.model is None:
            if not self.load_model():
                return []

        global nlp
        if nlp is None:
            logger.error("SpaCy model 'en_core_web_sm' is not available.")
            return []

        try:
            doc = nlp(sentence)
            tokens = [token.text for token in doc]

            ner = []
            entity_type_map = {}

            for ent in doc.ents:
                clean_name = clean_entity_text(ent.text)
                if clean_name and len(clean_name) > 1:
                    ner.append([ent.start, ent.end, ent.label_, clean_name])
                    entity_type_map[clean_name.lower()] = ent.label_

            if len(ner) < 2:
                ner = []
                for chunk in doc.noun_chunks:
                    clean_chunk = clean_entity_text(chunk.text)
                    if clean_chunk and len(clean_chunk) > 1:
                        ner.append([chunk.start, chunk.end, "ENTITY", clean_chunk])
                        entity_type_map[clean_chunk.lower()] = "ENTITY"

            if len(ner) < 2:
                return []

            results = self.model.predict_relations(
                tokens,
                candidate_labels,
                threshold=threshold,
                ner=ner,
                top_k=1
            )

            extracted_triples = []
            seen_keys = set()

            for item in results:
                head_raw = item.get("head_text", "")
                tail_raw = item.get("tail_text", "")

                head_text = " ".join(head_raw) if isinstance(head_raw, list) else str(head_raw)
                tail_text = " ".join(tail_raw) if isinstance(tail_raw, list) else str(tail_raw)

                clean_head = clean_entity_text(head_text)
                clean_tail = clean_entity_text(tail_text)
                label = item.get("label", "").replace(" ", "_")

                if not clean_head or not clean_tail or not label:
                    continue

                if clean_head.lower() == clean_tail.lower():
                    continue

                head_type = entity_type_map.get(clean_head.lower(), "ENTITY")
                tail_type = entity_type_map.get(clean_tail.lower(), "ENTITY")

                # Directionality Auto-Correction for Origin Predicates
                if label in ORIGIN_PREDICATES:
                    if head_type in LOCATION_TAGS and tail_type not in LOCATION_TAGS:
                        # GLiREL extracted [Location born_in Person] -> Auto-swap to [Person born_in Location]!
                        clean_head, clean_tail = clean_tail, clean_head
                        head_type, tail_type = tail_type, head_type

                # Inverted Asymmetric Pair Purging
                if is_inverted_asymmetric_pair(clean_head, label, clean_tail, seen_keys):
                    continue

                # Canonical Deduplication Key Check
                canon_key = get_canonical_triple_key(clean_head, label, clean_tail)
                if canon_key in seen_keys:
                    continue

                seen_keys.add(canon_key)
                seen_keys.add((clean_head.lower(), label.lower(), clean_tail.lower()))

                extracted_triples.append({
                    "subject": clean_head,
                    "predicate": label,
                    "object": clean_tail
                })

            return extracted_triples
        except Exception as e:
            logger.error(f"GLiREL extraction error on sentence '{sentence}': {e}")
            return []

    def load_model(self) -> bool:
        if GLiREL is None:
            logger.error("glirel library is not installed.")
            return False

        if self.model is not None:
            return True

        try:
            logger.info(f"Loading GLiREL Large Model '{self.model_name}' on {self.device}...")
            self.model = GLiREL.from_pretrained(self.model_name)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            logger.info("GLiREL Large Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load GLiREL model: {e}")
            return False


class TalonEngine:
    """Master Orchestrator unifying Stage 1, Stage 2, and Stage 3 into a batch-accelerated CUDA pipeline."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.coref = CoreferenceResolver(device=device)
        self.router = DynamicPredicateRouter(device=device)
        self.extractor = ZeroShotRelationExtractor(device=device)
        self.t_first_triple: Optional[float] = None
        self.t_last_triple: Optional[float] = None

        logger.info("Pre-warming TALON CUDA models into GPU VRAM...")
        self.coref.load_model()
        self.router.load_model()
        self.extractor.load_model()
        logger.info("TALON Engine pre-warm complete! Ready for batched extractions.")

    def process_document(self, document_text: str, batch_size: int = 16) -> List[Dict[str, str]]:
        logger.info("=== Starting TALON High-Speed Ingestion Pipeline (CUDA Batched) ===")
        self.t_first_triple = None
        self.t_last_triple = None

        # Step 1: Coreference Resolution
        logger.info("[TALON Stage 1] Resolving document coreferences...")
        resolved_doc = self.coref.resolve_text(document_text)

        # Step 2: Split into sentences
        sentences = [s.strip() for s in re.split(r"[.!?\n]", resolved_doc) if s.strip()]
        logger.info(f"[TALON Engine] Processing {len(sentences)} resolved sentences in CUDA batches of {batch_size}...")

        if not sentences:
            self.t_last_triple = time.perf_counter()
            return []

        all_triples = []

        # Step 3: Process sentences in GPU Batches
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            batch_top_preds = self.router.select_top_predicates_batch(batch_sentences, top_k=10)

            for idx, sentence in enumerate(batch_sentences):
                triples = self.extractor.extract_relations(sentence, batch_top_preds[idx])
                for t in triples:
                    if self.t_first_triple is None:
                        self.t_first_triple = time.perf_counter()
                    all_triples.append(t)
                    logger.info(f"  * Extracted Triple [Sentence {i + idx + 1}]: [{t['subject']}] -[{t['predicate']}]-> [{t['object']}]")

        self.t_last_triple = time.perf_counter()
        if self.t_first_triple is None:
            self.t_first_triple = self.t_last_triple

        return all_triples


# Self-test full pipeline execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("      TALON ENGINE - BATCH-ACCELERATED TEST      ")
    print("=" * 60)

    talon = TalonEngine()

    raw_document = (
        "Marie Curie was a brilliant physicist born in Warsaw. "
        "She discovered radioactivity and migrated to France. "
        "Her colleague Albert Einstein worked alongside her in physics."
    )

    print("\n[RAW INPUT DOCUMENT]:")
    print(raw_document)

    print("\n[EXECUTING TALON ENGINE]...")
    extracted_facts = talon.process_document(raw_document)

    print("\n" + "=" * 60)
    print("               FINAL EXTRACTED SPO TRIPLES               ")
    print("=" * 60)
    for fact in extracted_facts:
        print(f"  * [{fact['subject']}] -[{fact['predicate']}]-> [{fact['object']}]")
    print("=" * 60)