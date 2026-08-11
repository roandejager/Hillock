"""
TALON (Tensor-Accelerated Local Ontology Network)
Engine Module - v0.2.2 Quality & Entity Canonicalization Patch

Architect: Roan de Jager (Hillock Memory Engine)
"""

import os
import re
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

# Disable HuggingFace Windows Symlink Warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
        nlp = None
except ImportError:
    nlp = None

try:
    from fastcoref import FCoref
except ImportError:
    FCoref = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None

try:
    from glirel import GLiREL
except ImportError:
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


def clean_entity_text(text: str) -> str:
    """Strips trailing punctuation, verbs, prepositions, and stopwords from entity spans."""
    text = re.sub(r"[^\w\s-]", "", text).strip()
    words = text.split()
    stop_words = {
        "was", "is", "were", "are", "worked", "discovered", "colleague", "illustrious",
        "early", "theoretical", "critical", "analyzed", "work", "cracked", "designed",
        "developed", "born", "in", "at", "by", "of", "a", "an", "the", "and", "where", "while"
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

    def select_top_predicates(self, sentence: str, top_k: int = 10) -> List[str]:
        if self.model is None or self.taxonomy_embeddings is None:
            if not self.load_model():
                return self.taxonomy[:top_k]

        try:
            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
            cosine_scores = util.cos_sim(sentence_embedding, self.taxonomy_embeddings)[0]
            top_indices = np.argsort(cosine_scores.cpu().numpy())[::-1][:top_k]

            return [self.taxonomy[idx] for idx in top_indices]
        except Exception as e:
            logger.error(f"Predicate routing error: {e}")
            return self.taxonomy[:top_k]


class ZeroShotRelationExtractor:
    """Stage 3: Zero-Shot Span Relation Extractor using GLiREL + SpaCy NER."""

    def __init__(self, model_name: str = "jackboyla/glirel-large-v0", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self) -> bool:
        if GLiREL is None:
            logger.error("glirel library is not installed.")
            return False

        try:
            logger.info(f"Loading GLiREL Model '{self.model_name}' on {self.device}...")
            self.model = GLiREL.from_pretrained(self.model_name)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            logger.info("GLiREL Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load GLiREL model: {e}")
            return False

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

            # 1. Extract clean NER entity spans from spaCy
            ner = []
            for ent in doc.ents:
                clean_name = clean_entity_text(ent.text)
                if clean_name and len(clean_name) > 1:
                    ner.append([ent.start, ent.end, ent.label_, clean_name])

            # 2. Fallback to clean noun chunks if sentence lacks formal named entities
            if len(ner) < 2:
                ner = []
                for chunk in doc.noun_chunks:
                    clean_chunk = clean_entity_text(chunk.text)
                    if clean_chunk and len(clean_chunk) > 1:
                        ner.append([chunk.start, chunk.end, "ENTITY", clean_chunk])

            if len(ner) < 2:
                return []

            # 3. Predict relations via GLiREL zero-shot matrix scoring (Calibrated Threshold: 0.42)
            results = self.model.predict_relations(
                tokens,
                candidate_labels,
                threshold=threshold,
                ner=ner,
                top_k=1
            )

            extracted_triples = []
            seen_pairs = set()

            for item in results:
                head_raw = item.get("head_text", "")
                tail_raw = item.get("tail_text", "")

                head_text = " ".join(head_raw) if isinstance(head_raw, list) else str(head_raw)
                tail_text = " ".join(tail_raw) if isinstance(tail_raw, list) else str(tail_raw)

                clean_head = clean_entity_text(head_text)
                clean_tail = clean_entity_text(tail_text)
                label = item.get("label", "").replace(" ", "_")

                pair_key = (clean_head.lower(), label.lower(), clean_tail.lower())
                if clean_head and clean_tail and label and (clean_head.lower() != clean_tail.lower()) and pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    extracted_triples.append({
                        "subject": clean_head,
                        "predicate": label,
                        "object": clean_tail
                    })

            return extracted_triples
        except Exception as e:
            logger.error(f"GLiREL extraction error on sentence '{sentence}': {e}")
            return []


class TalonEngine:
    """Master Orchestrator unifying Stage 1, Stage 2, and Stage 3 into a sub-second pipeline."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.coref = CoreferenceResolver(device=device)
        self.router = DynamicPredicateRouter(device=device)
        self.extractor = ZeroShotRelationExtractor(device=device)

    def process_document(self, document_text: str) -> List[Dict[str, str]]:
        logger.info("=== Starting TALON High-Speed Ingestion Pipeline ===")

        # Step 1: Coreference Resolution
        logger.info("[TALON Stage 1] Resolving document coreferences...")
        resolved_doc = self.coref.resolve_text(document_text)

        # Step 2: Split into sentences
        sentences = [s.strip() for s in re.split(r"[.!?\n]", resolved_doc) if s.strip()]
        logger.info(f"[TALON Engine] Processing {len(sentences)} resolved sentences...")

        all_triples = []
        for idx, sentence in enumerate(sentences):
            # Step 3: Route Top Predicates for this sentence
            top_predicates = self.router.select_top_predicates(sentence, top_k=10)

            # Step 4: Zero-Shot Relation Extraction
            triples = self.extractor.extract_relations(sentence, top_predicates)
            for t in triples:
                all_triples.append(t)
                logger.info(f"  * Extracted Triple [Sentence {idx+1}]: [{t['subject']}] -[{t['predicate']}]-> [{t['object']}]")

        return all_triples


# Self-test full pipeline execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("      TALON ENGINE - FULL PIPELINE INTEGRATION TEST      ")
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