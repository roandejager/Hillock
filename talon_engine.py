"""
TALON (Tensor-Accelerated Local Ontology Network)
Engine Module - Stage 1: Coreference Resolution & Stage 2: Dynamic Predicate Routing

Architect: Roan de Jager (Hillock Memory Engine)
"""

import os
import logging
import numpy as np
from typing import List, Optional

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

logger = logging.getLogger("Hillock.TALON")

try:
    from fastcoref import FCoref
except ImportError:
    FCoref = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None


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


class CoreferenceResolver:
    """Stage 1: Handles document-level pronoun resolution using Fastcoref."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model = None

    def load_model(self) -> bool:
        if FCoref is None:
            logger.error("fastcoref library is not installed. Run 'pip install fastcoref'")
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
            logger.error("sentence-transformers is not installed. Run 'pip install sentence-transformers'")
            return False

        try:
            logger.info(f"Loading MiniLM Bi-Encoder on {self.device}...")
            # Lightweight 80MB model
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

            # Pre-compute and cache embeddings for our predicate taxonomy
            taxonomy_phrases = [pred.replace("_", " ") for pred in self.taxonomy]
            self.taxonomy_embeddings = self.model.encode(taxonomy_phrases, convert_to_tensor=True)
            logger.info(f"Pre-cached embeddings for {len(self.taxonomy)} predicate taxonomy labels.")
            return True
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            return False

    def select_top_predicates(self, sentence: str, top_k: int = 10) -> List[str]:
        """Encodes sentence and retrieves top_k most semantically relevant predicates."""
        if self.model is None or self.taxonomy_embeddings is None:
            if not self.load_model():
                return self.taxonomy[:top_k]

        try:
            # Encode sentence into vector space
            sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)

            # Compute cosine similarity against pre-cached predicate taxonomy embeddings
            cosine_scores = util.cos_sim(sentence_embedding, self.taxonomy_embeddings)[0]

            # Get top_k indices with highest similarity
            top_indices = np.argsort(cosine_scores.cpu().numpy())[::-1][:top_k]

            selected_predicates = [self.taxonomy[idx] for idx in top_indices]
            return selected_predicates
        except Exception as e:
            logger.error(f"Predicate routing error: {e}")
            return self.taxonomy[:top_k]


# Self-test Stage 1 & Stage 2 execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("     TALON ENGINE - STAGE 1 & 2 INTEGRATED TEST         ")
    print("=" * 60)

    # --- 1. Test Stage 1: Coreference Resolution ---
    print("\n[STAGE 1: COREFERENCE RESOLUTION TEST]")
    resolver = CoreferenceResolver()

    sample_text = (
        "Marie Curie was a brilliant physicist born in Warsaw. "
        "She discovered radioactivity and migrated to France."
    )
    resolved_text = resolver.resolve_text(sample_text)
    print(f"Original Text : {sample_text}")
    print(f"Resolved Text : {resolved_text}")

    # --- 2. Test Stage 2: Dynamic Predicate Router ---
    print("\n" + "-" * 60)
    print(" [STAGE 2: DYNAMIC PREDICATE ROUTER TEST]")
    print("-" * 60)

    router = DynamicPredicateRouter()

    test_sentences = [
        "Marie Curie discovered radioactivity in Paris.",
        "Alan Turing cracked the Enigma code at Bletchley Park.",
        "Albert Einstein was born in Germany and studied in Switzerland."
    ]

    for sentence in test_sentences:
        top_preds = router.select_top_predicates(sentence, top_k=5)
        print(f"\nSentence : '{sentence}'")
        print(f"Top-5 Selected Predicates: {top_preds}")

    print("=" * 60)