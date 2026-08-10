"""
TALON (Tensor-Accelerated Local Ontology Network)
Engine Module - Stage 1: Document-Level Coreference Resolution ONLY

Architect: Roan de Jager (Hillock Memory Engine)
"""

import os
import logging

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


class CoreferenceResolver:
    """Handles document-level pronoun resolution using Fastcoref before text chunking."""

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
        """
        Replaces pronoun spans with canonical head entities.
        Replaces from back to front so character indices remain aligned.
        """
        replacements = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # First tuple in cluster is the primary head entity
            head_start, head_end = cluster[0]
            head_name = text[head_start:head_end].strip()

            if not head_name:
                continue

            # Replace subsequent mentions (pronouns) with head_name
            for start, end in cluster[1:]:
                replacements.append((start, end, head_name))

        # Sort replacements in reverse order of start position
        replacements.sort(key=lambda x: x[0], reverse=True)

        # Apply replacements from back to front
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
                # Extract character span clusters from Fastcoref
                clusters = predictions[0].get_clusters(as_strings=False)
                if clusters:
                    return self._replace_clusters_in_text(text, clusters)
        except Exception as e:
            logger.error(f"Coreference resolution error: {e}")

        return text


# Self-test Stage 1 execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("      TALON ENGINE - STAGE 1: COREFERENCE TEST ONLY      ")
    print("=" * 60)

    resolver = CoreferenceResolver()

    sample_text = (
        "Marie Curie was a brilliant physicist born in Warsaw. "
        "She discovered radioactivity and migrated to France. "
        "Her colleague Albert Einstein worked alongside her in physics."
    )

    print("\n[Original Input Text]:")
    print(sample_text)

    print("\n[Processing via Coreference Resolver]...")
    resolved_text = resolver.resolve_text(sample_text)

    print("\n[Resolved Output Text]:")
    print(resolved_text)
    print("=" * 60)