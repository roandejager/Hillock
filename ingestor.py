"""
Hillock Ingestor Module (TALON Integration - v0.2.4 / v0.3 Timing Refinement)
Routes bulk document extractions through the TALON Engine.
"""

import os
import re
import logging
import time
import datetime
from typing import List, Dict, Tuple, Set, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pypdf
except ImportError:
    pypdf = None

# Import TALON Engine
try:
    from talon_engine import TalonEngine
except ImportError:
    TalonEngine = None

logger = logging.getLogger("Hillock.Ingestor")

# Lazy-loaded global instance of TalonEngine
_talon_instance = None

def get_talon_engine() -> Optional[TalonEngine]:
    """Lazy-loads and caches the singleton TalonEngine instance."""
    global _talon_instance
    if _talon_instance is None and TalonEngine is not None:
        try:
            logger.info("Initializing TALON Engine inside Ingestor...")
            _talon_instance = TalonEngine(device="cuda:0")
        except Exception as e:
            logger.error(f"Failed to initialize TALON Engine: {e}")
            _talon_instance = None
    return _talon_instance


def get_raw_document_text(file_path: str) -> str:
    """Reads raw TXT or PDF files into a single document string."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        if pypdf is None:
            raise ImportError("The 'pypdf' package is required for PDFs. Run 'pip install pypdf'.")
        reader = pypdf.PdfReader(file_path)
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(pages_text)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def ingest_document_parallel(file_path: str, hillock) -> Tuple[str, Dict[str, float]]:
    """
    Ingests documents using the TALON Engine.
    Returns (summary_str, timing_stats_dict).
    """
    t_start = time.perf_counter()

    try:
        raw_text = get_raw_document_text(file_path)
    except Exception as e:
        return f"Error reading file '{file_path}': {e}", {}

    if not raw_text.strip():
        return f"File '{file_path}' is empty.", {}

    # Fetch TALON Engine
    talon = get_talon_engine()

    extracted_relations = []
    active_entities_to_update = set()

    if talon is not None:
        print(f"\nHillock [INGESTOR]: Routing '{os.path.basename(file_path)}' through TALON Engine (CUDA Accelerated)...")
        triples = talon.process_document(raw_text)

        for triple in triples:
            sub_raw = triple.get("subject", "")
            pred_raw = triple.get("predicate", "")
            obj_raw = triple.get("object", "")

            if not sub_raw or not pred_raw or not obj_raw:
                continue

            sub = hillock.resolve_entity_identity(sub_raw)
            obj = hillock.resolve_entity_identity(obj_raw)
            norm_pred = hillock.predicate_map.get(pred_raw.strip().lower(), pred_raw.strip().lower().replace(" ", "_"))

            extracted_relations.append((sub, norm_pred, obj))
            active_entities_to_update.add(sub)
            active_entities_to_update.add(obj)

            # Allocate / update HDC Hypervectors
            hillock.hdc.get_or_allocate_hypervector(sub)
            hillock.hdc.get_or_allocate_hypervector(obj)

            t_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t_stamp} [TALON EXTRACTED]: [{sub}] -[{norm_pred}]-> [{obj}]")

        if extracted_relations:
            hillock.kg.update_relations_batch(extracted_relations)
            hillock.plasticity.update_associations(active_entities_to_update)

    else:
        print(f"Hillock [INGESTOR]: TALON Engine unavailable. Falling back to legacy parsing...")

    t_end = time.perf_counter()

    # Retrieve timing milestones from talon instance
    t_first = getattr(talon, "t_first_triple", None) if talon else None
    t_last = getattr(talon, "t_last_triple", None) if talon else None

    if t_first is None:
        t_first = t_end
    if t_last is None:
        t_last = t_end

    cold_start_time = max(0.0, t_first - t_start)
    pure_extraction_time = max(0.0, t_last - t_first)
    total_time = max(0.0, t_end - t_start)

    sentences_count = len([s for s in re.split(r"[.!?\n]", raw_text) if s.strip()])
    pure_rate = sentences_count / pure_extraction_time if pure_extraction_time > 0 else 0.0

    specs_str = ""
    if psutil:
        cpu_p = psutil.cpu_percent(interval=None)
        ram_p = psutil.virtual_memory().percent
        specs_str = f" | CPU: {cpu_p:.1f}%, RAM: {ram_p:.1f}%"

    timing_stats = {
        "load_and_first_extraction_time": cold_start_time,
        "pure_extraction_time": pure_extraction_time,
        "pure_rate": pure_rate,
        "total_time": total_time,
        "total_sentences": sentences_count,
        "extracted_triples": len(extracted_relations)
    }

    summary = (
        f"\n"
        f"========================================================\n"
        f"        TALON ENGINE BULK INGESTION SUMMARY REPORT     \n"
        f"========================================================\n"
        f"  * File Processed             : {os.path.basename(file_path)}\n"
        f"  * Total Sentences            : {sentences_count}\n"
        f"  * Extracted Triples          : {len(extracted_relations)}\n"
        f"  * Model Load & Cold-Start    : {cold_start_time:.2f} seconds\n"
        f"  * Pure Extraction Duration   : {pure_extraction_time:.2f} seconds\n"
        f"  * Pure Extraction Rate       : {pure_rate:.1f} sentences/sec{specs_str}\n"
        f"  * Total Processing Time      : {total_time:.2f} seconds\n"
        f"========================================================"
    )
    return summary, timing_stats