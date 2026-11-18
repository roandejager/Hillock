"""
Hillock Ingestor Module (TALON Integration - v0.4.1 Precision)
Routes bulk document extractions through the TALON Engine.
"""

from __future__ import annotations

import os
import re
import logging
import time
import datetime
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from talon_engine import TalonEngine

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
    from talon_engine import TalonEngine as _TalonEngine
except ImportError:
    _TalonEngine = None

logger = logging.getLogger("Hillock.Ingestor")

# Lazy-loaded global instance of TalonEngine
_talon_instance = None

def get_talon_engine() -> Optional[TalonEngine]:
    """Lazy-loads and caches the singleton TalonEngine instance."""
    global _talon_instance
    if _talon_instance is None and _TalonEngine is not None:
        try:
            logger.info("Initializing TALON Engine inside Ingestor...")
            import torch
            # Dynamically detect if a GPU is available, otherwise default to CPU
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Routing TALON extractions to device: {target_device}")
            _talon_instance = _TalonEngine(device=target_device)
        except Exception as e:
            logger.error(f"Failed to initialize TALON Engine: {e}")
            _talon_instance = None
    return _talon_instance


def _resolve_safe_project_file(file_path: str) -> Path:
    """Validates that file_path exists and lives inside the project directory."""
    requested_path = Path(file_path).resolve()
    project_root = Path(__file__).parent.resolve()
    try:
        requested_path.relative_to(project_root)
    except ValueError:
        raise ValueError(
            f"Access denied: '{file_path}' is outside the project directory."
        )
    if not requested_path.is_file():
        raise FileNotFoundError(f"File '{file_path}' does not exist or is not a file.")
    return requested_path


def get_raw_document_text(file_path: str) -> str:
    """
    Safely read a text or PDF file from within the project directory.
    Prevents path traversal attacks (e.g., /ingest ../../etc/passwd).
    """
    requested_path = _resolve_safe_project_file(file_path)
    ext = requested_path.suffix.lower()

    if ext == ".pdf":
        if pypdf is None:
            raise ImportError("pypdf is required to read PDF files. Install it via 'pip install pypdf'.")
        reader = pypdf.PdfReader(str(requested_path))
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(pages_text)
    else:
        with open(requested_path, "r", encoding="utf-8") as f:
            return f.read()


def get_raw_document_segments(file_path: str, max_chars_per_chunk: int = 3500) -> List[Tuple[str, str]]:
    """
    Reads a safe project file into labeled (page_or_line, text) segments so PDF pages
    and long text files stay bounded in laptop RAM during coreference resolution.
    """
    requested_path = _resolve_safe_project_file(file_path)
    ext = requested_path.suffix.lower()

    if ext == ".pdf":
        if pypdf is None:
            raise ImportError("pypdf is required to read PDF files. Install it via 'pip install pypdf'.")
        reader = pypdf.PdfReader(str(requested_path))
        segments: List[Tuple[str, str]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                segments.append((f"page {page_num}", page_text))
        return segments

    with open(requested_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return []

    segments = []
    buf: List[str] = []
    buf_chars = 0
    start_line = 1

    for idx, line in enumerate(lines, start=1):
        buf.append(line)
        buf_chars += len(line) + 1
        if buf_chars >= max_chars_per_chunk:
            chunk_text = "\n".join(buf)
            if chunk_text.strip():
                segments.append((f"lines {start_line}-{idx}", chunk_text))
            buf = []
            buf_chars = 0
            start_line = idx + 1

    if buf:
        chunk_text = "\n".join(buf)
        if chunk_text.strip():
            segments.append((f"lines {start_line}-{len(lines)}", chunk_text))

    return segments


def _build_sentence_windows(text: str, window_size: int = 3, stride: int = 2) -> List[str]:
    """Splits raw text into overlapping 1-3 sentence windows for fallback snippet search."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    windows: List[str] = []
    for idx in range(0, len(sentences), stride):
        chunk = " ".join(sentences[idx : idx + window_size]).strip()
        if chunk:
            windows.append(chunk)
        if idx + window_size >= len(sentences):
            break
    return windows


def ingest_document_parallel(file_path: str, hillock) -> Tuple[str, Dict[str, float]]:
    """
    Ingests documents using the TALON Engine and stores linked source snippets.
    Returns (summary_str, timing_stats_dict).
    """
    global _talon_instance
    t_start = time.perf_counter()

    try:
        segments = get_raw_document_segments(file_path)
    except Exception as e:
        return f"Error reading file '{file_path}': {e}", {}

    if not segments:
        return f"File '{file_path}' is empty.", {}

    # Fetch TALON Engine (halts loudly if uninstalled, preserving verify_hillock check)
    talon = get_talon_engine()

    if talon is None or talon.extractor.model is None:
        missing_warning = (
            "\n"
            "========================================================\n"
            "  ⚠️ WARNING: TALON EXTRACTION ENGINE IS UN-INSTALLED  \n"
            "========================================================\n"
            "  Required dependencies (glirel, fastcoref, spacy, torch) \n"
            "  are missing or failed to initialize.\n\n"
            "  Fix this by running:\n"
            "    pip install -r requirements.txt\n"
            "    python -m spacy download en_core_web_sm\n"
            "========================================================\n"
        )
        print(missing_warning)
        logger.error("Ingestion halted: TALON extraction stack is uninstalled or incomplete.")
        return missing_warning, {}

    doc_name = os.path.basename(file_path)
    device_label = "CUDA Accelerated" if "cuda" in str(getattr(talon, "device", "cpu")).lower() else "CPU"
    print(f"\nHillock [INGESTOR]: Routing '{doc_name}' through TALON Engine ({device_label})...")

    extracted_relations: List[Tuple[str, str, str]] = []
    triple_snippet_ids: List[Optional[str]] = []
    snippets_by_id: Dict[str, Tuple[str, str, str, str]] = {}
    active_entities_to_update: Set[str] = set()
    multi_hop_paths = []

    overall_t_first: Optional[float] = None
    overall_t_last: Optional[float] = None
    total_sentences = 0

    for page_or_line, seg_text in segments:
        total_sentences += len([s for s in re.split(r"[.!?\n]", seg_text) if s.strip()])

        # Always capture 1-3 sentence windows so dense prose remains searchable in FREE mode
        seg_windows = _build_sentence_windows(seg_text)
        fallback_sid: Optional[str] = None
        for win_text in seg_windows:
            sid = hillock.kg.make_snippet_id(doc_name, page_or_line, win_text)
            snippets_by_id[sid] = (sid, doc_name, page_or_line, win_text)
            if fallback_sid is None:
                fallback_sid = sid

        triples = talon.process_document(seg_text)
        seg_t_first = getattr(talon, "t_first_triple", None)
        seg_t_last = getattr(talon, "t_last_triple", None)
        if overall_t_first is None and seg_t_first is not None:
            overall_t_first = seg_t_first
        if seg_t_last is not None:
            overall_t_last = seg_t_last

        for triple in triples:
            sub_raw = triple.get("subject", "")
            pred_raw = triple.get("predicate", "")
            obj_raw = triple.get("object", "")

            if not sub_raw or not pred_raw or not obj_raw:
                continue

            sub = hillock.resolve_entity_identity(sub_raw)
            obj = hillock.resolve_entity_identity(obj_raw)
            norm_pred = hillock.predicate_map.get(
                pred_raw.strip().lower(),
                pred_raw.strip().lower().replace(" ", "_"),
            )

            ctx_text = (triple.get("context_snippet") or triple.get("source_sentence") or "").strip()
            if ctx_text:
                snip_id = hillock.kg.make_snippet_id(doc_name, page_or_line, ctx_text)
                snippets_by_id[snip_id] = (snip_id, doc_name, page_or_line, ctx_text)
            else:
                snip_id = fallback_sid

            extracted_relations.append((sub, norm_pred, obj))
            triple_snippet_ids.append(snip_id)
            active_entities_to_update.add(sub)
            active_entities_to_update.add(obj)

            hillock.hdc.get_or_allocate_hypervector(sub)
            hillock.hdc.get_or_allocate_hypervector(obj)

            t_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{t_stamp} [TALON EXTRACTED]: [{sub}] -[{norm_pred}]-> [{obj}]")

    if snippets_by_id:
        hillock.kg.save_snippets_batch(list(snippets_by_id.values()))

    if extracted_relations:
        hillock.kg.update_relations_batch(
            extracted_relations,
            source_doc=doc_name,
            snippet_ids=triple_snippet_ids,
        )
        hillock.plasticity.update_associations(active_entities_to_update)

        # --- HYPERGRAPH-HDC Multi-Hop Path Generation & Pruning ---
        adj = {}
        for idx, (sub, pred, obj) in enumerate(extracted_relations):
            if sub not in adj:
                adj[sub] = []
            adj[sub].append((pred, obj, idx))

        MAX_WINDOW = 6

        for start_node in adj:
            for p1, n1, idx1 in adj[start_node]:
                if n1 in adj:
                    for p2, n2, idx2 in adj[n1]:
                        if abs(idx2 - idx1) <= MAX_WINDOW and n2 != start_node:
                            multi_hop_paths.append(([start_node, n1, n2], [p1, p2]))

                            if n2 in adj:
                                for p3, n3, idx3 in adj[n2]:
                                    if abs(idx3 - idx1) <= MAX_WINDOW and n3 not in (start_node, n1):
                                        multi_hop_paths.append(([start_node, n1, n2, n3], [p1, p2, p3]))

        print(f"  [HYPERGRAPH-HDC]: Pruned and generated {len(multi_hop_paths)} valid multi-hop relational paths.")

        doc_accumulator = np.zeros(hillock.hdc.D, dtype=np.float64)
        for ents, preds in multi_hop_paths:
            path_hv = hillock.hdc.bind_sequential_path(ents, preds)
            doc_accumulator += path_hv.astype(np.float64)

        hillock.hdc.state += doc_accumulator
        hillock.kg.save_document_reservoir(doc_name, doc_accumulator, len(multi_hop_paths))

    t_end = time.perf_counter()

    if overall_t_first is None:
        overall_t_first = t_end
    if overall_t_last is None:
        overall_t_last = t_end

    cold_start_time = max(0.0, overall_t_first - t_start)
    pure_extraction_time = max(0.0, overall_t_last - overall_t_first)
    total_time = max(0.0, t_end - t_start)

    pure_rate = total_sentences / pure_extraction_time if pure_extraction_time > 0 else 0.0

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
        "total_sentences": total_sentences,
        "extracted_triples": len(extracted_relations),
        "snippets_saved": len(snippets_by_id),
    }

    summary = (
        f"\n"
        f"========================================================\n"
        f"        TALON ENGINE BULK INGESTION SUMMARY REPORT     \n"
        f"========================================================\n"
        f"  * File Processed             : {doc_name}\n"
        f"  * Total Sentences            : {total_sentences}\n"
        f"  * Extracted 1-Hop Triples    : {len(extracted_relations)}\n"
        f"  * Source Snippets Stored     : {len(snippets_by_id)}\n"
        f"  * Multi-Hop Paths Bound      : {len(multi_hop_paths)}\n"
        f"  * Model Load & Cold-Start    : {cold_start_time:.2f} seconds\n"
        f"  * Pure Extraction Duration   : {pure_extraction_time:.2f} seconds\n"
        f"  * Pure Extraction Rate       : {pure_rate:.1f} sentences/sec{specs_str}\n"
        f"  * Total Processing Time      : {total_time:.2f} seconds\n"
        f"========================================================"
    )
    talon.unload_models()
    _talon_instance = None
    return summary, timing_stats