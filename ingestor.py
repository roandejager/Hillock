"""Runs parallelized paragraph block extractions with real-time hardware monitoring."""

import os
import re
import json
import logging
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set
from config import BLOCK_SIZE, BLOCK_OVERLAP, MAX_WORKERS

# Optional import for hardware resource tracking
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("Hillock.Ingestor")

try:
    import pypdf
except ImportError:
    pypdf = None

def get_clean_sentences(file_path: str, hillock) -> List[str]:
    """Reads raw TXT or PDF files and segments them into cleaned sentence arrays."""
    ext = os.path.splitext(file_path)[1].lower()
    raw_text = ""

    if ext == ".pdf":
        if pypdf is None:
            raise ImportError("The 'pypdf' package is required for PDFs. Run 'pip install pypdf'.")
        reader = pypdf.PdfReader(file_path)
        pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
        raw_text = "\n".join(pages_text)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

    sentences = [s.strip() for s in re.split(r"[.!?\n]", raw_text) if s.strip()]
    return sentences

def segment_into_overlapping_blocks(sentences: List[str], block_size: int = BLOCK_SIZE, overlap: int = BLOCK_OVERLAP) -> List[str]:
    """Chunks sentence arrays into overlapping paragraph blocks."""
    blocks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + block_size]
        block_text = " ".join(chunk)
        blocks.append(block_text)
        i += (block_size - overlap)
        if i >= len(sentences) or len(chunk) < block_size:
            break
    return blocks

def extract_relations_from_block(block_text: str, hillock) -> List[Dict[str, str]]:
    """Passes a paragraph block to the local LLM to extract all relational triples."""
    system_prompt = (
        "You are a precise, structural information extraction engine. Output ONLY a JSON array.\n"
        "Analyze the provided text block and extract all clear factual relationships between prominent entities.\n\n"
        "Rules:\n"
        "1. Output ONLY a valid JSON list of objects, do not include any other conversational filler.\n"
        "2. Fields required:\n"
        "   - 'subject': string (normalized snake_case entity name, e.g. 'Marie_Curie' or 'Alan_Turing')\n"
        "   - 'predicate': string (relationship verb, e.g. 'born_in', 'collaborated_with', 'discovered', 'cracked')\n"
        "   - 'object': string (normalized snake_case target entity, e.g. 'Germany')\n"
        "3. Extract as many valid connections as are explicitly supported by the text. Multiple connections per block are expected.\n"
        "4. Be extremely careful with subject-predicate association. Only extract a relationship if the subject "
        "explicitly performed the action described by the predicate in the text. "
        "Do not attribute actions to adjacent entities mentioned in other sentences.\n"
        "5. If no clear relationships exist, return an empty array [].\n"
    )

    response = hillock.query_ollama(block_text, system_prompt)
    if not response:
        return []

    extracted_triples = hillock.parse_json_safely(response)
    if isinstance(extracted_triples, list):
        return extracted_triples

    # Regex fallback if JSON parsing fails on small models
    triples = []
    try:
        matches = re.findall(r"\{\s*\"subject\"\s*:\s*\"([^\"]+)\"\s*,\s*\"predicate\"\s*:\s*\"([^\"]+)\"\s*,\s*\"object\"\s*:\s*\"([^\"]+)\"\s*\}", response)
        for sub, pred, obj in matches:
            triples.append({"subject": sub, "predicate": pred, "object": obj})
    except Exception:
        pass
    return triples

def ingest_document_parallel(file_path: str, hillock) -> str:
    """Orchestrates high-speed paragraph extraction with real-time logs and specs."""
    start_time = time.perf_counter()

    try:
        sentences = get_clean_sentences(file_path, hillock)
    except Exception as e:
        return str(e)

    # Group sentences into paragraph blocks
    blocks = segment_into_overlapping_blocks(sentences)
    print(f"\nHillock [INGESTOR]: Chunked '{os.path.basename(file_path)}' ({len(sentences)} sentences) into {len(blocks)} blocks.")

    # Resolves parallel-framing contradiction by checking MAX_WORKERS setting
    if MAX_WORKERS > 1:
        print(f"Hillock [INGESTOR]: Spawning {MAX_WORKERS} parallel workers on your CPU cores...")
    else:
        print(f"Hillock [INGESTOR]: Running sequential ingestion (MAX_WORKERS = 1) to optimize local model caching...")

    extracted_relations = []
    completed_blocks = 0

    # Execute parallel LLM requests using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_relations_from_block, block, hillock): (idx, block) for idx, block in enumerate(blocks)}

        for future in as_completed(futures):
            idx, block_text = futures[future]
            completed_blocks += 1
            try:
                triples = future.result()
                valid_block_extractions = []
                for triple in triples:
                    sub = hillock.resolve_entity_identity(triple.get("subject", ""))
                    pred = triple.get("predicate", "").strip()
                    obj = hillock.resolve_entity_identity(triple.get("object", ""))

                    if not sub or not pred or not obj:
                        continue

                    # Apply predicate normalization map
                    norm_pred = hillock.predicate_map.get(pred.strip().lower(), pred.strip().lower().replace(" ", "_"))

                    # Store relation and seed HDC space
                    hillock.kg.update_relation(sub, norm_pred, obj)
                    hillock.plasticity.update_associations({sub, obj})
                    hillock.hdc.get_or_allocate_hypervector(sub)
                    hillock.hdc.get_or_allocate_hypervector(obj)

                    valid_block_extractions.append(f"[{sub}] -[{norm_pred}]-> [{obj}]")
                    extracted_relations.append((sub, norm_pred, obj))

                # Fetch millisecond-level timestamp
                t_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

                # Fetch system hardware specs
                specs_str = ""
                if psutil:
                    cpu_p = psutil.cpu_percent(interval=None)
                    ram_p = psutil.virtual_memory().percent
                    specs_str = f" | CPU: {cpu_p:.1f}%, RAM: {ram_p:.1f}%"

                # Real-time console update per block
                if valid_block_extractions:
                    print(f"{t_stamp} [INFO]  * [Block {completed_blocks}/{len(blocks)}{specs_str}]: Extracted: {', '.join(valid_block_extractions)}")
                else:
                    print(f"{t_stamp} [INFO]  * [Block {completed_blocks}/{len(blocks)}{specs_str}]: No relations found.")

            except Exception as e:
                logger.error(f"Error processing parallel block {idx}: {e}")

    elapsed_time = time.perf_counter() - start_time
    ingestion_rate = len(sentences) / elapsed_time if elapsed_time > 0 else 0.0

    # Generate structured summary report
    summary = (
        f"\n"
        f"========================================================\n"
        f"              BULK INGESTION SUMMARY REPORT             \n"
        f"========================================================\n"
        f"  * File Processed      : {os.path.basename(file_path)}\n"
        f"  * Total Sentences     : {len(sentences)}\n"
        f"  * Paragraph Blocks    : {len(blocks)}\n"
        f"  * Extracted Relations : {len(extracted_relations)}\n"
        f"  * Processing Time     : {elapsed_time:.2f} seconds\n"
        f"  * Ingestion Rate      : {ingestion_rate:.1f} sentences/sec\n"
        f"========================================================"
    )
    return summary