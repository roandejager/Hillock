"""Runs parallelized paragraph block extractions to maximize CPU throughput [1]."""

import os
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set
from config import BLOCK_SIZE, BLOCK_OVERLAP, MAX_WORKERS

logger = logging.getLogger("Exocortex.Ingestor")

try:
    import pypdf
except ImportError:
    pypdf = None


def get_clean_sentences(file_path: str, exocortex) -> List[str]:
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

    # Segment document into sentences using basic sentence boundary punctuation
    sentences = [s.strip() for s in re.split(r"[.!?\n]", raw_text) if s.strip()]
    return sentences


def segment_into_overlapping_blocks(sentences: List[str], block_size: int = BLOCK_SIZE, overlap: int = BLOCK_OVERLAP) -> \
List[str]:
    """Chunks sentence arrays into overlapping paragraph blocks [1]."""
    blocks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i: i + block_size]
        block_text = " ".join(chunk)
        blocks.append(block_text)
        i += (block_size - overlap)
        if i >= len(sentences) or len(chunk) < block_size:
            break
    return blocks


def extract_relations_from_block(block_text: str, exocortex) -> List[Dict[str, str]]:
    """
    Passes a paragraph block to the local LLM to extract all relational triples
    concurrently, bypassing the sentence-by-sentence bottleneck [1, 23].
    """
    system_prompt = (
        "You are a precise, structural information extraction engine. Output ONLY a JSON array.\n"
        "Analyze the provided text block and extract all clear factual relationships between prominent entities.\n\n"
        "Rules:\n"
        "1. Output ONLY a valid JSON list of objects, do not include any other conversational filler.\n"
        "2. Fields required:\n"
        "   - 'subject': string (normalized snake_case entity name, e.g. 'Marie_Curie' or 'Alan_Turing')\n"
        "   - 'predicate': string (relationship verb, e.g. 'born_in', 'collaborated_with', 'discovered', 'cracked')\n"
        "   - 'object': string (normalized snake_case target entity, e.g. 'Germany')\n"
        "3. If no clear relationships exist, return an empty array [].\n"
    )

    response = exocortex.query_ollama(block_text, system_prompt)
    if not response:
        return []

    extracted_triples = exocortex.parse_json_safely(response)
    if isinstance(extracted_triples, list):
        return extracted_triples

    # Regex fallback if JSON parsing fails on small models
    triples = []
    try:
        # Scan for matching regex blocks
        matches = re.findall(
            r"\{\s*\"subject\"\s*:\s*\"([^\"]+)\"\s*,\s*\"predicate\"\s*:\s*\"([^\"]+)\"\s*,\s*\"object\"\s*:\s*\"([^\"]+)\"\s*\}",
            response)
        for sub, pred, obj in matches:
            triples.append({"subject": sub, "predicate": pred, "object": obj})
    except Exception:
        pass
    return triples


def ingest_document_parallel(file_path: str, exocortex) -> str:
    """Orchestrates high-speed, parallelized paragraph extraction across 4 threads [1]."""
    try:
        sentences = get_clean_sentences(file_path, exocortex)
    except Exception as e:
        return str(e)

    # Group sentences into paragraph blocks
    blocks = segment_into_overlapping_blocks(sentences)
    logger.info(f"Chunked document into {len(blocks)} overlapping paragraph blocks. Starting parallel extraction...")

    extracted_count = 0

    # Execute parallel LLM requests using ThreadPoolExecutor [1]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_relations_from_block, block, exocortex): block for block in blocks}

        for future in as_completed(futures):
            block_text = futures[future]
            try:
                triples = future.result()
                for triple in triples:
                    sub = exocortex.resolve_entity_identity(triple.get("subject", ""))
                    pred = triple.get("predicate", "").strip()
                    obj = exocortex.resolve_entity_identity(triple.get("object", ""))

                    if not sub or not pred or not obj:
                        continue

                    # Apply predicate normalization map
                    norm_pred = exocortex.predicate_map.get(pred.strip().lower(),
                                                            pred.strip().lower().replace(" ", "_"))

                    # Store relation and seed HDC space
                    exocortex.kg.update_relation(sub, norm_pred, obj)
                    exocortex.plasticity.update_associations({sub, obj})
                    exocortex.hdc.get_or_allocate_hypervector(sub)
                    exocortex.hdc.get_or_allocate_hypervector(obj)

                    extracted_count += 1
                    logger.info(f" -> [PARALLEL INDEXED]: [{sub}] -[{norm_pred}]-> [{obj}]")
            except Exception as e:
                logger.error(f"Error processing parallel block: {e}")

    return f"Autonomous Ingestion complete. Successfully indexed {extracted_count} relations from {len(blocks)} parallel paragraph blocks."