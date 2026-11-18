"""The main execution orchestrator for the conversational chat console."""

import os
import re
import sys
import platform
import multiprocessing
import subprocess
import numpy as np
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from config import DB_FILE, HILLOCK_VERSION
from ingestor import ingest_document_parallel
from engine import IntegratedHillock

def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8"
        )
        return out.strip()
    except Exception:
        return "Non-NVIDIA GPU / CPU Execution Mode"

def print_system_dashboard(hillock: IntegratedHillock) -> None:
    gpu = get_gpu_name()
    cores = multiprocessing.cpu_count()
    os_name = f"{platform.system()} {platform.release()}"
    python_ver = platform.python_version()

    entities = hillock.kg.get_entity_count()
    relations = hillock.kg.get_relations_count()
    synapses = hillock.kg.get_synapse_count()
    snippets = hillock.kg.get_snippet_count()

    ram_str = "Active"
    if psutil:
        ram = psutil.virtual_memory()
        ram_str = f"{ram.used / (1024**3):.1f} GB / {ram.total / (1024**3):.1f} GB ({ram.percent}% Used)"

    print("\n" + "="*70)
    print(f"  ⛰️  HILLOCK v{HILLOCK_VERSION} | Local Neuro-Symbolic Engine")
    print("  💬 Join our Developer Discord: https://discord.gg/BGUPNBcVdp")
    print("  ⭐ Commercial License & Docs:  https://github.com/roandejager/Hillock")
    print("="*70)
    print(" [HARDWARE PROFILE]")
    print(f"  * OS Environment  : {os_name}")
    print(f"  * CPU Threads     : {cores} Logical Cores")
    print(f"  * GPU / Execution : {gpu}")
    print(f"  * System Memory   : {ram_str}")
    print(f"  * Python Host     : {python_ver}")
    print("-" * 70)
    print(" [PERSISTENT MEMORY GRAPH STATUS]")
    print(f"  * Database File   : {DB_FILE} ({'Active' if os.path.exists(DB_FILE) else 'Initializing'})")
    print(f"  * Unique Entities : {entities} registered nodes")
    print(f"  * Fact Triples    : {relations} stored relations")
    print(f"  * Synapses        : {synapses} active Hebbian connections")
    print(f"  * Snippets        : {snippets} stored text contexts")
    print(f"  * Active LLM      : {hillock.ollama_model}")
    print(f"  * Personality Mode: [{hillock.verbosity_mode}]")
    print(f"  * Debug Verbosity : [{hillock.debug_level}]")
    print("-" * 70)
    print(" [BUILT-IN COMMAND REFERENCE]")
    print("  * /ingest [file]                 : Index TXT/PDF files locally via TALON")
    print("  * /alias [target] [canonical]    : Merge fractured entities into one root node")
    print("  * /remember [note]               : Save a personal fact/note directly to memory")
    print("  * /mode [free/strict]            : Switch between FREE assistant and STRICT gate")
    print("  * /model [model_name]            : List local models or switch LLM on the fly")
    print("  * /inspect [entity]              : View stored triples & Hebbian weights")
    print("  * /status                        : Display live hardware & memory status")
    print("  * /debug [off/low/full]          : Change background log verbosity")
    print("  * /reset                         : Clear and re-seed database & HDC space")
    print("  * /help                          : Display this command reference")
    print("  * exit / quit                    : Safely terminate session")
    print("=" * 70 + "\n")

def handle_cli_input(
    user_input: str,
    hillock: IntegratedHillock,
    history: Optional[List[Dict[str, str]]] = None,
) -> bool:    

    """Processes one CLI line. Returns False when the user exits, True otherwise."""
    user_input = user_input.strip()
    if not user_input:
        return True

    cmd_parts = user_input.split()
    cmd = cmd_parts[0].lower()

    if cmd in ["exit", "quit", "/exit", "/quit"]:
        print("Safely shutting down local hillock.")
        return False

    if cmd == "/mode":
        if len(cmd_parts) == 2:
            mode_name = cmd_parts[1].strip().upper()
            if mode_name in ["FREE", "STRICT"]:
                hillock.verbosity_mode = mode_name
                print(f"Hillock [SYSTEM]: Mode set to [{mode_name}] successfully.")
            else:
                print("Hillock [SYSTEM]: Error. Modes available: free, strict.")
        else:
            print("Hillock [SYSTEM]: Error. Format is: /mode [free/strict]")
        return True

    if cmd == "/remember":
        if len(cmd_parts) >= 2:
            note_text = user_input.split(maxsplit=1)[1].strip()
            saved = hillock.remember_note(note_text)
            print(f"Hillock [MEMORY SAVED]: Stored snippet {saved['snippet_id']} in '{saved['source_doc']}'.")
        else:
            print("Hillock [SYSTEM]: Error. Format is: /remember [fact or note to store]")
        return True

    if cmd == "/model":
        available_models = hillock.list_local_ollama_models()
        if len(cmd_parts) >= 2:
            target_model = user_input.split(maxsplit=1)[1].strip()
            hillock.ollama_model = target_model
            print(f"Hillock [SYSTEM]: Active Ollama model switched to [{target_model}].")
        else:
            print(f"\nHillock [SYSTEM]: Active Model: [{hillock.ollama_model}]")
            if available_models:
                print(" [Available Local Ollama Models on your PC]:")
                for m in available_models:
                    star = " (Active)" if m == hillock.ollama_model else ""
                    print(f"  * {m}{star}")
            else:
                print(" (Could not connect to Ollama API or no models pulled yet)")
            print(" Usage: /model [model_name] to switch models\n")
        return True

    if cmd == "/debug":
        if len(cmd_parts) == 2:
            target_lvl = cmd_parts[1].strip().upper()
            if target_lvl in ["OFF", "LOW", "FULL"]:
                hillock.debug_level = target_lvl
                print(f"Hillock [SYSTEM]: Debug logging verbosity set to [{target_lvl}].")
            else:
                print("Hillock [SYSTEM]: Error. Debug levels available: off, low, full.")
        else:
            print(f"\nHillock [SYSTEM]: Current Debug Level: [{hillock.debug_level}]")
            print(" Options:")
            print("  * /debug off  : Clean chat output only")
            print("  * /debug low  : Show basic memory priming traces")
            print("  * /debug full : Show complete HDC cosine match scores and full diagnostics\n")
        return True

    if cmd == "/inspect":
        if len(cmd_parts) >= 2:
            ent_query = user_input.split(maxsplit=1)[1].strip()
            resolved_id = hillock.resolve_entity_identity(ent_query)
            facts = hillock.kg.get_all_facts_for_entities({resolved_id})
            weights = hillock.plasticity.get_associated_priming_context(resolved_id)

            print(f"\n" + "="*65)
            print(f" [INSPECTING ENTITY]: {resolved_id}")
            print("="*65)
            print("  Stored SPO Facts in Knowledge Graph:")
            if facts:
                for s, p, o, doc in facts:
                    print(f"   * [{s}] -[{p}]-> [{o}] (Source: {doc})")
            else:
                print("   (No stored facts found)")

            print("\n  Hebbian Synaptic Associations:")
            if weights:
                for target, w in weights:
                    print(f"   * Associated Concept: '{target:<15}' Strength: {w:.4f}")
            else:
                print("   (No active synaptic weights)")
            print("="*65 + "\n")
        else:
            print("Hillock [SYSTEM]: Error. Format is: /inspect [entity]")
        return True

    if cmd in ["/status", "/help"]:
        print_system_dashboard(hillock)
        return True

    if cmd == "/reset":
        print("Hillock [SYSTEM]: Initiating deliberate database reset...")
        hillock.kg.clear_and_reinitialize()
        hillock.hdc.state = np.zeros(hillock.hdc.D, dtype=np.float64)
        hillock.hdc.codebook.clear()
        hillock.hdc.vocab_book.clear()
        for ent_id in hillock.kg.get_all_entity_ids():
            hillock.hdc.get_or_allocate_hypervector(ent_id)
        print("Hillock [SYSTEM]: Database reset, re-seeded, and GloVe HDC space re-allocated.")
        if history is not None:
            history.clear()
        return True

    if cmd == "/ingest":
        if len(cmd_parts) >= 2:
            filename = cmd_parts[1].strip()
            print(f"Hillock [SYSTEM]: Initiating bulk ingestion for '{filename}' via TALON Engine...")
            result, _ = ingest_document_parallel(filename, hillock)
            print(f"Hillock [SYSTEM]: {result}")

            # --- ACTIVE DISAMBIGUATION QUIZ ---
            ambiguous_facts = hillock.get_ambiguous_facts()
            if ambiguous_facts:
                print(f"\nHillock [ACTIVE LEARNING]: I found {len(ambiguous_facts)} ambiguous facts during ingestion.")
                for s, p, o, doc in ambiguous_facts:
                    print(f"  Fact: [{s}] -[{p}]-> [{o}] (Source: {doc})")
                    ans = input(f"  Who or what does the ambiguous pronoun refer to? (Type name, or press Enter to skip): ").strip()

                    if ans:
                        ambiguous_terms = {"he", "she", "it", "they", "this", "that", "who", "whom", "which", "his", "her"}
                        new_s = ans if s.lower() in ambiguous_terms else s
                        new_o = ans if o.lower() in ambiguous_terms else o

                        hillock.resolve_ambiguous_fact(s, p, o, new_s, new_o)
                        print(f"  -> Successfully updated to: [{new_s}] -[{p}]-> [{new_o}]\n")
                    else:
                        print("  -> Skipped.\n")
        else:
            print("Hillock [SYSTEM]: Error. Correct format is: /ingest [filename.ext]")
        return True

    if cmd == "/alias":
        if len(cmd_parts) >= 3:
            target = cmd_parts[1].strip()
            canonical = cmd_parts[2].strip()
            success, msg = hillock.merge_entities(target, canonical)
            print(f"Hillock [SYSTEM]: {msg}")
        else:
            print("Hillock [SYSTEM]: Error. Format is: /alias [target_name] [canonical_name]")
        return True

    reply, primed, fingerprint, mode = (
        hillock.execute_chat_turn(user_input, history=history)
        if history
        else hillock.execute_chat_turn(user_input)
    )
    print(reply)
    if hillock.last_receipt:
        print(f"  {hillock.format_receipt_line(hillock.last_receipt)}")

    if history is not None and hillock.verbosity_mode == "FREE" and mode != "GREETING":
        clean_reply = re.sub(r"^Hillock.*? > ", "", str(reply)).strip()
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": clean_reply})
        if len(history) > 12:
            del history[:-12]

    if hillock.debug_level in ["LOW", "FULL"]:
        if primed:
            print("  [Memory Priming Node Activations]:")
            for node, weight in primed[:3]:
                print(f"    * Associated Concept: '{node:<13}'  Synaptic Connection Strength: {weight:.4f}")

        if fingerprint and mode == "RENDER_SUCCESS":
            print("  [HDC Conversational Fingerprint Traces]:")
            for node, sim in fingerprint[:3]:
                print(f"    * Active Semantic Echo: '{node:<13}'  Vector Cosine Similarity: {sim:.4f}")
    print()
    return True


def run_cli():
    hillock = IntegratedHillock(DB_FILE)
    chat_history: List[Dict[str, str]] = []
    print_system_dashboard(hillock)

    while True:
        try:
            user_input = input("User > ").strip()
            if not handle_cli_input(user_input, hillock, history=chat_history):
                break
        except KeyboardInterrupt:
            print("\nSafely shutting down local hillock.")
            break


if __name__ == "__main__":
    run_cli()