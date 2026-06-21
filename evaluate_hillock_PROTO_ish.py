"""
Hillock Scientific Evaluation Harness (Long-Form Research Edition)
Performs automated database seeding, sequential query testing,
and outputs performance metrics: Precision, Recall, Retrieval Accuracy, and Gate Accuracy.
"""

import os
import json
import logging
import sqlite3
import re
import numpy as np
from main import IntegratedHillock

# Set up logging to file instead of stdout to keep output clean
logging.basicConfig(level=logging.ERROR)

def generate_test_assets():
    """Auto-generates a long, complex, and highly rigorous test dataset to evaluate Hillock."""

    # 30 complex, multi-subject sentences with distractors, negatives, and diverse tenses
    facts = (
        "Marie Curie was a brilliant physicist who spent her early childhood years in Warsaw, Poland before migrating to France.\n"
        "During her illustrious career, she discovered radioactivity and collaborated closely with Albert Einstein, who was born in Germany.\n"
        "Einstein, famous for his theoretical work, also worked alongside the British computer scientist Alan Turing.\n"
        "Turing was born in London and is famous for his work at Bletchley Park where he cracked the Enigma code.\n"
        "Isaac Newton, a contemporary from another era, was born in Woolsthorpe, England, but did not work with Einstein.\n"
        "Ada Lovelace, born in London, wrote the first algorithm for Charles Babbage's mechanical computer.\n"
        "Babbage designed the Analytical Engine while living in England, though it was never fully built.\n"
        "Nikola Tesla, born in Smiljan, Croatia, migrated to America and pioneered alternating current electricity.\n"
        "Tesla worked briefly for Thomas Edison, who patented the light bulb at Menlo Park, New Jersey.\n"
        "Grace Hopper was born in New York and became a pioneer who developed the first compiler.\n"
        "Galileo Galilei, born in Pisa, Italy, used early telescopes to discover the moons of Jupiter.\n"
        "Richard Feynman, a Nobel Prize winner born in New York, formulated the path integral formulation of quantum mechanics.\n"
        "Feynman collaborated closely with Murray Gell-Mann, the physicist who proposed the quark model.\n"
        "Gell-Mann proposed quarks while working at the California Institute of Technology.\n"
        "Erwin Schrodinger was born in Vienna, Austria, and formulated the wave equation that describes wave functions.\n"
        "Werner Heisenberg was born in Wurzburg, Germany, and established the uncertainty principle of quantum mechanics.\n"
        "John von Neumann was born in Budapest, Hungary, and described the self-replicating automata architecture.\n"
        "Claude Shannon was born in Michigan and founded information theory by mapping Boolean algebra to circuits.\n"
        "Johannes Kepler was born in Weil der Stadt, Germany, and derived the laws of planetary motion.\n"
        "Kepler analyzed astronomical data collected by Tycho Brahe, who was born in Denmark.\n"
        "Katherine Johnson was born in West Virginia and calculated the orbital trajectories for the Apollo 11 moon mission.\n"
        "Johnson worked at NASA alongside Dorothy Vaughan, who managed the West Area Computers unit.\n"
        "Rosalind Franklin was born in London and captured the X-ray diffraction images of DNA.\n"
        "Franklin worked in London, but her critical data was shared with James Watson, who was born in Chicago.\n"
        "Watson, along with Francis Crick, proposed the double-helix model of DNA at Cambridge.\n"
        "Gregor Mendel was born in Heinzendorf, Austria, and discovered the fundamental laws of genetic inheritance.\n"
        "Guglielmo Marconi was born in Bologna, Italy, and developed the first practical radio wave communication system.\n"
        "Marconi built upon the electromagnetic wave discoveries of Heinrich Hertz, who was born in Hamburg, Germany.\n"
        "Aristotle was born in Stagira, Greece, and founded the formal system of syllogistic logic.\n"
        "Aristotle studied in Athens under Plato, who established the famous Academy.\n"
        "Bertrand Russell was born in Trellech, Wales, and wrote the Principia Mathematica to ground mathematics in formal logic.\n"
        "Russell collaborated closely on mathematical logic with Alfred North Whitehead, who was born in Ramsgate, England.\n"
    )

    questions = [
        # --- ANSWERABLE QUESTIONS (20 total) ---
        {"question": "Where was Marie Curie born?", "expected_subject": "Marie_Curie", "expected_predicate": "born_in", "expected_object": "Poland", "answerable": True},
        {"question": "Where was Alan Turing born?", "expected_subject": "Alan_Turing", "expected_predicate": "born_in", "expected_object": "London", "answerable": True},
        {"question": "Where was Albert Einstein born?", "expected_subject": "Albert_Einstein", "expected_predicate": "born_in", "expected_object": "Germany", "answerable": True},
        {"question": "Who did Turing work with?", "expected_subject": "Alan_Turing", "expected_predicate": "collaborated_with", "expected_object": "Albert_Einstein", "answerable": True},
        {"question": "What did Alan Turing crack?", "expected_subject": "Alan_Turing", "expected_predicate": "cracked", "expected_object": "Enigma", "answerable": True},
        {"question": "What did Marie Curie discover?", "expected_subject": "Marie_Curie", "expected_predicate": "discovered", "expected_object": "Radioactivity", "answerable": True},
        {"question": "Where was Ada Lovelace born?", "expected_subject": "Ada_Lovelace", "expected_predicate": "born_in", "expected_object": "London", "answerable": True},
        {"question": "Who designed the Analytical Engine?", "expected_subject": "Charles_Babbage", "expected_predicate": "designed", "expected_object": "Analytical_Engine", "answerable": True},
        {"question": "Where was Nikola Tesla born?", "expected_subject": "Nikola_Tesla", "expected_predicate": "born_in", "expected_object": "Croatia", "answerable": True},
        {"question": "What did Grace Hopper develop?", "expected_subject": "Grace_Hopper", "expected_predicate": "developed", "expected_object": "compiler", "answerable": True},
        {"question": "Where was Galileo Galilei born?", "expected_subject": "Galileo_Galilei", "expected_predicate": "born_in", "expected_object": "Pisa", "answerable": True},
        {"question": "Where was Richard Feynman born?", "expected_subject": "Richard_Feynman", "expected_predicate": "born_in", "expected_object": "New_York", "answerable": True},
        {"question": "Where was Erwin Schrodinger born?", "expected_subject": "Erwin_Schrodinger", "expected_predicate": "born_in", "expected_object": "Austria", "answerable": True},
        {"question": "Where was John von Neumann born?", "expected_subject": "John_von_Neumann", "expected_predicate": "born_in", "expected_object": "Hungary", "answerable": True},
        {"question": "Where was Rosalind Franklin born?", "expected_subject": "Rosalind_Franklin", "expected_predicate": "born_in", "expected_object": "London", "answerable": True},
        {"question": "Where was Gregor Mendel born?", "expected_subject": "Gregor_Mendel", "expected_predicate": "born_in", "expected_object": "Austria", "answerable": True},
        {"question": "Where was Bertrand Russell born?", "expected_subject": "Bertrand_Russell", "expected_predicate": "born_in", "expected_object": "Wales", "answerable": True},
        {"question": "Who did Bertrand Russell collaborate with?", "expected_subject": "Bertrand_Russell", "expected_predicate": "collaborated_with", "expected_object": "Alfred_North_Whitehead", "answerable": True},
        {"question": "Where was Guglielmo Marconi born?", "expected_subject": "Guglielmo_Marconi", "expected_predicate": "born_in", "expected_object": "Italy", "answerable": True},
        {"question": "Who did Aristotle study under?", "expected_subject": "Aristotle", "expected_predicate": "collaborated_with", "expected_object": "Plato", "answerable": True},

        # --- UNANSWERABLE QUESTIONS / HARD NEGATIVES (10 total) ---
        {"question": "Who is Turing?", "answerable": False},
        {"question": "Did Nikola Tesla work with Isaac Newton?", "answerable": False}, # Newton is 1600s, Tesla is 1800s
        {"question": "What did Albert Einstein discover?", "answerable": False},      # Curie discovered radioactivity, Einstein didn't!
        {"question": "Where was Thomas Edison born?", "answerable": False},           # Menlo Park is active work, birth not mentioned
        {"question": "Who cracked Enigma?", "answerable": False},                     # Enigma is target object, testing link-routing direction
        {"question": "What did Richard Feynman discover?", "answerable": False},      # Text says made contributions, did not discover
        {"question": "What did Claude Shannon patent?", "answerable": False},         # Founded information theory, patents not mentioned
        {"question": "Where was Plato born?", "answerable": False},                   # Studied in Athens under him, but his birthplace is omitted
        {"question": "Who did Heinrich Hertz collaborate with?", "answerable": False}, # Marconi built on his discoveries, but no active collaboration
        {"question": "What did Johannes Kepler discover in Italy?", "answerable": False} # Kepler was in Germany, Galileo was in Italy
    ]

    # Force write fresh copies of the expanded files (safely in root directory)
    with open("eval_facts.txt", "w", encoding="utf-8") as f:
        f.write(facts)

    with open("eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

def run_evaluation():
    db_file = "hillock_eval.db"

    # 1. Clean up old evaluation DB physically and logically to guarantee pristine starting environment
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            print("[RESET] Cleaned old evaluation database file from disk.")
        except PermissionError:
            print("Error: Close all active SQL connection tools before running evaluation.")
            return

    print("========================================================")
    print("        HILLOCK SCIENTIFIC BENCHMARKING PIPELINE        ")
    print("========================================================")

    # Initialize fresh Hillock instance (this seeds default knowledge)
    hillock = IntegratedHillock(db_file)
    hillock.kg.clear_and_reinitialize() # Extra safe double-wipe and seed re-run

    # 2. RUN AUTONOMOUS INGESTION
    print("\n[Step 1/3]: Ingesting facts from 'eval_facts.txt'...")
    from ingestor import ingest_document_parallel
    ingest_result = ingest_document_parallel("eval_facts.txt", hillock)
    print(f" -> {ingest_result}")

    # 3. EVALUATE EXTRACTION (Precision & Recall)
    # Target relations in our upgraded long-form eval_facts.txt (Total: 22 targets)
    target_facts = {
        ("Marie_Curie", "born_in", "Poland"),
        ("Alan_Turing", "born_in", "London"),
        ("Albert_Einstein", "born_in", "Germany"),
        ("Alan_Turing", "collaborated_with", "Albert_Einstein"),
        ("Albert_Einstein", "collaborated_with", "Marie_Curie"),
        ("Marie_Curie", "discovered", "Radioactivity"),
        ("Alan_Turing", "cracked", "Enigma"),
        ("Ada_Lovelace", "born_in", "London"),
        ("Charles_Babbage", "designed", "Analytical_Engine"),
        ("Nikola_Tesla", "born_in", "Croatia"),
        ("Grace_Hopper", "born_in", "New_York"),
        ("Grace_Hopper", "developed", "compiler"),
        ("Galileo_Galilei", "born_in", "Pisa"),
        ("Galileo_Galilei", "discovered", "moons_of_Jupiter"),
        ("Richard_Feynman", "born_in", "New_York"),
        ("Erwin_Schrodinger", "born_in", "Austria"),
        ("Werner_Heisenberg", "born_in", "Germany"),
        ("John_von_Neumann", "born_in", "Hungary"),
        ("Bertrand_Russell", "born_in", "Wales"),
        ("Bertrand_Russell", "collaborated_with", "Alfred_North_Whitehead"),
        ("Guglielmo_Marconi", "born_in", "Italy"),
        ("Aristotle", "collaborated_with", "Plato")
    }

    # Fetch all active relations from SQLite
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT source_id, predicate, target_id FROM relations")
        extracted_facts = set(cursor.fetchall())

    correct_extractions = extracted_facts.intersection(target_facts)
    total_extracted = len(extracted_facts)
    total_targets = len(target_facts)

    precision = (len(correct_extractions) / total_extracted) if total_extracted > 0 else 0.0
    recall = (len(correct_extractions) / total_targets) if total_targets > 0 else 0.0

    # 4. RUN RETRIEVAL BENCHMARK
    print("\n[Step 2/3]: Querying the Gated Retriever...")
    with open("eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    correct_answers = 0
    correct_blocks = 0
    incorrect_blocks = 0
    incorrect_leaks = 0

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Question Asked':<40} | {'Expected':<8} | {'Mode':<10} | {'Status'}")
    print("--------------------------------------------------------------------------------")

    for q_data in questions:
        q = q_data["question"]
        is_answerable = q_data["answerable"]

        # Track linked entities for debug log
        active_ents = hillock.link_entities(q)

        # Execute turn
        reply, primed_info, hdc_fingerprint, mode = hillock.execute_chat_turn(q)

        status = "UNKNOWN"
        # Evaluate Gate Accuracy & Retrieval
        if is_answerable:
            if mode in ["RENDER_SUCCESS", "RENDER_FALLBACK"]:
                # Gate correctly opened. Now check if the resolved fact is correct.
                candidate_facts = hillock.kg.get_all_facts_for_entities(active_ents)

                # Uses select_answering_facts list extraction
                matched_list = hillock.select_answering_facts(q, candidate_facts)

                if matched_list:
                    # Validate against the top-matching fact
                    matched = matched_list[0]
                    if matched[0] == q_data["expected_subject"] and matched[1] == q_data["expected_predicate"] and matched[2] == q_data["expected_object"]:
                        status = "CORRECT"
                        correct_answers += 1
                    else:
                        status = "WRONG_FACT"
                else:
                    status = "FALSE_BLOCK"
                    incorrect_blocks += 1
            else:
                status = "FALSE_BLOCK"
                incorrect_blocks += 1
        else:
            if mode == "DETERMINISTIC_GATED_FALLBACK":
                status = "CORRECT_BLOCK"
                correct_blocks += 1
            else:
                status = "HALLUCINATION_LEAK"
                incorrect_leaks += 1

        expected_str = "Answer" if is_answerable else "Block"
        actual_str = "Answered" if mode == "RENDER_SUCCESS" else "Blocked"
        print(f"{q:<40} | {expected_str:<8} | {actual_str:<10} | {status}")

        # NERD LEVEL DIAGNOSTIC DEBUG INFO LOG (Only prints if entities were linked)
        if active_ents:
            print(f"  [DEBUG INFO FOR QUERY: '{q}']")
            print(f"    * Linked Entities: {list(active_ents)}")

            # Print Hebbian weights trace
            if primed_info:
                print(f"    * Synaptic Association Weights (Hebbian Engine):")
                for node, weight in primed_info[:2]:
                    print(f"        - Path: [{list(active_ents)[0]} -> {node}] Strength: {weight:.4f}")

            # Print HDC Context Fingerprint trace
            if hdc_fingerprint:
                print(f"    * Context Fingerprint (Top HDC Reservoir Traces):")
                for node, sim in hdc_fingerprint[:2]:
                    print(f"        - Active Echo: '{node:<15}' Cosine Similarity: {sim:.4f}")

            # Print the Similarity Gating Table
            candidate_facts = hillock.kg.get_all_facts_for_entities(active_ents)
            if candidate_facts:
                print(f"    * HDC Similarity Gate Evaluation Table:")
                for s, p, o in candidate_facts:
                    # Manually evaluate similarity outside of core engine for trace printing
                    # Builds fact components matching select_answering_facts
                    s_resolved = hillock.resolve_entity_identity(s)
                    o_resolved = hillock.resolve_entity_identity(o)
                    pred_keywords = [p.lower().replace("_", " ")]
                    if p in ["collaborated_with", "work"]:
                        pred_keywords.extend(["work", "worked", "with", "partner", "collaborated"])
                    elif p in ["born_in", "bear"]:
                        pred_keywords.extend(["born", "in", "birth"])
                    elif p in ["discovered", "discover"]:
                        pred_keywords.extend(["discover", "discovered", "found"])
                    elif p in ["cracked", "crack"]:
                        pred_keywords.extend(["crack", "cracked", "broke"])

                    best_pred_word = p.lower()
                    for kw in pred_keywords:
                        if kw in set(re.sub(r"[^\w\s]", "", q).lower().split()):
                            best_pred_word = kw
                            break

                    components = [s_resolved, o_resolved, best_pred_word]
                    fact_hv = np.zeros(hillock.hdc.d, dtype=np.int32)
                    for comp in set(components):
                        resolved_comp = hillock.resolve_entity_identity(comp)
                        if resolved_comp in hillock.hdc.codebook:
                            fact_hv += hillock.hdc.get_or_allocate_hypervector(resolved_comp, is_vocab_token=False)
                        else:
                            fact_hv += hillock.hdc.get_or_allocate_hypervector(comp, is_vocab_token=True)

                    query_tokens = set(re.sub(r"[^\w\s]", "", q).lower().split())
                    query_hv = np.zeros(hillock.hdc.d, dtype=np.int32)
                    for token in query_tokens:
                        resolved = hillock.resolve_entity_identity(token)
                        if resolved in hillock.hdc.codebook:
                            query_hv += hillock.hdc.get_or_allocate_hypervector(resolved, is_vocab_token=False)
                        else:
                            query_hv += hillock.hdc.get_or_allocate_hypervector(token, is_vocab_token=True)

                    q_norm = np.linalg.norm(query_hv)
                    f_norm = np.linalg.norm(fact_hv)
                    sim = np.dot(query_hv, fact_hv) / (q_norm * f_norm) if (q_norm > 0 and f_norm > 0) else 0.0

                    gate_status = "PASSED (GATE OPEN)" if sim >= 0.40 else "BLOCKED (GATE CLOSED)"
                    print(f"        - Fact: [{s} {p} {o}] Cosine Sim: {sim:.4f} -> {gate_status}")
            print("-" * 80)

    print("--------------------------------------------------------------------------------")

    # 5. CALCULATE STATISTICAL SCORES
    retrieval_acc = correct_answers / len([q for q in questions if q["answerable"]])
    gate_acc = (correct_blocks + correct_answers) / len(questions)

    print("\n[Step 3/3]: Generating Research Performance Metrics:")
    print("--------------------------------------------------")
    print(f"  * Extraction Precision : {precision*100:.1f}%  (Correctly structured factual nodes)")
    print(f"  * Extraction Recall    : {recall*100:.1f}%  (Completeness of indexed relations)")
    print(f"  * Retrieval Accuracy   : {retrieval_acc*100:.1f}%  (Factual accuracy on answerable queries)")
    print(f"  * Gate Accuracy        : {gate_acc*100:.1f}%  (Hallucination defense rate)")
    print("--------------------------------------------------")

    # Clean up evaluation DB file cleanly
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except PermissionError:
            pass

if __name__ == "__main__":
    generate_test_assets()
    run_evaluation()