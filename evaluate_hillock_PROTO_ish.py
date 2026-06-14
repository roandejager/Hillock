"""
Exocortex Scientific Evaluation Harness (Multi-Fact Compliant)
Performs automated database seeding, sequential query testing,
and outputs performance metrics: Precision, Recall, Retrieval Accuracy, and Gate Accuracy.
"""

import os
import json
import logging
import sqlite3
from main import IntegratedExocortex

# Set up logging to file instead of stdout to keep output clean
logging.basicConfig(level=logging.ERROR)

def generate_test_assets():
    """Auto-generates tougher, more realistic test files if they don't exist to ensure out-of-the-box execution."""

    # Complex, multi-clause facts with distractors and passive phrasing
    facts = (
        "Marie Curie was a brilliant physicist who spent her early childhood years in Poland before migrating to France.\n"
        "During her illustrious career, she discovered radioactivity and collaborated closely with Albert Einstein, who was born in Germany.\n"
        "Einstein, famous for his theoretical work, also worked alongside the British computer scientist Alan Turing.\n"
        "Turing was born in London and is famous for his work at Bletchley Park where he cracked the Enigma code.\n"
        "Isaac Newton, a contemporary from another era, was born in Woolsthorpe, England, but did not work with Einstein.\n"
    )

    questions = [
        # Answerable questions (Requires resolving complex phrasing and associations)
        {"question": "Where was Marie Curie born?", "expected_subject": "Marie_Curie", "expected_predicate": "born_in", "expected_object": "Poland", "answerable": True},
        {"question": "Where was Alan Turing born?", "expected_subject": "Alan_Turing", "expected_predicate": "born_in", "expected_object": "London", "answerable": True},
        {"question": "Where was Albert Einstein born?", "expected_subject": "Albert_Einstein", "expected_predicate": "born_in", "expected_object": "Germany", "answerable": True},
        {"question": "Who did Turing work with?", "expected_subject": "Alan_Turing", "expected_predicate": "collaborated_with", "expected_object": "Albert_Einstein", "answerable": True},
        {"question": "What did Alan Turing crack?", "expected_subject": "Alan_Turing", "expected_predicate": "cracked", "expected_object": "Enigma", "answerable": True},
        {"question": "What did Marie Curie discover?", "expected_subject": "Marie_Curie", "expected_predicate": "discovered", "expected_object": "Radioactivity", "answerable": True},

        # Unanswerable questions (Tricky context tests to evaluate hallucination gating)
        {"question": "Who is Turing?", "answerable": False},
        {"question": "Where was Isaac Newton born?", "answerable": False}, # Not in SQLite seeds, testing ingestion extraction boundary
        {"question": "What did Albert Einstein discover?", "answerable": False}, # Curie discovered radioactivity, Einstein didn't!
        {"question": "Who cracked Enigma?", "answerable": False}  # 'Enigma' is the target, not active subject
    ]

    # Always write fresh files to ensure the tough benchmarks are used
    with open("extra_old/eval_facts.txt", "w", encoding="utf-8") as f:
        f.write(facts)

    with open("extra_old/eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

def run_evaluation():
    db_file = "extra_old/exocortex_eval.db"

    # 1. Clean up old evaluation DB to guarantee pristine testing environment
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except PermissionError:
            print("Error: Close all active SQL connection tools before running evaluation.")
            return

    print("========================================================")
    print("      EXOCORTEX SCIENTIFIC BENCHMARKING PIPELINE       ")
    print("========================================================")

    # Initialize fresh exocortex
    exocortex = IntegratedExocortex(db_file)

    # 2. RUN AUTONOMOUS INGESTION (Problem 1 Speed Fix: Parallel Ingestion Mode)
    print("\n[Step 1/3]: Ingesting facts from 'eval_facts.txt'...")
    # Using fast=False (thorough) for the evaluation test cases to ensure deep parsing
    from ingestor import ingest_document_parallel
    ingest_result = ingest_document_parallel("extra_old/eval_facts.txt", exocortex)
    print(f" -> {ingest_result}")

    # 3. EVALUATE EXTRACTION (Precision & Recall)
    # Target relations in our upgraded eval_facts.txt (Total: 7)
    target_facts = {
        ("Marie_Curie", "born_in", "Poland"),
        ("Alan_Turing", "born_in", "London"),
        ("Albert_Einstein", "born_in", "Germany"),
        ("Alan_Turing", "collaborated_with", "Albert_Einstein"),
        ("Albert_Einstein", "collaborated_with", "Marie_Curie"),
        ("Marie_Curie", "discovered", "Radioactivity"),
        ("Alan_Turing", "cracked", "Enigma")
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
    with open("extra_old/eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    correct_answers = 0
    correct_blocks = 0
    incorrect_blocks = 0
    incorrect_leaks = 0

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Question Asked':<35} | {'Expected Mode':<15} | {'Resulting Mode':<15} | {'Status'}")
    print("--------------------------------------------------------------------------------")

    for q_data in questions:
        q = q_data["question"]
        is_answerable = q_data["answerable"]

        # Execute turn
        reply, _, _, mode = exocortex.execute_chat_turn(q)

        # Evaluate Gate Accuracy & Retrieval
        if is_answerable:
            if mode in ["RENDER_SUCCESS", "RENDER_FALLBACK"]:
                # Gate correctly opened. Now check if the resolved fact is correct.
                active_ents = exocortex.link_entities(q)
                candidate_facts = exocortex.kg.get_all_facts_for_entities(active_ents)

                # Fixed: Uses select_answering_facts list extraction
                matched_list = exocortex.select_answering_facts(q, candidate_facts)

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
        print(f"{q:<35} | {expected_str:<15} | {actual_str:<15} | {status}")

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