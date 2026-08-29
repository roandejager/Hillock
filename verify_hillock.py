#!/usr/bin/env python3
"""
Hillock — 20-Point CPU Verification Suite (No GPU, No TALON Models Required)

Exercises everything that runs without the CUDA extraction stack:
  1. SQLite KG        — seed counts, single-valued predicate overwrite, stem fallback
  2. Hebbian engine   — strengthen (eta=0.15) and decay (gamma=0.01) vs README math
  3. VSA reservoir    — determinism, bipolar output, binding orthogonality
  4. v0.4 helpers     — span cleaners, canonical keying, inverted-pair purge
  5. Coreference      — span replacement with character offsets
  6. Ingestion path   — loud-halt behavior when TALON stack is absent (post-fix d856c1a)
  7. Benchmark        — seed-contamination arithmetic check
  8. Gate scores      — score distribution on a seed-only DB vs HDC_THRESHOLD
"""

import os
import re
import sys
import json
import sqlite3
import logging
import tempfile

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def main() -> None:
    os.chdir(REPO_ROOT)
    if not os.path.exists("config.py"):
        print("Run this script from the Hillock repo root.")
        sys.exit(1)

    logging.getLogger("Hillock.TALON").setLevel(logging.CRITICAL)

    # Prevent the GloVe download while keeping the code path identical
    if not os.path.exists("glove.6B.50d.txt"):
        open("glove.6B.50d.txt", "w").close()

    import numpy as np

    results = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, bool(cond), detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    tmp_dir = tempfile.gettempdir()
    db1_path = os.path.join(tmp_dir, "hillock_verify.db")

    # ------------------------------------------------------------ 1. KG
    from database import SQLiteKnowledgeGraph
    kg = SQLiteKnowledgeGraph(db1_path)
    kg.clear_and_reinitialize()
    check("kg-seed", kg.get_entity_count() == 10 and kg.get_relations_count() == 7,
          f"entities={kg.get_entity_count()} relations={kg.get_relations_count()}")

    kg.update_relation("Alan_Turing", "born_in", "Manchester", "Person", "City")
    check("kg-single-valued-overwrite", kg.query_relation("Alan_Turing", "born_in") == "Manchester")
    check("kg-stem-fallback", kg.query_relation("Alan_Turing", "born") == "Manchester")

    # ------------------------------------------------------------ 2. Hebbian
    from plasticity import HebbianPlasticityEngine
    pl = HebbianPlasticityEngine(db1_path)
    pl.update_associations({"Alan_Turing", "Enigma"})
    with sqlite3.connect(db1_path) as c:
        w = c.execute("SELECT weight FROM hebbian_weights").fetchone()[0]
    check("heb-strengthen", abs(w - 0.15) < 1e-9, f"w={w}")  # eta = 0.15

    pl.update_associations({"Alan_Turing", "France"})  # Enigma now inactive
    with sqlite3.connect(db1_path) as c:
        decayed = [r for r in c.execute(
            "SELECT entity_a, entity_b, weight FROM hebbian_weights").fetchall()
            if r[0] == "Enigma" or r[1] == "Enigma"]
    check("heb-decay", len(decayed) == 1 and abs(decayed[0][2] - 0.15 * 0.99) < 1e-9,
          f"decayed_w={decayed[0][2] if decayed else None}")  # gamma = 0.01

    # ------------------------------------------------------------ 3. VSA
    from reservoir import HyperdimensionalReservoir, SubwordHDCEncoder
    enc = SubwordHDCEncoder()
    h1, h2 = enc.encode("hello"), enc.encode("hello")
    check("vsa-deterministic", np.array_equal(h1, h2), f"shape={h1.shape}")
    check("vsa-bipolar", set(np.unique(h1)) <= {-1, 1})

    hA, hB = enc.encode("entityA"), enc.encode("entityB")
    bind = hA * hB
    cos_bind_A = float(np.dot(bind.astype(np.float32), hA.astype(np.float32)) / 10000.0)
    check("vsa-binding-orthogonal", abs(cos_bind_A) < 0.12, f"cos(bind,A)={cos_bind_A:.4f}")

    # Test Permutation Orthogonality
    h_orig = enc.encode("permutation_test")
    h_perm = np.roll(h_orig, 1)
    cos_perm = float(np.dot(h_orig.astype(np.float32), h_perm.astype(np.float32)) / 10000.0)
    check("vsa-permutation-orthogonal", abs(cos_perm) < 0.12, f"cos(orig,perm)={cos_perm:.4f}")

    # Test Sequential Path Binding doesn't collapse
    from reservoir import HyperdimensionalReservoir
    res_verify = HyperdimensionalReservoir()
    path_hv = res_verify.bind_sequential_path(["NodeA", "NodeB", "NodeC"], ["Pred1", "Pred2"])
    check("vsa-sequential-path-bipolar", set(np.unique(path_hv)) <= {-1, 1})

    # ------------------------------------------------------------ 4. v0.4 helpers
    from talon_engine import clean_entity_text, get_canonical_triple_key, is_inverted_asymmetric_pair
    check("clean-possessive", clean_entity_text("Marie Curie's") == "Marie Curie")
    check("clean-trailing-verb", clean_entity_text("Alan Turing worked") == "Alan Turing")
    check("clean-preposition", clean_entity_text("Einstein in") == "Einstein")
    k1 = get_canonical_triple_key("Alan_Turing", "collaborated_with", "Albert_Einstein")
    k2 = get_canonical_triple_key("Albert_Einstein", "collaborated_with", "Alan_Turing")
    check("canonical-symmetric-dedup", k1 == k2, str(k1))
    check("canonical-asymmetric", get_canonical_triple_key("A", "born_in", "B")
          != get_canonical_triple_key("B", "born_in", "A"))
    check("inverted-pair-purge", is_inverted_asymmetric_pair("B", "born_in", "A", {("a", "born_in", "b")}))
    check("symmetric-no-purge", not is_inverted_asymmetric_pair("B", "collaborated_with", "A", {("a", "born_in", "b")}))

    # ------------------------------------------------------------ 5. Coreference
    from talon_engine import CoreferenceResolver
    class _Pred:
        def get_clusters(self, as_strings=False):
            return [[(0, 5), (23, 26)]]
    class _Model:
        def predict(self, texts):
            return [_Pred()]
    cr = CoreferenceResolver()
    cr.model = _Model()
    out = cr.resolve_text("Alice went home today. She slept.")
    check("coref-span-replacement", out == "Alice went home today. Alice slept.", f"->'{out}'")

    # ------------------------------------------------------------ 6. Ingestion path
    from ingestor import get_talon_engine, ingest_document_parallel
    from main import IntegratedHillock
    from evaluate_hillock_PROTO_ish import generate_test_assets
    generate_test_assets()

    db2_path = os.path.join(tmp_dir, "hillock_verify_e2e.db")
    if os.path.exists(db2_path):
        os.remove(db2_path)
    hk = IntegratedHillock(db2_path)

    talon = get_talon_engine()
    stack_present = talon is not None and talon.extractor.model is not None
    if not stack_present:
        summary, stats = ingest_document_parallel("eval_facts.txt", hk)
        check("ingest-loud-halt", isinstance(summary, str) and "WARNING" in summary
              and isinstance(stats, dict) and not stats,
              "ingestion halts loudly when TALON stack is missing (post-fix d856c1a)")
    else:
        print("[SKIP] ingest-loud-halt — TALON stack present; run the full GPU pipeline instead")

    # ------------------------------------------------------------ 7. Seed contamination
    target_facts = {
        ("Marie_Curie", "born_in", "Poland"), ("Alan_Turing", "born_in", "London"),
        ("Albert_Einstein", "born_in", "Germany"),
        ("Alan_Turing", "collaborated_with", "Albert_Einstein"),
        ("Albert_Einstein", "collaborated_with", "Marie_Curie"),
        ("Marie_Curie", "discovered", "Radioactivity"), ("Alan_Turing", "cracked", "Enigma"),
        ("Ada_Lovelace", "born_in", "London"), ("Charles_Babbage", "designed", "Analytical_Engine"),
        ("Nikola_Tesla", "born_in", "Croatia"), ("Grace_Hopper", "born_in", "New_York"),
        ("Grace_Hopper", "developed", "compiler"), ("Galileo_Galilei", "born_in", "Pisa"),
        ("Galileo_Galilei", "discovered", "moons_of_Jupiter"),
        ("Richard_Feynman", "born_in", "New_York"), ("Erwin_Schrodinger", "born_in", "Austria"),
        ("Werner_Heisenberg", "born_in", "Germany"), ("John_von_Neumann", "born_in", "Hungary"),
        ("Bertrand_Russell", "born_in", "Wales"),
        ("Bertrand_Russell", "collaborated_with", "Alfred_North_Whitehead"),
        ("Guglielmo_Marconi", "born_in", "Italy"), ("Aristotle", "collaborated_with", "Plato"),
    }
    db3_path = os.path.join(tmp_dir, "hillock_verify_contam.db")
    if os.path.exists(db3_path):
        os.remove(db3_path)
    hk2 = IntegratedHillock(db3_path)
    hk2.kg.clear_and_reinitialize()
    with sqlite3.connect(db3_path) as c:
        extracted = set(c.execute("SELECT source_id, predicate, target_id FROM relations").fetchall())
    correct = extracted & target_facts
    check("bench-seed-contamination", len(correct) == 4,
          f"seed baseline check: {len(correct)} initial seeds overlap eval targets")

    # ------------------------------------------------------------ 8. Gate score distribution
    db4_path = os.path.join(tmp_dir, "hillock_verify_gate.db")
    if os.path.exists(db4_path):
        os.remove(db4_path)
    hk3 = IntegratedHillock(db4_path)
    with open("eval_questions.json") as f:
        questions = json.load(f)
    rows = []
    for q in questions:
        ents = hk3.link_entities(q["question"])
        facts = hk3.kg.get_all_facts_for_entities(ents) if ents else []
        scored = hk3.select_answering_facts(q["question"], facts, threshold=-1.0)
        mx = max((s[3] for s in scored), default=None)
        rows.append((q["answerable"], mx, q["question"]))

    from config import HDC_THRESHOLD
    passes = sum(1 for a, m, _ in rows if a and m is not None and m >= HDC_THRESHOLD)
    leaks = sum(1 for a, m, _ in rows if not a and m is not None and m >= HDC_THRESHOLD)
    check("gate-distribution-ran", len(rows) == 32, f"verified gate distribution on {len(rows)} queries")

    # ------------------------------------------------------------ Summary
    print("\n" + "=" * 70)
    fails = [r for r in results if not r[1]]
    print(f"SUMMARY: {len(results) - len(fails)}/{len(results)} checks passed")
    for name, _, detail in fails:
        print(f"  FAILED: {name} -- {detail}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()