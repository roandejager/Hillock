#!/usr/bin/env python3
"""Reproducible, isolated multi-hop benchmark for Hillock v1.0.0."""

import argparse
import gc
import json
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

VERSION = "1.0.0"
SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "benchmark_results"
RESULTS_FILE = RESULTS_DIR / f"labyrinth_v{VERSION}.json"

import numpy as np

import config

config.GLOVE_PATH = str(SCRIPT_DIR / config.GLOVE_PATH)

from engine import IntegratedHillock


TIERS = (
    "Supplier",
    "Warehouse",
    "Distributor",
    "RetailStore",
    "Department",
    "Shelf",
    "Product",
    "BatchCode",
)
RELATIONS = (
    "supplies_to",
    "ships_to",
    "distributes_to",
    "stocks_in",
    "assigned_to",
    "holds_item",
    "has_batch",
)
CHAIN_LABELS = (
    "Northstar",
    "Cedar",
    "Summit",
    "Maple",
    "Harbor",
    "Pioneer",
    "Willow",
    "Canyon",
    "Redwood",
    "Bluebird",
)
NODE_LABELS = (
    "Supplier",
    "Warehouse",
    "Distributor",
    "RetailStore",
    "Department",
    "Shelf",
    "Product",
    "BatchCode",
)


def _entity_id(chain_index: int, tier_index: int) -> str:
    return f"{CHAIN_LABELS[chain_index]}{NODE_LABELS[tier_index]}{chain_index:02d}"


def _query_text(start_entity: str, hop_depth: int) -> str:
    target_tier = TIERS[hop_depth]
    path_relations = ", ".join(RELATIONS[:hop_depth])
    return (
        f"Which {target_tier} is connected to {start_entity} "
        f"through {path_relations}?"
    )


def generate_benchmark_data(
    rng: random.Random,
) -> Tuple[List[List[Tuple[str, str, str]]], List[Dict[str, Any]], Dict[str, int]]:
    """Build deterministic branching chains and 105 answerable/unanswerable queries."""
    triples_by_hop: List[List[Tuple[str, str, str]]] = [
        [] for _ in RELATIONS
    ]
    distractor_edge_count = 0
    positive_paths: List[List[str]] = []
    for chain_index in range(10):
        path = [_entity_id(chain_index, tier_index) for tier_index in range(8)]
        positive_paths.append(path)
        for hop_index, predicate in enumerate(RELATIONS):
            source = path[hop_index]
            canonical_target = path[hop_index + 1]
            triples_by_hop[hop_index].append(
                (source, predicate, canonical_target)
            )
            branch_count = rng.randint(2, 3)
            for distractor_index in range(1, branch_count):
                distractor = (
                    f"Distractor{CHAIN_LABELS[chain_index]}"
                    f"{NODE_LABELS[hop_index + 1]}{chain_index:02d}"
                    f"H{hop_index + 1}B{distractor_index}"
                )
                triples_by_hop[hop_index].append(
                    (source, predicate, distractor)
                )
                distractor_edge_count += 1

    queries: List[Dict[str, Any]] = []
    for hop_depth in range(1, len(TIERS)):
        for chain_index, path in enumerate(positive_paths):
            start_entity = path[0]
            queries.append(
                {
                    "query_id": f"hop_{hop_depth}_positive_{chain_index + 1:02d}",
                    "hop_depth": hop_depth,
                    "answerable": True,
                    "start_entity": start_entity,
                    "ground_truth_entity": path[hop_depth],
                    "predicates": list(RELATIONS[:hop_depth]),
                    "query": _query_text(start_entity, hop_depth),
                }
            )

        for negative_index in range(5):
            start_entity = f"UnlinkedSupplierH{hop_depth}N{negative_index:02d}"
            if hop_depth == 1:
                # Keep the query's source in Hillock's entity linker without
                # creating an edge in the requested supplies_to path.
                triples_by_hop[0].append(
                    (
                        start_entity,
                        "has_unrelated_record",
                        f"UnrelatedRecordH{hop_depth}N{negative_index:02d}",
                    )
                )
            else:
                cut_after = rng.randint(1, hop_depth - 1)
                for hop_index in range(cut_after):
                    predicate = RELATIONS[hop_index]
                    source = (
                        start_entity
                        if hop_index == 0
                        else f"BrokenPathH{hop_depth}N{negative_index:02d}T{hop_index}"
                    )
                    target = (
                        f"BrokenPathH{hop_depth}N{negative_index:02d}T{hop_index + 1}"
                    )
                    triples_by_hop[hop_index].append(
                        (source, predicate, target)
                    )
                    branch_count = rng.randint(2, 3)
                    for distractor_index in range(1, branch_count):
                        distractor = (
                            f"BrokenDistractorH{hop_depth}N{negative_index:02d}"
                            f"T{hop_index + 1}B{distractor_index}"
                        )
                        triples_by_hop[hop_index].append(
                            (source, predicate, distractor)
                        )
                        distractor_edge_count += 1

            queries.append(
                {
                    "query_id": f"hop_{hop_depth}_negative_{negative_index + 1:02d}",
                    "hop_depth": hop_depth,
                    "answerable": False,
                    "start_entity": start_entity,
                    "ground_truth_entity": None,
                    "disconnected_target": positive_paths[negative_index][hop_depth],
                    "predicates": list(RELATIONS[:hop_depth]),
                    "query": _query_text(start_entity, hop_depth),
                }
            )

    graph_counts = {
        "triples": sum(len(triples) for triples in triples_by_hop),
        "positive_chains": len(positive_paths),
        "distractor_edges": distractor_edge_count,
    }
    return triples_by_hop, queries, graph_counts


def _retrieve_exact_hops(
    hillock: IntegratedHillock,
    start_entity: str,
    predicates: Sequence[str],
) -> Set[str]:
    """Traverse directed SQLite relations for the exact requested predicate path."""
    frontier = {start_entity}
    for predicate in predicates:
        facts = hillock.kg.get_all_facts_for_entities(frontier)
        frontier = {
            target
            for source, fact_predicate, target, _source_doc in facts
            if source in frontier and fact_predicate == predicate
        }
        if not frontier:
            break
    return frontier


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _latency_summary(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _hardware_specs() -> Dict[str, Any]:
    cpu_model = platform.processor()
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as processor_key:
                cpu_model = winreg.QueryValueEx(
                    processor_key, "ProcessorNameString"
                )[0].strip()
        except OSError:
            pass
        cpu_model = cpu_model or os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpu_model = cpu_model or platform.machine() or "Unknown"
    return {
        "cpu_model": cpu_model,
        "thread_count": os.cpu_count(),
        "os": platform.platform(),
    }


def run_benchmark() -> Dict[str, Any]:
    rng = random.Random(SEED)
    triples_by_hop, queries, graph_counts = generate_benchmark_data(rng)
    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="hillock_labyrinth_") as temp_dir:
        database_path = Path(temp_dir) / "labyrinth_eval.db"
        hillock = IntegratedHillock(str(database_path))

        for hop_index, triples in enumerate(triples_by_hop):
            if hop_index == 0:
                ordinary_triples = [
                    triple
                    for triple in triples
                    if triple[1] == RELATIONS[hop_index]
                ]
                unrelated_triples = [
                    triple
                    for triple in triples
                    if triple[1] != RELATIONS[hop_index]
                ]
                if ordinary_triples:
                    hillock.kg.update_relations_batch(
                        ordinary_triples,
                        source_type=TIERS[hop_index],
                        target_type=TIERS[hop_index + 1],
                        source_doc="labyrinth_v1.0.0",
                    )
                if unrelated_triples:
                    hillock.kg.update_relations_batch(
                        unrelated_triples,
                        source_type=TIERS[0],
                        target_type="Generic",
                        source_doc="labyrinth_v1.0.0",
                    )
            elif triples:
                hillock.kg.update_relations_batch(
                    triples,
                    source_type=TIERS[hop_index],
                    target_type=TIERS[hop_index + 1],
                    source_doc="labyrinth_v1.0.0",
                )

        for entity_id in hillock.kg.get_all_entity_ids():
            hillock.hdc.get_or_allocate_hypervector(entity_id)

        suite_started_ns = time.perf_counter_ns()
        for query_data in queries:
            query_started_ns = time.perf_counter_ns()

            active_entities = hillock.link_entities(query_data["query"])
            if query_data["start_entity"] not in active_entities:
                raise RuntimeError(
                    f"Entity linker failed to activate query source "
                    f"{query_data['start_entity']!r} for {query_data['query_id']}."
                )
            candidate_facts = hillock.kg.get_all_facts_for_entities(active_entities)

            retrieval_started_ns = time.perf_counter_ns()
            retrieved_entities = _retrieve_exact_hops(
                hillock,
                query_data["start_entity"],
                query_data["predicates"],
            )
            retrieval_latency_ms = (
                time.perf_counter_ns() - retrieval_started_ns
            ) / 1_000_000

            gate_started_ns = time.perf_counter_ns()
            selected_facts = hillock.select_answering_facts(
                query_data["query"], candidate_facts
            )
            gate_latency_ms = (time.perf_counter_ns() - gate_started_ns) / 1_000_000
            total_latency_ms = (time.perf_counter_ns() - query_started_ns) / 1_000_000

            passed_gate = bool(selected_facts)
            exact_target_retrieved = (
                query_data["ground_truth_entity"] in retrieved_entities
                if query_data["answerable"]
                else False
            )
            if query_data["answerable"] and not exact_target_retrieved:
                raise RuntimeError(
                    f"Generated answerable path was not retrievable for "
                    f"{query_data['query_id']}."
                )
            if not query_data["answerable"] and query_data["disconnected_target"] in retrieved_entities:
                raise RuntimeError(
                    f"Generated hard negative became reachable for "
                    f"{query_data['query_id']}."
                )

            if query_data["answerable"]:
                classification = "SUCCESS" if passed_gate else "FALSE_BLOCK"
            else:
                classification = (
                    "HALLUCINATION_LEAK" if passed_gate else "CORRECT_REFUSAL"
                )

            results.append(
                {
                    **query_data,
                    "active_entities": sorted(active_entities),
                    "candidate_fact_count": len(candidate_facts),
                    "selected_fact_count": len(selected_facts),
                    "retrieved_entities": sorted(retrieved_entities),
                    "exact_target_retrieved": exact_target_retrieved,
                    "classification": classification,
                    "latency_ms": total_latency_ms,
                    "exact_retrieval_latency_ms": retrieval_latency_ms,
                    "gate_latency_ms": gate_latency_ms,
                }
            )
        suite_duration_ns = time.perf_counter_ns() - suite_started_ns
        del hillock
        gc.collect()

    metrics_by_hop: Dict[str, Any] = {}
    for hop_depth in range(1, len(TIERS)):
        hop_results = [
            result for result in results if result["hop_depth"] == hop_depth
        ]
        answerable = [result for result in hop_results if result["answerable"]]
        unanswerable = [result for result in hop_results if not result["answerable"]]
        counts = {
            category: sum(
                result["classification"] == category for result in hop_results
            )
            for category in (
                "SUCCESS",
                "FALSE_BLOCK",
                "CORRECT_REFUSAL",
                "HALLUCINATION_LEAK",
            )
        }
        metrics_by_hop[str(hop_depth)] = {
            "answerable_queries": len(answerable),
            "unanswerable_queries": len(unanswerable),
            "classification_counts": counts,
            "accuracy_pct": 100.0 * counts["SUCCESS"] / len(answerable),
            "overall_accuracy_pct": 100.0
            * (counts["SUCCESS"] + counts["CORRECT_REFUSAL"])
            / len(hop_results),
            "refusal_rate_pct": 100.0
            * counts["CORRECT_REFUSAL"]
            / len(unanswerable),
            "hallucination_rate_pct": 100.0
            * counts["HALLUCINATION_LEAK"]
            / len(unanswerable),
            "latency_ms": _latency_summary(
                [result["latency_ms"] for result in hop_results]
            ),
            "exact_retrieval_latency_ms": _latency_summary(
                [result["exact_retrieval_latency_ms"] for result in hop_results]
            ),
            "gate_latency_ms": _latency_summary(
                [result["gate_latency_ms"] for result in hop_results]
            ),
        }

    if len(results) != 105:
        raise RuntimeError(f"Expected 105 queries, generated {len(results)}.")

    return {
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "hardware": _hardware_specs(),
        "topology": {
            "tiers": list(TIERS),
            "relations": list(RELATIONS),
            "branching_factor": [2, 3],
            **graph_counts,
        },
        "query_count": len(results),
        "suite_duration_seconds": suite_duration_ns / 1_000_000_000,
        "total_queries_per_second": len(results)
        / (suite_duration_ns / 1_000_000_000),
        "metrics_by_hop": metrics_by_hop,
        "query_results": results,
    }


def save_results(data: Dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=RESULTS_DIR,
            prefix=".labyrinth_",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            json.dump(data, output_file, indent=2)
            output_file.write("\n")
        os.replace(temp_path, RESULTS_FILE)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return RESULTS_FILE


def print_dashboard(data: Dict[str, Any]) -> None:
    headers = (
        "Hop",
        "Accuracy",
        "Refusal",
        "Hallucination",
        "Mean ms",
        "Median ms",
        "P95 ms",
        "P99 ms",
        "S/FB/CR/HL",
    )
    rows = []
    for hop_depth in range(1, len(TIERS)):
        metrics = data["metrics_by_hop"][str(hop_depth)]
        counts = metrics["classification_counts"]
        rows.append(
            (
                str(hop_depth),
                f"{metrics['accuracy_pct']:.1f}%",
                f"{metrics['refusal_rate_pct']:.1f}%",
                f"{metrics['hallucination_rate_pct']:.1f}%",
                f"{metrics['latency_ms']['mean']:.3f}",
                f"{metrics['latency_ms']['median']:.3f}",
                f"{metrics['latency_ms']['p95']:.3f}",
                f"{metrics['latency_ms']['p99']:.3f}",
                "/".join(
                    str(counts[name])
                    for name in (
                        "SUCCESS",
                        "FALSE_BLOCK",
                        "CORRECT_REFUSAL",
                        "HALLUCINATION_LEAK",
                    )
                ),
            )
        )

    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nThe Labyrinth Benchmark - Hillock v1.0.0")
    print(separator)
    print("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(separator)
    for row in rows:
        print("| " + " | ".join(row[i].rjust(widths[i]) for i in range(len(row))) + " |")
    print(separator)
    print(
        "S/FB/CR/HL = SUCCESS / FALSE_BLOCK / CORRECT_REFUSAL / HALLUCINATION_LEAK"
    )
    print(
        f"Queries: {data['query_count']} | "
        f"QPS: {data['total_queries_per_second']:.2f} | "
        f"Results: {RESULTS_FILE}"
    )


def plot_comparisons() -> bool:
    result_files = sorted(RESULTS_DIR.glob("labyrinth_v*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No benchmark result files found in {RESULTS_DIR}."
        )
    datasets = []
    for result_file in result_files:
        with result_file.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
        datasets.append((result_file.stem.removeprefix("labyrinth_"), data))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Warning: matplotlib is not installed; install it to generate plots.",
            file=sys.stderr,
        )
        return False

    figure, (accuracy_axis, latency_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for version_label, data in datasets:
        hop_depths = list(range(1, len(TIERS)))
        metrics = data["metrics_by_hop"]
        accuracy_axis.plot(
            hop_depths,
            [metrics[str(hop)]["accuracy_pct"] for hop in hop_depths],
            marker="o",
            label=version_label,
        )
        latency_axis.plot(
            hop_depths,
            [metrics[str(hop)]["latency_ms"]["p95"] for hop in hop_depths],
            marker="o",
            label=version_label,
        )

    accuracy_axis.set_title("Accuracy vs. Hop Depth")
    accuracy_axis.set_xlabel("Hop depth")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_xticks(range(1, len(TIERS)))
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    latency_axis.set_title("P95 Latency vs. Hop Depth")
    latency_axis.set_xlabel("Hop depth")
    latency_axis.set_ylabel("P95 latency (ms)")
    latency_axis.set_xticks(range(1, len(TIERS)))
    latency_axis.grid(True, alpha=0.3)
    latency_axis.legend()

    figure.tight_layout()
    plot_path = RESULTS_DIR / "labyrinth_comparison.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"Comparison plot saved to {plot_path}")
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot saved benchmark result files without running a new benchmark.",
    )
    args = parser.parse_args(argv)

    if args.plot:
        plot_comparisons()
        return 0

    data = run_benchmark()
    save_results(data)
    print_dashboard(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
