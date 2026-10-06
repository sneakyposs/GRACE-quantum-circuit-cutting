"""
batch_worker.py
---------------
Run ONE circuit through the full GRACE LangGraph pipeline in an isolated
process. batch_test.py launches this as a subprocess so that a crash,
hang, or segfault inside qiskit / LangGraph on one circuit can never
take down the whole batch.

Prints exactly one machine-readable JSON line to stdout, delimited by
@@RESULT@@ markers, containing everything the orchestrator needs
(strategy, in-graph validation result, exported run_dir, timing, error).

Not meant to be invoked by hand, but it works if you want to:
    python batch_worker.py path/to/circuit.qasm --export-dir some_dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

RESULT_MARKER = "@@RESULT@@"


def emit(payload: dict) -> None:
    print(f"\n{RESULT_MARKER}{json.dumps(payload)}{RESULT_MARKER}", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("qasm_path")
    p.add_argument("--export-dir", required=True,
                   help="Directory the run folder is exported into.")
    args = p.parse_args(argv)

    t0 = time.monotonic()
    result: dict = {
        "qasm_path": args.qasm_path,
        "strategy": None,
        "in_graph_validation_passed": None,
        "equivalence_passed": None,
        "equivalence_detail": None,
        "equivalence_seconds": None,
        "equivalence_skipped": None,
        "validation_attempts": None,
        "failed_strategies": None,
        "pipeline_error": None,
        "run_dir": None,
        "num_qubits": None,
        "depth": None,
        "pipeline_seconds": None,
    }

    try:
        # Import inside try: an import-time failure should still produce
        # a parseable result record rather than a bare traceback.
        from subgraph import build_subgraph
        from inspection_export import export_cutting_results

        app = build_subgraph()

        initial_state = {
            "qasm_path": args.qasm_path,
            "circuit": None, "circuit_summary": None,
            "analysis": None, "analysis_summary": None,
            "cutting_strategy": None, "strategy_reasoning": None,
            "selector_note": None, "decision_rationale": None,
            "gate_ids": None, "auto_partition": False,
            "partition_labels": None, "cut_result": None,
            "original_circuit": None, "cut_circuit": None,
            "qpd_bases": None, "gate_info": None,
            "total_sampling_overhead": None, "subcircuits": None,
            "metadata": None,
            "validation_passed": None, "validation_reason": None,
            "validation_attempts": 0,
            "equivalence_passed": None, "equivalence_detail": None,
            "equivalence_seconds": None, "equivalence_skipped": None,
            "cut_locations": None, "cut_info": None,
            "marked_circuit": None, "minimum_reached": None,
            "failed_strategies": [],
            "error": None,
        }

        final = app.invoke(initial_state)

        result["strategy"] = final.get("cutting_strategy")
        result["in_graph_validation_passed"] = final.get("validation_passed")
        # strategy's equivalence validator itself now); the harness
        # consumes these instead of re-running the validator offline.
        result["equivalence_passed"] = final.get("equivalence_passed")
        result["equivalence_detail"] = final.get("equivalence_detail")
        result["equivalence_seconds"] = final.get("equivalence_seconds")
        result["equivalence_skipped"] = final.get("equivalence_skipped")
        result["validation_attempts"] = final.get("validation_attempts")
        result["failed_strategies"] = final.get("failed_strategies")
        result["pipeline_error"] = final.get("error")

        circ = final.get("original_circuit") or final.get("circuit")
        if circ is not None:
            try:
                result["num_qubits"] = circ.num_qubits
                result["depth"] = circ.depth()
            except Exception:
                pass

        # Export artifacts so the offline equivalence validator can run.
        # export_cutting_results returns the created run folder.
        run_dir = export_cutting_results(final, output_dir=args.export_dir)
        result["run_dir"] = str(run_dir)

    except Exception:
        result["pipeline_error"] = traceback.format_exc()

    result["pipeline_seconds"] = round(time.monotonic() - t0, 3)
    emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
