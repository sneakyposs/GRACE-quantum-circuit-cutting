"""
subgraph.py
-----------
Wires Parse_circuit → Analyze_circuit into a LangGraph subgraph.
This is the foundation you'll extend with Strategy_router (Node 3)
and the cutting nodes (Nodes 4–7).
"""

from langgraph.graph import StateGraph, END

from parse_node   import AgentState, parse_circuit_node
from analyze_node import analyze_circuit_node

from typing import TypedDict, Optional, Any
from qiskit import QuantumCircuit

from strategy_router import (
    strategy_router_node,
    route_to_cutter,
)

from validate_node import (
    validate_node,
    route_after_validation,
)

from gate_cutting_node import gate_cutting_langgraph_node
from auto_finder import autofinder_langgraph_node

from llm_custom_cut_node import llm_custom_cut_langgraph_node

from inspection_export import export_cutting_results


# 1.  BUILD THE SUBGRAPH

def build_subgraph():
    """Compile the Parse → Analyze subgraph."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("parse_circuit",   parse_circuit_node)
    workflow.add_node("analyze_circuit", analyze_circuit_node)


    workflow.add_node("strategy_router", strategy_router_node)

    workflow.add_node("qpd_gate_cut", gate_cutting_langgraph_node)
    workflow.add_node("auto_finder", autofinder_langgraph_node)

    workflow.add_node("llm_custom_cut", llm_custom_cut_langgraph_node)

    workflow.add_node("validate", validate_node)


    # Wire the flow
    workflow.set_entry_point("parse_circuit")
    workflow.add_edge("parse_circuit", "analyze_circuit")
    workflow.add_edge("analyze_circuit", "strategy_router")

    workflow.add_conditional_edges(
        "strategy_router",
        route_to_cutter,
        {
            "qpd_gate_cut": "qpd_gate_cut",
            "auto_finder": "auto_finder",

            "llm_custom_cut": "llm_custom_cut",
        },
    )
    workflow.add_edge("qpd_gate_cut", "validate")
    workflow.add_edge("auto_finder", "validate")
    workflow.add_edge("llm_custom_cut", "validate")

    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "strategy_router": "strategy_router",
            "end": END,
        },
    )


    return workflow.compile()


# 2.  RUN AN EXAMPLE CIRCUIT

def run(qasm_path: str, export_dir: str | None = "cutting_runs") -> AgentState:
    """Invoke the subgraph on a single QASM file.

    Parameters
    ----------
    qasm_path : str
        Path to the QASM file to process.
    export_dir : str | None
        If set (default "cutting_runs"), every circuit produced during the
        run (original, cut, marked, and all partitioned subcircuits) plus
        the run's metadata is written to disk under this directory for
        manual inspection / equivalence testing. Pass ``None`` to skip
        exporting entirely.
    """
    app = build_subgraph()

    initial_state: AgentState = {
        "qasm_path":        qasm_path,
        "circuit":          None,
        "circuit_summary":  None,
        "analysis":         None,
        "analysis_summary": None,

        "cutting_strategy": None,
        "strategy_reasoning": None,
        "selector_note": None,
        "decision_rationale": None,
        "gate_ids": None,
        "auto_partition": False,
        "partition_labels": None,
        "cut_result": None,

        "original_circuit": None,
        "cut_circuit": None,
        "qpd_bases": None,
        "gate_info": None,
        "total_sampling_overhead": None,
        "subcircuits": None,
        "metadata": None,

        "validation_passed": None,
        "validation_reason": None,
        "validation_attempts": 0,

        "equivalence_passed": None,
        "equivalence_detail": None,
        "equivalence_seconds": None,
        "equivalence_skipped": None,

        "cut_locations": None,
        "cut_info": None,
        "marked_circuit": None,
        "minimum_reached": None,

        # this run, so strategy_router never re-routes to one of them.
        "failed_strategies": [],

        "llm_fallback":     False,

        "error":            None,

    }

    final_state = app.invoke(initial_state)

    if export_dir is not None:
        export_cutting_results(final_state, output_dir=export_dir)

    return final_state


# 3.  CLI ENTRY POINT

if __name__ == "__main__":
    import sys, json

    path  = sys.argv[1] if len(sys.argv) > 1 else "test.qasm"
    final = run(path)

    if final["error"]:
        print(f"ERROR: {final['error']}")
        sys.exit(1)

    print("=" * 60)
    print("PARSE OUTPUT")
    print("=" * 60)
    print(final["circuit_summary"])

    print("\n" + "=" * 60)
    print("ANALYZE OUTPUT (LLM-readable)")
    print("=" * 60)
    print(final["analysis_summary"])

    print("\n" + "=" * 60)
    print("ANALYZE OUTPUT (structured dict for Strategy_router)")
    print("=" * 60)
    print(json.dumps(final["analysis"], indent=2))


    print("\n" + "=" * 60)
    print("STRATEGY")
    print("=" * 60)
    print(final.get("cutting_strategy"))

    print("\n" + "=" * 60)
    print("SELECTOR NOTE (why this strategy over others)")
    print("=" * 60)
    print(final.get("selector_note"))

    print("\n" + "=" * 60)
    print("DECISION RATIONALE (router's own reasoning for this circuit)")
    print("=" * 60)
    print(final.get("decision_rationale"))

    print("\n" + "=" * 60)
    print("FULL ROUTING SUMMARY")
    print("=" * 60)
    print(final.get("strategy_reasoning"))

    print("\n" + "=" * 60)
    print("CUT GATE INDICES")
    print("=" * 60)
    print(final.get("gate_ids"))

    print("\n" + "=" * 60)
    print("GATE CUTTING OUTPUT KEYS")
    print("=" * 60)
    print(list(final.keys()))
