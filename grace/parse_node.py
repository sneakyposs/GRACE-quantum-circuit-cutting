"""
parse_node.py
-------------
LangGraph node that reads a QASM file and produces an LLM-friendly
summary stored in the shared agent state.

Supports OpenQASM 2.0 (.qasm) via Qiskit.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Any
from qiskit import QuantumCircuit


# 1.  SHARED STATE
#     Every node in your LangGraph reads/writes
#     this dict.  Add more fields as your graph
#     grows (strategy, cut_result, …).

class AgentState(TypedDict):
    qasm_path:        str                # INPUT  – path to the .qasm file
    circuit:          Optional[Any]      # parsed QuantumCircuit object
    circuit_summary:  Optional[str]      # LLM-readable text block
    error:            Optional[str]      # non-None means something went wrong

    analysis:         Optional[dict]
    analysis_summary: Optional[str]

    cutting_strategy:   Optional[str]
    strategy_reasoning: Optional[str]
    # the terse one-liner from the strategy selector (why this technique
    # over the others); `decision_rationale` is a dedicated LLM paragraph
    # the rationale just copied analyze_circuit_node's reasoning_notes.
    selector_note:      Optional[str]
    decision_rationale: Optional[str]
    gate_ids:           Optional[list[int]]
    auto_partition:     Optional[bool]
    partition_labels:   Optional[list]
    cut_result:         Optional[dict]

    # in gate_cutting_node.py. LangGraph silently drops any key a node
    # returns that isn't declared in this schema, which is why validate_node
    # was always seeing "original_circuit"/"cut_circuit"/"subcircuits" as
    # missing even though gate_cutting_langgraph_node was writing them.
    original_circuit:       Optional[Any]
    cut_circuit:             Optional[Any]
    qpd_bases:               Optional[list]
    gate_info:               Optional[list]
    total_sampling_overhead: Optional[float]
    subcircuits:             Optional[dict]
    metadata:                Optional[dict]

    validation_passed: Optional[bool]
    validation_reason: Optional[str]   #-> all new validation node stuff
    validation_attempts: Optional[int]

    # Same root cause as the comment above: LangGraph silently drops any key
    # a node returns that isn't declared here, so auto_finder's
    # minimum_reached was being dropped on the floor.
    cut_locations:   Optional[list]
    cut_info:        Optional[list]
    marked_circuit:  Optional[Any]
    minimum_reached: Optional[bool]

    # validation during this run, so strategy_router can avoid
    # re-selecting a technique that's already proven to fail instead of
    # looping on it until validate_node's MAX_VALIDATION_LOOPS hits.
    # Written to by validate_node, read by strategy_router.
    failed_strategies: Optional[list[str]]

    # validate_node (which now dispatches the strategy's offline
    # equivalence validator itself). Declared here because LangGraph
    # silently drops any key a node returns that isn't in this schema.
    #   equivalence_passed  : True/False once the validator ran; None if
    #                         it was skipped or never reached.
    #   equivalence_detail  : validator output tail / skip reason.
    #   equivalence_seconds : wall-clock of the validator subprocess.
    #   equivalence_skipped : None, 'infeasible' (cost model says the
    #                         check can't finish within budget), or
    #                         'no_cut' (legitimate AutoFinder no-cut run,
    #                         nothing to reconstruct).
    equivalence_passed:  Optional[bool]
    equivalence_detail:  Optional[str]
    equivalence_seconds: Optional[float]
    equivalence_skipped: Optional[str]

    # LangGraph silently drops any key a node returns that isn't declared
    # here, so these MUST be present or the LLM's reasoning paragraph and
    # chosen technique would be dropped on the floor before
    # inspection_export.py / validate_node ever see them.
    llm_cut_technique: Optional[str]   # "qpd_gate_cut" or "qpd_wire_cut"
    llm_cut_reasoning: Optional[str]   # the LLM's justification paragraph
    llm_fallback:      Optional[bool]  # True when LLM was unavailable/failed
                                       # and a deterministic heuristic was used


# 2.  NODE FUNCTION
#     Signature required by LangGraph:
#       fn(state: State) -> State

def parse_circuit_node(state: AgentState) -> AgentState:
    """
    Node 1 – Parse Circuit
    Reads the QASM file at state["qasm_path"], converts it to a
    QuantumCircuit, and writes a plain-English summary the LLM can use.
    """
    path = state.get("qasm_path", "")

    if not path:
        return {**state, "error": "No qasm_path provided in state."}

    try:
        circuit = QuantumCircuit.from_qasm_file(path)
        summary = _format_for_llm(circuit)
        return {
            **state,
            "circuit":         circuit,
            "circuit_summary": summary,
            "error":           None,
        }
    except FileNotFoundError:
        return {**state, "error": f"File not found: {path}"}
    except Exception as exc:
        return {**state, "error": f"Parse failed: {exc}"}


# 3.  FORMATTER
#     Turns a QuantumCircuit into a structured
#     string the LLM can reason about directly.

def _format_for_llm(circuit: QuantumCircuit) -> str:
    """Return a structured, LLM-readable description of the circuit."""

    lines: list[str] = []

    # ── Basic metadata ──────────────────────────────────────────────────
    lines += [
        "=== CIRCUIT METADATA ===",
        f"Qubits          : {circuit.num_qubits}",
        f"Classical bits  : {circuit.num_clbits}",
        f"Circuit depth   : {circuit.depth()}",
        f"Total operations: {len(circuit.data)}",
    ]

    # ── Gate type counts ────────────────────────────────────────────────
    lines += ["", "=== GATE COUNTS ==="]
    for gate, count in sorted(circuit.count_ops().items()):
        lines.append(f"  {gate:<12} x{count}")

    # ── Full gate sequence ──────────────────────────────────────────────
    lines += ["", "=== GATE SEQUENCE (index | gate | qubits | params) ==="]
    for idx, inst in enumerate(circuit.data):
        name   = inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        clbits = [circuit.find_bit(c).index for c in inst.clbits]
        params = inst.operation.params

        qubit_str = ", ".join(f"q{q}" for q in qubits)
        param_str = f"  params=({', '.join(f'{p:.4g}' for p in params)})" if params else ""
        meas_str  = f"  -> c{clbits}" if clbits else ""

        lines.append(f"  [{idx:3d}] {name:<8} on {qubit_str}{param_str}{meas_str}")

    # ── Two-qubit interactions (critical for partitioning decisions) ─────
    two_q = [i for i in circuit.data if len(i.qubits) == 2]
    if two_q:
        lines += ["", "=== TWO-QUBIT INTERACTIONS ==="]
        seen: set[tuple[int,int]] = set()
        for inst in two_q:
            a = circuit.find_bit(inst.qubits[0]).index
            b = circuit.find_bit(inst.qubits[1]).index
            pair = (min(a, b), max(a, b))
            if pair not in seen:
                lines.append(f"  q{pair[0]} <-> q{pair[1]}")
                seen.add(pair)

    # ── ASCII diagram ───────────────────────────────────────────────────

    if circuit.num_qubits <= 4:
        lines += ["", "=== CIRCUIT DIAGRAM ===",
                  str(circuit.draw(output="text", fold=-1))]
    else:
        # end of the function, so _format_for_llm returned None for every
        # circuit with >4 qubits. That None became the literal string
        # "None" in the strategy_router LLM prompt (and an empty summary in
        # llm_custom_cut), silently discarding the gate sequence and the
        # two-qubit interaction list for the bulk of a 3-10 qubit sweep.
        lines += ["", "=== CIRCUIT DIAGRAM ===",
                  "Circuit too large for useful diagram."]

    return "\n".join(lines)
# 4.  QUICK SMOKE-TEST  (python parse_node.py)

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test.qasm"

    state: AgentState = {
        "qasm_path":       path,
        "circuit":         None,
        "circuit_summary": None,
        "error":           None,
        "failed_strategies": [],
    }

    result = parse_circuit_node(state)

    if result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        print(result["circuit_summary"])
