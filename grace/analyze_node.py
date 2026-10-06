"""
analyze_node.py
---------------
LangGraph node that consumes a parsed QuantumCircuit (from parse_circuit_node)
and produces:
  1. A structured `analysis` dict (JSON-like) for the future Strategy_router.
  2. An LLM-readable `analysis_summary` text block.


`_invoke_llm_reasoner()` now calls openai/gpt-oss-120b:free (via
OpenRouter, see llm_client.py) to write the `reasoning_notes` field,
instead of the old hardcoded-rules stub. The original rules are kept as
`_deterministic_reasoning_notes()`, used as an automatic fallback if the
LLM call fails for any reason.
"""

from __future__ import annotations

from typing import Optional, TypedDict
from qiskit import QuantumCircuit

# Reuse the shared state defined in parse_node.py
#from sub_circuit_test.parse_node import AgentState
from parse_node import AgentState

# openai/gpt-oss-120b:free via OpenRouter. See _invoke_llm_reasoner() below
# for where it's actually used.
from llm_client import call_llm


# 1.  ANALYSIS SCHEMA
#     This is the contract that Strategy_router
#     will consume. Keep it stable & JSON-safe.

class CircuitAnalysis(TypedDict):
    # Core size metrics
    num_qubits:            int
    num_clbits:            int
    depth:                 int
    total_operations:      int

    # Gate breakdown
    gate_counts:           dict          # {"h": 4, "cx": 6, ...}
    single_qubit_gates:    int
    two_qubit_gates:       int
    multi_qubit_gates:     int           # 3+ qubit gates (rare, e.g. ccx)
    measurement_count:     int
    non_clifford_count:    int           # t, tdg, rz, rx, ry, u, u3, etc.

    # Connectivity
    interacting_pairs:     list          # [[0,1], [1,2], ...]
    num_unique_pairs:      int
    connectivity_density:  float         # unique_pairs / max_possible_pairs

    # Derived signals (cheap heuristics for the router)
    gate_density:          float         # total_ops / (qubits * depth)
    is_dense:              bool          # connectivity_density > 0.5
    is_deep:               bool          # depth > 50

    # LLM hand-off (filled deterministically for now)
    reasoning_notes:       str


# 2.  NODE FUNCTION
#     Signature required by LangGraph:
#       fn(state: State) -> State

def analyze_circuit_node(state: AgentState) -> AgentState:
    """
    Node 2 – Analyze Circuit
    Reads the parsed QuantumCircuit from state["circuit"], computes
    quantitative metrics, and writes both a structured `analysis` dict
    and an LLM-readable `analysis_summary` into the shared state.
    """
    # Short-circuit if a previous node failed
    if state.get("error"):
        return state

    circuit: Optional[QuantumCircuit] = state.get("circuit")
    if circuit is None:
        return {**state, "error": "No circuit found in state. Run parse_circuit_node first."}

    try:
        analysis = _compute_metrics(circuit)
        analysis["reasoning_notes"] = _invoke_llm_reasoner(analysis)
        summary = _format_analysis_for_llm(analysis)

        return {
            **state,
            "analysis":         analysis,
            "analysis_summary": summary,
            "error":            None,
        }
    except Exception as exc:
        return {**state, "error": f"Analyze failed: {exc}"}


# 3.  METRIC COMPUTATION
#     Pure, deterministic, no LLM required.

# Gates considered "non-Clifford" — these dominate classical simulation cost
# and are useful signals for the cutting-strategy router.
_NON_CLIFFORD = {"t", "tdg", "rz", "rx", "ry", "u", "u1", "u2", "u3", "p"}


def _compute_metrics(circuit: QuantumCircuit) -> CircuitAnalysis:
    """Walk the circuit once and build the structured analysis dict."""

    gate_counts = dict(circuit.count_ops())

    single_q = two_q = multi_q = meas = non_cliff = 0
    interacting: set[tuple[int, int]] = set()

    for inst in circuit.data:
        name   = inst.operation.name
        n_q    = len(inst.qubits)

        if name == "measure":
            meas += 1
            continue

        # interacting pair), inflating two_qubit_gates / connectivity
        # metrics that the Strategy_router's LLM prompt relies on.
        if name in ("barrier", "reset"):
            continue

        if n_q == 1:
            single_q += 1
        elif n_q == 2:
            two_q += 1
            a = circuit.find_bit(inst.qubits[0]).index
            b = circuit.find_bit(inst.qubits[1]).index
            interacting.add((min(a, b), max(a, b)))
        else:
            multi_q += 1

        if name in _NON_CLIFFORD:
            non_cliff += 1

    n           = circuit.num_qubits
    depth       = circuit.depth()
    total_ops   = len(circuit.data)
    max_pairs   = n * (n - 1) // 2 if n > 1 else 1
    gate_dens   = total_ops / (n * depth) if (n and depth) else 0.0
    conn_dens   = len(interacting) / max_pairs if max_pairs else 0.0

    return {
        "num_qubits":           n,
        "num_clbits":           circuit.num_clbits,
        "depth":                depth,
        "total_operations":     total_ops,

        "gate_counts":          gate_counts,
        "single_qubit_gates":   single_q,
        "two_qubit_gates":      two_q,
        "multi_qubit_gates":    multi_q,
        "measurement_count":    meas,
        "non_clifford_count":   non_cliff,

        "interacting_pairs":    [list(p) for p in sorted(interacting)],
        "num_unique_pairs":     len(interacting),
        "connectivity_density": round(conn_dens, 4),

        "gate_density":         round(gate_dens, 4),
        "is_dense":             conn_dens > 0.5,
        "is_deep":              depth > 50,

        "reasoning_notes":      "",   # filled by _invoke_llm_reasoner
    }


# 4.  LLM HOOK
#     openai/gpt-oss-120b:free via llm_client.py
#     original rules were kept as a fallback in
#     _deterministic_reasoning_notes() below.

def _invoke_llm_reasoner(analysis: CircuitAnalysis) -> str:
    """
    
    Calls openai/gpt-oss-120b:free (via OpenRouter, see llm_client.py) to
    write the `reasoning_notes` field that Strategy_router reads.

    This REPLACES the old hardcoded-rules body that lived here. The LLM
    is asked to reason in plain English about what the circuit's metrics
    (connectivity, depth, density, non-Clifford content) imply for a
    later cutting-strategy decision — it is deliberately NOT asked to
    pick a strategy itself; that choice belongs to Strategy_router (Node 3).

    If the LLM call fails for any reason (no .env / API key, no internet,
    rate limit, empty response, etc.) this falls back to the original
    deterministic heuristic so the rest of the graph keeps working.
    """
    try:
        prompt = _build_analysis_prompt(analysis)
        notes = call_llm(
            system_prompt=(
                "You are a quantum-computing assistant embedded inside an "
                "automated circuit-partitioning pipeline. You will be given "
                "quantitative metrics about a quantum circuit. Write 2-4 "
                "concise sentences of plain-English reasoning about what "
                "these metrics imply for how 'cuttable' the circuit is — "
                "covering connectivity/sparsity, depth, and non-Clifford "
                "content where relevant. Do NOT recommend a specific "
                "cutting strategy by name; that decision belongs to a "
                "later step in the pipeline. Reply with prose only, no "
                "headers or bullet points."
            ),
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=300,
        )
        return notes
    except Exception as exc:
        # Fallback: the original hardcoded heuristic, with a short note
        # so it's obvious in the output that the LLM call did not run.
        fallback = _deterministic_reasoning_notes(analysis)
        return f"{fallback} [LLM unavailable, used fallback heuristic: {exc}]"


def _deterministic_reasoning_notes(analysis: CircuitAnalysis) -> str:
    """
    Original hardcoded-rules logic, kept as the fallback path for
    _invoke_llm_reasoner() above when the LLM call cannot complete.
    """
    notes: list[str] = []

    if analysis["two_qubit_gates"] == 0:
        notes.append("No two-qubit gates — circuit is trivially separable.")
    elif analysis["num_unique_pairs"] <= 2:
        notes.append("Sparse connectivity — wire-cutting is likely cheap.")
    elif analysis["is_dense"]:
        notes.append("Dense connectivity — gate-cutting may be preferable to wire-cutting.")
    else:
        notes.append("Moderate connectivity — AutoFinder is a reasonable default.")

    if analysis["is_deep"]:
        notes.append(f"Deep circuit (depth={analysis['depth']}) — consider depth-aware partitioning.")

    if analysis["non_clifford_count"] > 0:
        notes.append(
            f"{analysis['non_clifford_count']} non-Clifford gate(s) present — "
            "expect higher classical-simulation overhead for any cut."
        )

    return " ".join(notes)


def _build_analysis_prompt(analysis: CircuitAnalysis) -> str:
    """
    
    Builds the user-turn prompt sent to the LLM by
    _invoke_llm_reasoner(), mirroring the style of
    strategy_router.py's _build_strategy_prompt().
    """
    return f"""Here are the structural metrics computed for a quantum circuit:

- Qubits                  : {analysis['num_qubits']}
- Circuit depth           : {analysis['depth']}
- Total operations        : {analysis['total_operations']}
- Gate density             : {analysis['gate_density']}
- Single-qubit gates       : {analysis['single_qubit_gates']}
- Two-qubit gates          : {analysis['two_qubit_gates']}
- Multi-qubit gates        : {analysis['multi_qubit_gates']}
- Measurements             : {analysis['measurement_count']}
- Non-Clifford gates       : {analysis['non_clifford_count']}
- Unique interacting pairs : {analysis['num_unique_pairs']} -> {analysis['interacting_pairs']}
- Connectivity density     : {analysis['connectivity_density']}
- Dense?                   : {analysis['is_dense']}
- Deep?                    : {analysis['is_deep']}

In 2-4 sentences, explain what these numbers suggest about how "cuttable"
this circuit is, and which structural properties matter most for a
downstream circuit-cutting decision.
"""


# 5.  FORMATTER
#     Mirrors _format_for_llm() in parse_node.py

def _format_analysis_for_llm(a: CircuitAnalysis) -> str:
    """Return a structured, LLM-readable description of the analysis."""
    lines: list[str] = []

    lines += [
        "=== CIRCUIT ANALYSIS ===",
        f"Qubits              : {a['num_qubits']}",
        f"Depth               : {a['depth']}",
        f"Total operations    : {a['total_operations']}",
        f"Gate density        : {a['gate_density']}",
        "",
        "=== GATE BREAKDOWN ===",
        f"Single-qubit gates  : {a['single_qubit_gates']}",
        f"Two-qubit gates     : {a['two_qubit_gates']}",
        f"Multi-qubit gates   : {a['multi_qubit_gates']}",
        f"Measurements        : {a['measurement_count']}",
        f"Non-Clifford gates  : {a['non_clifford_count']}",
        "",
        "=== CONNECTIVITY ===",
        f"Unique qubit pairs  : {a['num_unique_pairs']}",
        f"Connectivity density: {a['connectivity_density']}",
        f"Dense?              : {a['is_dense']}",
        f"Deep?               : {a['is_deep']}",
        "",
        "=== REASONING NOTES ===",
        a["reasoning_notes"],
    ]
    return "\n".join(lines)


# 6.  QUICK SMOKE-TEST  (python analyze_node.py)

if __name__ == "__main__":
    import sys
    from parse_node import parse_circuit_node

    path = sys.argv[1] if len(sys.argv) > 1 else "test.qasm"

    state: AgentState = {
        "qasm_path":        path,
        "circuit":          None,
        "circuit_summary":  None,
        "analysis":         None,
        "analysis_summary": None,
        "error":            None,
    }

    state = parse_circuit_node(state)
    state = analyze_circuit_node(state)

    if state["error"]:
        print(f"ERROR: {state['error']}")
    else:
        print(state["analysis_summary"])
        print("\n--- Raw analysis dict (for Strategy_router) ---")
        import json
        print(json.dumps(state["analysis"], indent=2))
