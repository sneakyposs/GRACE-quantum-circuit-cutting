"""
wire_cutting_node.py
====================

A standalone QPD-based wire-cutting node for agentic quantum circuit cutting
pipelines.  Designed for direct use *and* trivial integration into LangGraph.

Mirrors the style and structure of ``gate_cutting_node.py`` so that the two
nodes are interchangeable from the perspective of the Strategy_router and
validate nodes.

Wire cutting is performed using the open-source ``qiskit-addon-cutting``
package.  Single-qubit ``CutWire`` placeholders are inserted on the
user-selected wire locations, then the official ``cut_wires`` helper
converts them into two-qubit ``Move`` instructions wrapped in
``TwoQubitQPDGate`` placeholders.  No custom wire-cutting math is performed
here.

Requires:
    pip install 'qiskit-addon-cutting>=0.10' qiskit

Author : Ezekiel Laney
License: Apache-2.0
"""

from __future__ import annotations
from parse_node import AgentState

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import CircuitInstruction
from qiskit_addon_cutting import (
    cut_wires,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.instructions import CutWire
from qiskit_addon_cutting.qpd import QPDBasis
from qiskit_addon_cutting.qpd.instructions import TwoQubitQPDGate

# 1.  Data containers

@dataclass
class WireCuttingResult:
    """Structured output of the wire-cutting node.

    Attributes
    ----------
    original_circuit : QuantumCircuit
        The circuit exactly as it was received (before any modification).
    cut_circuit : QuantumCircuit
        The circuit after the requested wire cuts have been applied.  This
        circuit has one additional qubit for every cut wire, and contains
        ``TwoQubitQPDGate`` placeholders wrapping ``Move`` instructions.
    marked_circuit : QuantumCircuit
        Intermediate circuit with single-qubit ``CutWire`` markers inserted
        at the user-specified locations (before ``cut_wires`` expansion).
        Useful for debugging / visualization.
    qpd_bases : list[QPDBasis]
        One ``QPDBasis`` per cut wire, in the same order as ``cut_locations``.
        Each basis encodes the quasi-probability decomposition (maps,
        coefficients, sampling overhead) for that wire cut.
    cut_locations : list[dict[str, Any]]
        Echo of the user-supplied cut locations (qubit index + instruction
        index) for traceability.
    cut_info : list[dict[str, Any]]
        Human-readable metadata for every cut wire (qubit, position, kappa,
        overhead).
    total_sampling_overhead : float
        Product of per-cut overheads.  This is the multiplicative factor by
        which the total shot budget must increase to maintain the same
        statistical precision as the uncut circuit.
    partition_labels : list[int | None] | None
        If automatic partitioning was requested, the labels assigned to
        each qubit of the *expanded* circuit.  ``None`` otherwise.
    subcircuits : dict | None
        If ``partition_problem`` was invoked, a mapping from partition
        label to subcircuit.  ``None`` otherwise.
    metadata : dict[str, Any]
        Miscellaneous metadata for downstream nodes.
    """

    original_circuit: QuantumCircuit
    cut_circuit: QuantumCircuit
    marked_circuit: QuantumCircuit
    qpd_bases: list[QPDBasis]
    cut_locations: list[dict[str, Any]]
    cut_info: list[dict[str, Any]]
    total_sampling_overhead: float
    partition_labels: list[int | None] | None = None
    subcircuits: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (e.g. for LangGraph state)."""
        return {
            "original_circuit": self.original_circuit,
            "cut_circuit": self.cut_circuit,
            "marked_circuit": self.marked_circuit,
            "qpd_bases": self.qpd_bases,
            "cut_locations": self.cut_locations,
            "cut_info": self.cut_info,
            "total_sampling_overhead": self.total_sampling_overhead,
            "partition_labels": self.partition_labels,
            "subcircuits": self.subcircuits,
            "metadata": self.metadata,
        }


# 2.  Internal helpers

def _load_circuit(circuit_input: str | QuantumCircuit) -> QuantumCircuit:
    """Load a circuit from a QASM string or pass through a ``QuantumCircuit``."""
    if isinstance(circuit_input, QuantumCircuit):
        return circuit_input
    if not isinstance(circuit_input, str):
        raise TypeError(
            "circuit_input must be a QuantumCircuit or an OpenQASM string, "
            f"got {type(circuit_input).__name__}"
        )
    # Try QASM 3 first, then QASM 2.
    text = circuit_input.strip()
    if text.startswith("OPENQASM 3") or "OPENQASM 3" in text.splitlines()[0]:
        try:
            return qasm3.loads(text)
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        return qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception:
        # Last-ditch: try qasm3 even if header didn't match.
        return qasm3.loads(text)


def _strip_classical_bits(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of ``circuit`` with measurements / classical regs stripped.

    The cutting addon does not operate on measurements; downstream sampling
    handled by ``generate_cutting_experiments`` re-introduces measurements
    as needed.
    """
    new = QuantumCircuit(*circuit.qregs)
    for inst in circuit.data:
        # survives into the cut circuit and partition_problem's automatic
        # connectivity labelling treats it as a link between qubits,
        # gluing partitions together (same bug as gate_cutting_node).
        # Note: list_cuttable_wires uses this same helper, so cut-location
        # indices stay consistent with the work circuit built here.
        if inst.operation.name in ("measure", "reset", "barrier"):
            continue
        if len(inst.clbits) > 0:
            continue
        new.append(inst.operation, inst.qubits, [])
    return new


def _normalize_cut_locations(
    cut_locations: Sequence[Any],
) -> list[dict[str, int]]:
    """Normalize user-supplied cut-location specs into a uniform list of dicts.

    Accepted item formats:
      * ``{"qubit": int, "after_instruction": int}``
      * ``(qubit, after_instruction)`` tuple/list of length 2
    """
    normalized: list[dict[str, int]] = []
    for i, item in enumerate(cut_locations):
        if isinstance(item, dict):
            if "qubit" not in item or "after_instruction" not in item:
                raise ValueError(
                    f"cut_locations[{i}] dict must contain keys "
                    "'qubit' and 'after_instruction'."
                )
            normalized.append(
                {
                    "qubit": int(item["qubit"]),
                    "after_instruction": int(item["after_instruction"]),
                }
            )
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            normalized.append(
                {
                    "qubit": int(item[0]),
                    "after_instruction": int(item[1]),
                }
            )
        else:
            raise ValueError(
                f"cut_locations[{i}] has unsupported format: {item!r}. "
                "Expected dict with 'qubit'/'after_instruction' or (qubit, idx) tuple."
            )
    return normalized


def _insert_cut_wire_markers(
    circuit: QuantumCircuit,
    cut_locations: list[dict[str, int]],
) -> QuantumCircuit:
    """Return a new circuit with single-qubit ``CutWire`` markers inserted.

    A marker ``CutWire`` is appended to qubit ``q`` *immediately after* the
    instruction at index ``after_instruction`` in the *original* circuit's
    ``data`` list.  Indices refer to the input circuit (not the output),
    so multiple cuts can be specified relative to the same reference.
    """
    # Group cuts by their insertion point in the original instruction list.
    # We sort by ``after_instruction`` ascending so we can walk the data once.
    sorted_cuts = sorted(
        enumerate(cut_locations), key=lambda kv: kv[1]["after_instruction"]
    )

    new = QuantumCircuit(*circuit.qregs)
    cut_iter = iter(sorted_cuts)
    next_cut = next(cut_iter, None)

    for idx, inst in enumerate(circuit.data):
        new.append(inst.operation, inst.qubits, inst.clbits)
        # After appending instruction `idx`, insert any cuts targeting this index.
        while next_cut is not None and next_cut[1]["after_instruction"] == idx:
            q = next_cut[1]["qubit"]
            if q < 0 or q >= circuit.num_qubits:
                raise ValueError(
                    f"Cut qubit {q} out of range for {circuit.num_qubits}-qubit circuit."
                )
            new.append(CutWire(), [new.qubits[q]], [])
            next_cut = next(cut_iter, None)

    # Handle cuts whose after_instruction is beyond the last instruction
    # (i.e., cuts at the very end of the circuit).
    while next_cut is not None:
        q = next_cut[1]["qubit"]
        if q < 0 or q >= circuit.num_qubits:
            raise ValueError(
                f"Cut qubit {q} out of range for {circuit.num_qubits}-qubit circuit."
            )
        new.append(CutWire(), [new.qubits[q]], [])
        next_cut = next(cut_iter, None)

    return new


def _extract_qpd_bases(cut_circuit: QuantumCircuit) -> list[QPDBasis]:
    """Pull every ``QPDBasis`` from ``TwoQubitQPDGate`` placeholders in order."""
    bases: list[QPDBasis] = []
    for inst in cut_circuit.data:
        if isinstance(inst.operation, TwoQubitQPDGate):
            bases.append(inst.operation.basis)
    return bases


# 3.  Convenience helpers for exploration

def list_cuttable_wires(circuit_input: str | QuantumCircuit) -> list[dict[str, Any]]:
    """Return metadata describing every plausible wire-cut location.

    A *wire cut* can in principle be placed on any qubit at any point in
    time.  In practice, useful cut locations sit *between* two-qubit
    operations on the same wire, since that is where they can disconnect
    a circuit into smaller pieces.  This helper enumerates such candidate
    locations to assist an LLM agent (or human) in choosing wire cuts.

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        OpenQASM string or ``QuantumCircuit``.

    Returns
    -------
    list[dict]
        One dict per candidate location, containing ``qubit``,
        ``after_instruction``, ``preceding_gate``, and ``following_gate``.
    """
    circuit = _load_circuit(circuit_input)
    circuit = _strip_classical_bits(circuit)

    # For each qubit, walk the instruction list and record every position
    # that sits between two two-qubit gates touching that qubit.
    candidates: list[dict[str, Any]] = []
    for q in range(circuit.num_qubits):
        last_two_qubit_idx: int | None = None
        last_two_qubit_name: str | None = None
        for i, inst in enumerate(circuit.data):
            if circuit.qubits[q] not in inst.qubits:
                continue
            if inst.operation.name == "barrier":
                continue
            is_two_qubit = len(inst.qubits) == 2
            if is_two_qubit and last_two_qubit_idx is not None:
                candidates.append(
                    {
                        "qubit": q,
                        "after_instruction": last_two_qubit_idx,
                        "preceding_gate": last_two_qubit_name,
                        "following_gate": inst.operation.name,
                    }
                )
            if is_two_qubit:
                last_two_qubit_idx = i
                last_two_qubit_name = inst.operation.name
    return candidates


def estimate_wire_cut_overhead(
    circuit_input: str | QuantumCircuit,
    cut_locations: Sequence[Any],
) -> dict[str, Any]:
    """Quickly estimate the sampling overhead for a proposed set of wire cuts.

    Performs the cut (without partitioning) and returns the per-cut and
    total sampling overheads.  Intended for cheap "what-if" probing by an
    upstream agent before committing to a full ``wire_cutting_node`` call.
    """
    circuit = _load_circuit(circuit_input)
    circuit = _strip_classical_bits(circuit)
    norm = _normalize_cut_locations(cut_locations)
    marked = _insert_cut_wire_markers(circuit, norm)
    cut = cut_wires(marked)
    bases = _extract_qpd_bases(cut)
    overheads = [b.overhead for b in bases]
    total = 1.0
    for o in overheads:
        total *= o
    return {
        "num_cuts": len(bases),
        "per_cut_overhead": overheads,
        "total_sampling_overhead": total,
    }


# 4.  Main node entry-point
def wire_cutting_langgraph_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    circuit = state.get("circuit")
    if circuit is None:
        return {**state, "error": "No circuit found."}

    try:
        result = wire_cutting_node(
            circuit_input=circuit,
            cut_locations=state.get("cut_locations") or [],
            partition_labels=state.get("partition_labels"),
            auto_partition=state.get("auto_partition", False),
        )
    except Exception as exc:
        # cut-point patterns (e.g. CircuitError 'duplicate bit arguments'
        # exception escaped this node and killed the entire LangGraph
        # run (PIPELINE_ERROR). Instead, clear the cut outputs so the
        # Validate node records qpd_wire_cut as a failed strategy and
        # the Strategy_router reroutes to an alternative technique.
        print(f"[WIRE_CUT] ERROR: {type(exc).__name__}: {exc} "
              f"-> marking strategy failed so the router can retry")
        return {
            **state,
            "error": None,
            "cut_circuit": None,
            "subcircuits": None,
            "cut_result": None,
            "marked_circuit": None,
            "cut_info": None,
            "total_sampling_overhead": None,
        }

    return {
        **state,
        **result.to_dict(),
        "error": None,
    }

def wire_cutting_node(
    circuit_input: str | QuantumCircuit,
    cut_locations: Sequence[Any],
    partition_labels: Sequence[int | None] | None = None,
    auto_partition: bool = False,
) -> WireCuttingResult:
    """Perform QPD-based wire cutting on specified wires of a quantum circuit.

    This is the single entry-point designed for both standalone use and
    LangGraph integration.

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        An OpenQASM 2/3 string **or** a Qiskit ``QuantumCircuit``.
    cut_locations : Sequence
        Locations at which to cut wires.  Each entry may be either:

          * a ``dict`` ``{"qubit": int, "after_instruction": int}``, or
          * a ``(qubit, after_instruction)`` tuple/list.

        ``after_instruction`` is an index into ``circuit.data`` of the
        *original* circuit; the cut is placed immediately after that
        instruction on the specified qubit.
    partition_labels : Sequence[int | None] | None, optional
        Explicit partition labels for each qubit of the **expanded** cut
        circuit (which has one additional qubit per cut).  If provided
        together with *auto_partition=True*, the explicit labels take
        precedence.
    auto_partition : bool, default False
        If ``True`` **and** *partition_labels* is ``None``, call
        ``partition_problem`` to automatically separate the cut circuit
        into subcircuits.

    Returns
    -------
    WireCuttingResult
        A structured result containing the original circuit, marked
        circuit, cut circuit, QPD bases, per-cut metadata, total sampling
        overhead, and (optionally) partition information.
    """
    # ---- 1. Load and snapshot the original circuit ---------------------
    original = _load_circuit(circuit_input)
    original_snapshot = copy.deepcopy(original)

    work_circuit = _strip_classical_bits(original)

    # ---- 2. Normalize cut locations & validate -------------------------
    norm_cuts = _normalize_cut_locations(cut_locations)
    if len(norm_cuts) == 0:
        warnings.warn(
            "wire_cutting_node called with no cut_locations; "
            "returning the original circuit unchanged.",
            stacklevel=2,
        )

    # ---- 3. Insert CutWire markers, then expand to Move/TwoQubitQPDGate
    marked_circuit = _insert_cut_wire_markers(work_circuit, norm_cuts)
    cut_circuit = cut_wires(marked_circuit)

    # ---- 4. Extract QPD bases & per-cut info ---------------------------
    qpd_bases = _extract_qpd_bases(cut_circuit)

    cut_info: list[dict[str, Any]] = []
    for i, (loc, basis) in enumerate(zip(norm_cuts, qpd_bases)):
        cut_info.append(
            {
                "cut_index": i,
                "qubit": loc["qubit"],
                "after_instruction": loc["after_instruction"],
                "kappa": basis.kappa,
                "overhead": basis.overhead,
                "num_maps": len(basis.maps),
            }
        )

    total_overhead = 1.0
    for b in qpd_bases:
        total_overhead *= b.overhead

    # ---- 5. Optional partitioning --------------------------------------
    resolved_labels: list[int | None] | None = None
    subcircuits = None
    if partition_labels is not None:
        resolved_labels = list(partition_labels)
    elif auto_partition and len(qpd_bases) > 0:
        # When labels=None, partition_problem infers connectivity.
        pp = partition_problem(circuit=cut_circuit)
        subcircuits = dict(pp.subcircuits)
        resolved_labels = (
            list(pp.partition_labels)
            if getattr(pp, "partition_labels", None) is not None
            else None
        )

    if partition_labels is not None:
        pp = partition_problem(
            circuit=cut_circuit, partition_labels=list(partition_labels)
        )
        subcircuits = dict(pp.subcircuits)

    # ---- 6. Build structured result ------------------------------------
    return WireCuttingResult(
        original_circuit=original_snapshot,
        cut_circuit=cut_circuit,
        marked_circuit=marked_circuit,
        qpd_bases=qpd_bases,
        cut_locations=norm_cuts,
        cut_info=cut_info,
        total_sampling_overhead=total_overhead,
        partition_labels=resolved_labels,
        subcircuits=subcircuits,
        metadata={
            "num_cuts": len(qpd_bases),
            "num_qubits_original": work_circuit.num_qubits,
            "num_qubits_cut": cut_circuit.num_qubits,
            "num_instructions_original": len(work_circuit.data),
            "num_instructions_cut": len(cut_circuit.data),
        },
    )


# 5.  Standalone demo (mirrors gate_cutting_node's __main__ block)

if __name__ == "__main__":
    # Demo circuit: GHZ-style 3-qubit circuit identical to the included
    # test.qasm file used to verify the gate-cutting node.
    demo_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
"""

    # Show every plausible wire-cut location
    candidates = list_cuttable_wires(demo_qasm)
    print("Candidate wire-cut locations:")
    for c in candidates:
        print(f"  {c}")

    # Estimate overhead for cutting qubit 1 between the two CX gates.
    # In the stripped circuit, instructions are:
    #   0: h q[0]
    #   1: cx q[0], q[1]
    #   2: cx q[1], q[2]
    # so "after_instruction=1, qubit=1" inserts a wire cut on q[1] between
    # the two CX gates.
    proposed = [{"qubit": 1, "after_instruction": 1}]

    overhead_info = estimate_wire_cut_overhead(demo_qasm, proposed)
    print(f"\nOverhead preview (cutting qubit 1 after CX #1): {overhead_info}")

    # Execute the node
    result = wire_cutting_node(demo_qasm, cut_locations=proposed)

    print(f"\nOriginal circuit ({result.original_circuit.num_qubits} qubits):")
    print(result.original_circuit.draw(output="text"))

    print(f"\nMarked circuit ({result.marked_circuit.num_qubits} qubits, with CutWire markers):")
    print(result.marked_circuit.draw(output="text"))

    print(f"\nCut circuit ({result.cut_circuit.num_qubits} qubits, expanded):")
    print(result.cut_circuit.draw(output="text"))

    print(f"\nNumber of QPD bases: {len(result.qpd_bases)}")
    for i, info in enumerate(result.cut_info):
        print(f"  Cut {i}: {info}")

    print(f"\nTotal sampling overhead: {result.total_sampling_overhead}")
    print(f"Metadata: {result.metadata}")

    # Demonstrate with auto-partitioning
    result_partitioned = wire_cutting_node(
        demo_qasm, cut_locations=proposed, auto_partition=True
    )
    if result_partitioned.subcircuits:
        print(
            f"\nAuto-partitioned into {len(result_partitioned.subcircuits)} subcircuits:"
        )
        for label, sub in result_partitioned.subcircuits.items():
            print(f"  Partition {label!r}: {sub.num_qubits} qubits, "
                  f"{len(sub.data)} instructions")
