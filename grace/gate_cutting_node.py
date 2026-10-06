"""
gate_cutting_node.py
====================

A standalone QPD-based gate-cutting node for agentic quantum circuit cutting
pipelines.  Designed for direct use *and* trivial integration into LangGraph.

Requires:
    pip install 'qiskit-addon-cutting>=0.10' qiskit

Author : Ezekiel Laney
License: Apache-2.0
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import CircuitInstruction
from qiskit_addon_cutting import (
    cut_gates,
    partition_problem,
    generate_cutting_experiments,
    reconstruct_expectation_values,
)
from qiskit_addon_cutting.qpd import QPDBasis
from qiskit_addon_cutting.qpd.instructions import TwoQubitQPDGate


# 1.  Data containers

@dataclass
class GateCuttingResult:
    """Structured output of the gate-cutting node.

    Attributes
    ----------
    original_circuit : QuantumCircuit
        The circuit exactly as it was received (before any modification).
    cut_circuit : QuantumCircuit
        The circuit after the selected gates have been replaced with
        ``TwoQubitQPDGate`` placeholders.
    qpd_bases : list[QPDBasis]
        One ``QPDBasis`` per cut gate, in the same order as ``gate_ids``.
        Each basis encodes the quasi-probability decomposition (maps,
        coefficients, sampling overhead) for that gate.
    gate_ids : list[int]
        The instruction indices that were cut (echoed back for traceability).
    gate_info : list[dict[str, Any]]
        Human-readable metadata for every cut gate (name, parameters,
        qubit indices, kappa, overhead).
    total_sampling_overhead : float
        Product of per-gate overheads.  This is the multiplicative factor
        by which the total shot budget must increase to maintain the same
        statistical precision as the uncut circuit.
    partition_labels : list[int | None] | None
        If automatic partitioning was requested, the labels assigned to
        each qubit.  ``None`` otherwise.
    subcircuits : dict | None
        If ``partition_problem`` was invoked, the resulting subcircuit
        dictionary.  ``None`` otherwise.
    metadata : dict[str, Any]
        Catch-all for downstream consumers (LangGraph state, logging, etc.).
    """

    original_circuit: QuantumCircuit
    cut_circuit: QuantumCircuit
    qpd_bases: list[QPDBasis]
    gate_ids: list[int]
    gate_info: list[dict[str, Any]]
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
            "qpd_bases": self.qpd_bases,
            "gate_ids": self.gate_ids,
            "gate_info": self.gate_info,
            "total_sampling_overhead": self.total_sampling_overhead,
            "partition_labels": self.partition_labels,
            "subcircuits": self.subcircuits,
            "metadata": self.metadata,
        }


# 2.  Helper utilities

def _load_circuit(source: str | QuantumCircuit) -> QuantumCircuit:
    """Accept an OpenQASM 2/3 string *or* a ``QuantumCircuit`` and return a
    ``QuantumCircuit``.

    Parameters
    ----------
    source : str | QuantumCircuit
        OpenQASM 2.0 string, OpenQASM 3.0 string, or an already-parsed
        ``QuantumCircuit``.

    Returns
    -------
    QuantumCircuit

    Raises
    ------
    TypeError
        If *source* is neither ``str`` nor ``QuantumCircuit``.
    ValueError
        If the QASM string cannot be parsed.
    """
    if isinstance(source, QuantumCircuit):
        return source.copy()

    if not isinstance(source, str):
        raise TypeError(
            f"Expected an OpenQASM string or QuantumCircuit, got {type(source).__name__}"
        )

    # Try OpenQASM 2 first (most common in research pipelines)
    try:
        return qasm2.loads(source)
    except Exception:
        pass

    # Fall back to OpenQASM 3
    try:
        return qasm3.loads(source)
    except Exception as exc:
        raise ValueError(
            "Could not parse the input as OpenQASM 2 or OpenQASM 3."
        ) from exc


def _strip_classical_bits(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of *circuit* with all classical registers/bits removed
    (measurements, resets, AND barriers stripped). See
    :func:`_strip_to_unitary_core` for why barriers must go too."""
    core, _ = _strip_to_unitary_core(circuit)
    return core


def _strip_to_unitary_core(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, dict[int, int]]:
    """Return (core_circuit, index_map) where *core_circuit* contains only
    the unitary gates of *circuit* and *index_map* maps original
    ``circuit.data`` indices -> core-circuit indices.

     barriers were kept in the work circuit.
    A multi-qubit barrier is a 2+-qubit *instruction*, so after cutting the
    entangling gates ``partition_problem``'s automatic labelling still saw
    the barrier as connectivity linking the qubits and lumped everything
    into ONE partition — i.e. the "cut" produced a single subcircuit
    identical in width to the original and no actual partitioning happened.
    Barriers are not gates; they must be stripped along with measurements.

     gate indices chosen upstream (Strategy_router's
    ``_find_two_qubit_gate_ids``) refer to the ORIGINAL circuit's
    ``.data``. Because measurements/barriers are removed here, those
    indices can point at the wrong instruction in the work circuit whenever
    a measurement or barrier appears *before* a target gate. The returned
    ``index_map`` lets the caller translate original indices to core
    indices instead of assuming they line up.
    """
    core = QuantumCircuit(*circuit.qregs)
    index_map: dict[int, int] = {}
    for orig_idx, inst in enumerate(circuit.data):
        if inst.operation.name in ("measure", "reset", "barrier"):
            continue
        if len(inst.clbits) != 0:
            continue
        index_map[orig_idx] = len(core.data)
        core.append(inst.operation, inst.qubits)
    return core, index_map


def _gate_metadata(
    circuit: QuantumCircuit,
    gate_id: int,
    basis: QPDBasis,
) -> dict[str, Any]:
    """Build a human-readable metadata dict for a single cut gate."""
    inst= circuit.data[gate_id]
    qubit_indices = [circuit.find_bit(q).index for q in inst.qubits]
    return {
        "gate_index": gate_id,
        "gate_name": inst.operation.name,
        "gate_params": [float(p) for p in inst.operation.params] if inst.operation.params else [],
        "qubit_indices": qubit_indices,
        "num_qpd_maps": len(basis.maps),
        "kappa": float(basis.kappa),
        "sampling_overhead": float(basis.overhead),
    }


# 3.  Validation

def _validate_gate_ids(
    circuit: QuantumCircuit,
    gate_ids: Sequence[int],
) -> None:
    """Raise informative errors if any gate index is invalid or points to a
    gate that cannot be decomposed via QPD.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit whose gates are being referenced.
    gate_ids : Sequence[int]
        Indices into ``circuit.data``.

    Raises
    ------
    IndexError
        If a gate index is out of range.
    ValueError
        If a gate is not a two-qubit gate (QPD requires exactly 2 qubits).
    """
    n = len(circuit.data)
    for gid in gate_ids:
        if gid < 0 or gid >= n:
            raise IndexError(
                f"gate_id {gid} is out of range for a circuit with "
                f"{n} instructions (valid range: 0..{n - 1})."
            )
        inst = circuit.data[gid]
        num_qubits = len(inst.qubits)
        if num_qubits != 2:
            raise ValueError(
                f"Gate at index {gid} ('{inst.operation.name}') acts on "
                f"{num_qubits} qubit(s), but QPD gate cutting requires "
                f"exactly 2-qubit gates."
            )


# 4.  Core node function

def gate_cutting_node(
    circuit_input: str | QuantumCircuit,
    gate_ids: Sequence[int],
    *,
    partition_labels: Sequence[int | None] | None = None,
    auto_partition: bool = False,
) -> GateCuttingResult:
    """Perform QPD-based gate cutting on specified gates of a quantum circuit.

    This is the single entry-point designed for both standalone use and
    LangGraph integration.

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        An OpenQASM 2/3 string **or** a Qiskit ``QuantumCircuit``.
    gate_ids : Sequence[int]
        Indices (into ``circuit.data``) of the two-qubit gates to cut.
        Use ``enumerate(circuit.data)`` to inspect the instruction list.
    partition_labels : Sequence[int | None] | None, optional
        Explicit partition labels for each qubit.  If provided together
        with *auto_partition=True*, the explicit labels take precedence.
    auto_partition : bool, default False
        If ``True`` **and** *partition_labels* is ``None``, call
        ``partition_problem`` to automatically separate the cut circuit
        into subcircuits.

    Returns
    -------
    GateCuttingResult
        A structured result containing the original circuit, cut circuit,
        QPD bases, gate metadata, and (optionally) subcircuits.

    Raises
    ------
    TypeError
        If *circuit_input* is not a supported type.
    ValueError
        If the QASM string cannot be parsed, or a gate index is invalid.
    IndexError
        If a gate index is out of range.

    Examples
    --------
    >>> qasm = '''
    ... OPENQASM 2.0;
    ... include "qelib1.inc";
    ... qreg q[4];
    ... h q[0];
    ... cx q[0], q[1];
    ... cx q[2], q[3];
    ... '''
    >>> result = gate_cutting_node(qasm, gate_ids=[1])
    >>> print(result.total_sampling_overhead)
    9.0
    """

    # ---- Step 1: Load & preserve the original circuit --------------------
    original_circuit = _load_circuit(circuit_input)

    # ---- Step 2: Prepare a classical-bit-free copy for cut_gates ---------
    work_circuit, index_map = _strip_to_unitary_core(original_circuit)

    # ---- Step 2b: Remap gate indices from the original circuit to the
    #      stripped work circuit (see _strip_to_unitary_core docstring).
    #      Indices that already refer to the work circuit (legacy callers)
    #      are accepted as-is when they can't be interpreted as original
    #      indices pointing at a removed instruction.
    remapped: list[int] = []
    for gid in gate_ids:
        if gid in index_map:
            remapped.append(index_map[gid])
        else:
            # Original index points at a stripped instruction (measure/
            # barrier/reset) or is out of range for the original circuit;
            # fall back to treating it as a work-circuit index so existing
            # callers that pre-stripped are not broken. _validate_gate_ids
            # below will still reject anything invalid.
            remapped.append(gid)
    gate_ids = remapped

    # ---- Step 3: Validate gate indices -----------------------------------
    _validate_gate_ids(work_circuit, gate_ids)

    # ---- Step 4: Apply QPD gate cutting ----------------------------------
    #
    # ``cut_gates`` replaces each selected gate with a ``TwoQubitQPDGate``
    # and returns the corresponding ``QPDBasis`` objects.
    #
    # Under the hood it calls ``QPDBasis.from_instruction(gate)`` which
    # dispatches to the decomposition registry in
    # ``qiskit_addon_cutting.qpd.decompositions``.  For standard gates
    # (CX, CZ, RZZ, …) an analytic decomposition is used; for arbitrary
    # unitaries a KAK (Weyl) decomposition is performed automatically.
    #
    cut_circuit, qpd_bases = cut_gates(
        work_circuit,
        gate_ids=list(gate_ids),
        inplace=False,
    )

    # ---- Step 5: Collect per-gate metadata -------------------------------
    gate_info = [
        _gate_metadata(work_circuit, gid, basis)
        for gid, basis in zip(gate_ids, qpd_bases)
    ]

    total_overhead = 1.0
    for basis in qpd_bases:
        total_overhead *= basis.overhead

    # ---- Step 6 (optional): Partition into subcircuits -------------------
    resolved_labels: list[int | None] | None = None
    subcircuits: dict | None = None

    if partition_labels is not None or auto_partition:
        labels_arg = (
            list(partition_labels) if partition_labels is not None else None
        )
        partitioned = partition_problem(
            circuit=cut_circuit,
            partition_labels=labels_arg,
        )
        subcircuits = dict(partitioned.subcircuits)
        # Recover the labels that were actually used
        if labels_arg is not None:
            resolved_labels = labels_arg
        else:
            # partition_problem infers labels; we can reconstruct them
            # from the subcircuit keys
            resolved_labels = None  # auto-detected, stored in subcircuits keys

    # ---- Step 7: Assemble result -----------------------------------------
    return GateCuttingResult(
        original_circuit=original_circuit,
        cut_circuit=cut_circuit,
        qpd_bases=qpd_bases,
        gate_ids=list(gate_ids),
        gate_info=gate_info,
        total_sampling_overhead=total_overhead,
        partition_labels=resolved_labels,
        subcircuits=subcircuits,
        metadata={
            "num_cuts": len(gate_ids),
            "num_qubits": work_circuit.num_qubits,
            "num_instructions": len(work_circuit.data),
        },
    )


# 5.  Convenience helpers for exploration

def list_cuttable_gates(circuit_input: str | QuantumCircuit) -> list[dict[str, Any]]:
    """Return metadata for every two-qubit gate in the circuit.

    Useful for letting an LLM agent (or a human) decide *which* gates to
    cut.

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        OpenQASM string or ``QuantumCircuit``.

    Returns
    -------
    list[dict]
        One dict per two-qubit gate, containing ``index``, ``name``,
        ``params``, and ``qubit_indices``.
    """
    circuit = _load_circuit(circuit_input)
    # NOTE: indices are reported in the frame of the circuit AS PASSED IN
    # (not a stripped copy). gate_cutting_node() now remaps incoming
    # gate_ids from the input circuit to its internal barrier/measure-free
    # work circuit, so callers (LLM planner, router, humans) can use these
    # indices directly without worrying about stripping offsets.
    result = []
    for i, inst in enumerate(circuit.data):
        if inst.operation.name in ("barrier", "measure", "reset"):
            continue
        if len(inst.qubits) == 2 and len(inst.clbits) == 0:
            result.append({
                "index": i,
                "name": inst.operation.name,
                "params": [float(p) for p in inst.operation.params] if inst.operation.params else [],
                "qubit_indices": [circuit.find_bit(q).index for q in inst.qubits],
            })
    return result


def preview_overhead(circuit_input: str | QuantumCircuit, gate_ids: Sequence[int]) -> dict[str, Any]:
    """Estimate the sampling overhead *without* modifying the circuit.

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        OpenQASM string or ``QuantumCircuit``.
    gate_ids : Sequence[int]
        Indices of the gates to (hypothetically) cut.

    Returns
    -------
    dict
        Per-gate and total overhead information.
    """
    circuit = _load_circuit(circuit_input)
    circuit, index_map = _strip_to_unitary_core(circuit)
    # Same index contract as gate_cutting_node(): gate_ids refer to the
    # input circuit; remap them onto the stripped core.
    gate_ids = [index_map.get(gid, gid) for gid in gate_ids]
    _validate_gate_ids(circuit, gate_ids)

    per_gate = []
    total = 1.0
    for gid in gate_ids:
        inst = circuit.data[gid]
        basis = QPDBasis.from_instruction(inst.operation)
        overhead = float(basis.overhead)
        total *= overhead
        per_gate.append({
            "gate_index": gid,
            "gate_name": inst.operation.name,
            "kappa": float(basis.kappa),
            "overhead": overhead,
        })

    return {
        "per_gate": per_gate,
        "total_sampling_overhead": total,
        "note": (
            "Total shots needed ≈ original_shots × total_sampling_overhead "
            "to maintain the same statistical precision."
        ),
    }


# 6.  LangGraph adapter (thin wrapper)

def gate_cutting_langgraph_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph-compatible node wrapper.

    Expects the following keys in *state*:

    - ``"circuit"`` : str | QuantumCircuit  (required)
    - ``"gate_ids"`` : list[int]            (required)
    - ``"partition_labels"`` : list | None   (optional)
    - ``"auto_partition"`` : bool            (optional, default False)

    Returns a dict that can be merged directly into the LangGraph state.
    """
    result = gate_cutting_node(
        circuit_input=state["circuit"],
        gate_ids=state["gate_ids"],
        partition_labels=state.get("partition_labels"),
        auto_partition=state.get("auto_partition", False),
    )
    #return result.to_dict() -> chat is yapping about removing this bring back if it breaks
    return{
        **state,
        **result.to_dict(),
    }


# 7.  Self-test / demo

if __name__ == "__main__":
    # A small 4-qubit circuit with two CX gates bridging two pairs
    demo_qasm = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
cx q[0], q[1];
h q[2];
cx q[2], q[3];
cx q[1], q[2];
"""

    print("=" * 60)
    print("Gate Cutting Node — Self-Test")
    print("=" * 60)

    # List cuttable gates
    cuttable = list_cuttable_gates(demo_qasm)
    print("\nCuttable two-qubit gates:")
    for g in cuttable:
        print(f"  index={g['index']}  name={g['name']}  qubits={g['qubit_indices']}")

    # Preview overhead for cutting the bridge gate (index 4)
    overhead_info = preview_overhead(demo_qasm, gate_ids=[4])
    print(f"\nOverhead preview (cutting gate 4): {overhead_info}")

    # Execute the node
    result = gate_cutting_node(demo_qasm, gate_ids=[4])

    print(f"\nOriginal circuit ({result.original_circuit.num_qubits} qubits):")
    print(result.original_circuit.draw(output="text"))

    print(f"\nCut circuit ({result.cut_circuit.num_qubits} qubits):")
    print(result.cut_circuit.draw(output="text"))

    print(f"\nNumber of QPD bases: {len(result.qpd_bases)}")
    for i, info in enumerate(result.gate_info):
        print(f"  Cut {i}: {info}")

    print(f"\nTotal sampling overhead: {result.total_sampling_overhead}")
    print(f"Metadata: {result.metadata}")

    # Demonstrate with auto-partitioning
    result_partitioned = gate_cutting_node(
        demo_qasm, gate_ids=[4], auto_partition=True
    )
    if result_partitioned.subcircuits:
        print(f"\nAuto-partitioned into {len(result_partitioned.subcircuits)} subcircuits:")
        for label, subcirc in result_partitioned.subcircuits.items():
            print(f"  Partition '{label}': {subcirc.num_qubits} qubits")
            print(subcirc.draw(output="text"))

    print("\n✓ Self-test complete.")
