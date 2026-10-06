"""
auto_finder.py
==============

AutoFinder node for an agentic quantum circuit-cutting workflow.

This module implements Node 4 (`AutoFinder`) of a LangGraph-based pipeline for
automated quantum circuit cutting. It wraps `find_cuts()` from the Qiskit Addon
for Circuit Cutting (`qiskit_addon_cutting`) and provides:

    * A configurable, type-hinted interface that exposes the major optional
      parameters of `find_cuts()` (qubits-per-subcircuit, max_gamma,
      max_backjumps, seed, gate_lo, wire_lo).
    * Flexible input handling: accepts either a path to a `.qasm` file
      (OpenQASM 2.0 or 3.0) or a `QuantumCircuit` object.
    * Production of partitioned subcircuits suitable for downstream
      reconstruction in a circuit-knitting workflow.
    * A structured, serializable result object containing the original circuit,
      the cut circuit, the subcircuits, cut metadata, and success/failure info.
    * Robust error handling and structured logging.
    * Clear separation between core logic (`run_autofinder`) and the
      LangGraph-facing wrapper (`autofinder_node`), so this file is fully
      usable as a standalone Python module today and trivially wrappable as a
      LangGraph node later.

Design notes on `find_cuts()`
-----------------------------
`find_cuts(circuit, optimization, constraints)` performs a best-first
(Dijkstra-style) search over candidate cut placements. At each search step it
considers cutting either a two-qubit gate (an LO "gate cut") or a wire (an LO
"wire cut", which introduces an auxiliary qubit since the cut-finder assumes
no qubit reuse), and it scores candidate cut schemes by their *sampling
overhead* gamma. The search terminates when it finds a scheme that satisfies
the `DeviceConstraints` (i.e. every resulting subcircuit fits within
`qubits_per_subcircuit`) while minimizing gamma, or when one of the user-
supplied termination criteria (`max_backjumps`, `max_gamma`) is hit. The
returned metadata's `minimum_reached` flag indicates whether the search was
exhaustive.

How the returned subcircuits are reconstructed
-----------------------------------------------
The `find_cuts()` call returns a circuit that contains `BaseQPDGate`
placeholder instructions at every chosen cut location. To go from this object
to executable subcircuits and back to an estimator of the original
observable, the standard circuit-knitting workflow is:

    1. `partition_problem(cut_circuit, partition_labels, observables)`
       -> splits the cut circuit into independent subcircuits and
          subobservables, one per partition label.
    2. `generate_cutting_experiments(subcircuits, subobservables, num_samples)`
       -> samples the quasi-probability decomposition of every cut and emits
          a set of concrete (no-QPD) experiments to run on hardware/simulator.
    3. Execute the experiments with a Sampler/Estimator primitive.
    4. `reconstruct_expectation_values(results, coefficients, subobservables)`
       -> classically post-processes the per-subcircuit measurement outcomes,
          weighting them by the QPD coefficients to recover an unbiased
          estimate of the original (uncut) circuit's expectation value. The
          shot budget needed grows as O(gamma^2), which is exactly why the
          cut-finder optimizes for minimum gamma.

This module performs step (1) so that the next nodes in the LangGraph
pipeline (e.g. `Validate`) can immediately reason about the partitioned
subcircuits without re-running the cut-finder.
"""

from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Hashable, Mapping, Optional, Union

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
# A module-level logger so consumers (including LangGraph) can attach handlers
# without us forcing a configuration on them.
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Provide a sensible default so the module is usable standalone, but do
    # not override a configuration the host application may have set.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def autofinder_langgraph_node(state):

    if state.get("error"):
        return state

    result = run_autofinder(
        state["circuit"]
    )

    if not result.success:
        # every downstream node early-returns on error, so the Validate
        # node's failed-strategy retry loop can never reroute and the run
        # dies as PIPELINE_ERROR (observed on ccx-containing circuits,
        # where find_cuts() rejects gates wider than 2 qubits). Treat it
        # as a failed strategy instead: clear the cut outputs, leave
        # error unset, and let Validate mark auto_finder failed so the
        # Strategy_router retries with an alternative technique.
        print(f"[AUTO_FINDER] failed: {result.error} "
              f"-> marking strategy failed so the router can retry")
        return {
            **state,
            "original_circuit": result.original_circuit,
            "cut_circuit": None,
            "subcircuits": None,
            "metadata": None,
            "error": None,
        }

    # ---- No-cut detection ------------------------------------------------ #
    # find_cuts() may "succeed" by deciding the circuit already fits within
    # device constraints, placing zero QPD gates.  Detect this by checking
    # for BaseQPDGate instructions in the returned circuit.
    from qiskit_addon_cutting.qpd import BaseQPDGate
    has_qpd = any(
        isinstance(inst.operation, BaseQPDGate)
        for inst in result.cut_circuit.data
    )

    metadata = dict(result.cut_metadata)

    if not has_qpd:
        metadata["no_cut_note"] = (
            "NO CUTS MADE: AutoFinder determined no cuts are needed "
            "(circuit already fits the configured constraints). "
            "Accepted as a legitimate no-cut outcome; the 'subcircuits' "
            "are the original circuit, not a cut decomposition."
        )
        return {
            **state,
            "original_circuit":      result.original_circuit,
            "cut_circuit":           result.cut_circuit,
            "subcircuits":           result.subcircuits,
            "metadata":              metadata,
            "minimum_reached":       metadata.get("minimum_reached"),
            "total_sampling_overhead": 1.0,
            "no_cut":                True,
            "error":                 None,
        }

    return {
        **state,
        "original_circuit":
            result.original_circuit,

        "cut_circuit":
            result.cut_circuit,

        "subcircuits":
            result.subcircuits,

        "metadata":
            metadata,

        "minimum_reached":
            metadata.get("minimum_reached"),

        "error":
            None,
    }

# --------------------------------------------------------------------------- #
# Lazy / guarded imports of the quantum stack                                  #
# --------------------------------------------------------------------------- #
# We import qiskit + qiskit_addon_cutting at module load time, but wrap them in
# a try/except so that import failures produce an actionable error message
# rather than a cryptic ModuleNotFoundError deep in a LangGraph trace.
try:
    from qiskit import QuantumCircuit
    from qiskit.qasm2 import load as qasm2_load
    from qiskit.qasm2 import QASM2ParseError

    try:
        # QASM3 support is optional in some Qiskit builds.
        from qiskit.qasm3 import load as qasm3_load  # type: ignore
    except Exception:  # pragma: no cover - depends on Qiskit build
        qasm3_load = None  # type: ignore

    from qiskit_addon_cutting import find_cuts, partition_problem
    from qiskit_addon_cutting.automated_cut_finding import (
        DeviceConstraints,
        OptimizationParameters,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "auto_finder.py requires `qiskit` and `qiskit-addon-cutting`. "
        "Install them with:\n"
        "    pip install qiskit qiskit-addon-cutting\n"
        f"Original import error: {exc}"
    ) from exc


# --------------------------------------------------------------------------- #
# Public data types                                                            #
# --------------------------------------------------------------------------- #
CircuitInput = Union[str, os.PathLike, "QuantumCircuit"]


@dataclass
class AutoFinderConfig:
    """User-tunable configuration for the AutoFinder node.

    All fields map directly onto `find_cuts()`'s `DeviceConstraints` and
    `OptimizationParameters` so that the full configurability of the
    underlying API is preserved.

    Attributes
    ----------
    qubits_per_subcircuit:
        Hard upper bound on the qubit width of each resulting subcircuit.
        This is the single most important knob: it determines the target
        device size that the cut scheme must respect.
    max_gamma:
        Upper bound on the sampling overhead gamma. If the search cannot
        find a scheme with gamma <= max_gamma, it terminates without a
        solution. Larger values allow more (and more expensive) cuts.
    max_backjumps:
        Cap on the number of best-first-search backjumps. `None` removes
        the cap and lets the search run to completion (which is required
        to *guarantee* optimality).
    seed:
        Seed for the NumPy RNG used inside the priority queue, for
        reproducibility.
    gate_lo:
        Whether the search may place LO gate cuts.
    wire_lo:
        Whether the search may place LO wire cuts. (Each wire cut adds a
        new qubit, since the cut-finder assumes no qubit reuse.)
    partition_observables:
        Optional observables to pass through to `partition_problem` so the
        returned partitioned problem also contains subobservables. If
        omitted, only the subcircuits are produced (sufficient for the
        next pipeline stages, which can add observables later).
    """

    qubits_per_subcircuit: int = 4
    # to ~1e6 - never validatable. 81 caps overhead at ~6561 (the cost of
    # 4 CX gate cuts); if find_cuts cannot meet it, the node fails the
    # strategy and the router reroutes.
    max_gamma: float = 81.0
    max_backjumps: Optional[int] = 10_000
    # best-first search unseeded and stochastic across runs. That
    # nondeterminism caused the same circuit to flip between PASS and
    # PASS_NO_CUT (and shifted per-circuit strategy attribution) between
    # otherwise-identical batches. A fixed default makes find_cuts()
    # reproducible so LLM-vs-baseline comparisons and repeated trials are
    # controlled. Override via AutoFinderConfig(seed=...) or the
    # GRACE_AUTOFINDER_SEED env var (see below) to run multiple *distinct*
    # seeded trials for mean/SD reporting.
    seed: Optional[int] = int(os.environ.get("GRACE_AUTOFINDER_SEED", "12345"))
    gate_lo: bool = True
    wire_lo: bool = True
    partition_observables: Optional[Any] = None  # PauliList | str | None

    def validate(self) -> None:
        """Light sanity-checking of user input."""
        if self.qubits_per_subcircuit < 1:
            raise ValueError("qubits_per_subcircuit must be >= 1.")
        if self.max_gamma <= 0:
            raise ValueError("max_gamma must be positive.")
        if self.max_backjumps is not None and self.max_backjumps < 0:
            raise ValueError("max_backjumps must be >= 0 or None.")
        if not (self.gate_lo or self.wire_lo):
            raise ValueError(
                "At least one of gate_lo or wire_lo must be True; "
                "otherwise the cut-finder has no moves available."
            )


@dataclass
class AutoFinderResult:
    """Structured output of the AutoFinder node.

    This object is intentionally JSON-light: `QuantumCircuit` objects are
    kept as live Python references (downstream LangGraph nodes need them),
    while `cut_metadata` and `info` are plain dicts. Use :meth:`summary` for
    a logging-friendly representation.
    """

    success: bool
    original_circuit: Optional["QuantumCircuit"] = None
    cut_circuit: Optional["QuantumCircuit"] = None
    subcircuits: Dict[Hashable, "QuantumCircuit"] = field(default_factory=dict)
    cut_metadata: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        """Return a small, log-friendly dict (no QuantumCircuit objects)."""
        return {
            "success": self.success,
            "num_qubits_original": (
                self.original_circuit.num_qubits if self.original_circuit else None
            ),
            "depth_original": (
                self.original_circuit.depth() if self.original_circuit else None
            ),
            "num_qubits_cut": (
                self.cut_circuit.num_qubits if self.cut_circuit else None
            ),
            "depth_cut": self.cut_circuit.depth() if self.cut_circuit else None,
            "num_subcircuits": len(self.subcircuits),
            "subcircuit_widths": {
                str(k): v.num_qubits for k, v in self.subcircuits.items()
            },
            "cut_metadata": self.cut_metadata,
            "info": self.info,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #
def _load_circuit(circuit_input: CircuitInput) -> "QuantumCircuit":
    """Load a `QuantumCircuit` from either a path or a live object.

    Supports OpenQASM 2.0 (`.qasm`) and OpenQASM 3.0 (`.qasm3`/`.qasm`) by
    trying QASM2 first and falling back to QASM3 on failure -- this matches
    the common case where the file extension is just `.qasm` regardless of
    the dialect.
    """
    if isinstance(circuit_input, QuantumCircuit):
        logger.debug("Received a live QuantumCircuit (no file I/O needed).")
        return circuit_input

    if isinstance(circuit_input, (str, os.PathLike)):
        path = Path(circuit_input)
        if not path.is_file():
            raise FileNotFoundError(f"QASM file not found: {path}")
        logger.info("Loading circuit from QASM file: %s", path)

        # Try QASM2 first.
        try:
            return qasm2_load(str(path))
        except QASM2ParseError as qasm2_err:
            logger.debug("QASM2 parse failed (%s); attempting QASM3.", qasm2_err)
            if qasm3_load is None:
                raise ValueError(
                    f"Failed to parse {path} as OpenQASM 2.0, and the "
                    "installed Qiskit build does not provide QASM3 support."
                ) from qasm2_err
            try:
                return qasm3_load(str(path))
            except Exception as qasm3_err:
                raise ValueError(
                    f"Failed to parse {path} as either OpenQASM 2.0 or 3.0. "
                    f"QASM2 error: {qasm2_err}. QASM3 error: {qasm3_err}."
                ) from qasm3_err

    raise TypeError(
        "circuit_input must be a QuantumCircuit or a path to a .qasm file, "
        f"got {type(circuit_input).__name__}."
    )


def _strip_classical_bits(circuit: "QuantumCircuit") -> "QuantumCircuit":
    """Return a copy of `circuit` with all classical registers/bits removed.

    `find_cuts()` (via `qiskit_addon_cutting.cut_gates`) hard-requires the
    input circuit to contain *no* classical registers or bits at all --
    not even unused ones left over from terminal measurements. Circuits
    produced by `parse_circuit_node` (and most QASM files people actually
    have lying around) include final measurements, so without this step
    `find_cuts()` raises:

        "Circuits input to cut_gates should contain no classical
         registers or bits."

    Strategy
    --------
    1. Try `circuit.remove_final_measurements(inplace=False)`. This is the
       Qiskit-native way to drop trailing `measure` instructions and then
       prune any classical register/bit no longer referenced by anything.
       It covers the overwhelmingly common case (measurements only at the
       end of the circuit, which is what `parse_circuit_node`/QASM files
       typically produce).
    2. If classical bits still remain afterward (e.g. mid-circuit
       measurements or classically-controlled gates), fall back to
       manually rebuilding a purely-quantum circuit, skipping any
       instruction that touches a classical bit or has a `condition`, and
       log a warning -- since silently dropping those instructions can
       change circuit semantics if they fed into classical control flow.
    """
    stripped = circuit.remove_final_measurements(inplace=False)

    # connectivity by partition_problem()'s automatic labelling (used in
    # _partition_subcircuits), which can glue otherwise-separated
    # partitions into one subcircuit spanning every qubit — the same
    # barrier bug fixed in gate_cutting_node / wire_cutting_node. find_cuts
    # itself doesn't need barriers either; they are visualization aids,
    # not gates.
    if any(inst.operation.name == "barrier" for inst in stripped.data):
        no_barrier = QuantumCircuit(*stripped.qregs, *stripped.cregs)
        for inst in stripped.data:
            if inst.operation.name == "barrier":
                continue
            no_barrier.append(inst.operation, inst.qubits, inst.clbits)
        stripped = no_barrier

    if stripped.num_clbits == 0:
        return stripped

    logger.warning(
        "Circuit still has %d classical bit(s) after "
        "remove_final_measurements() (likely mid-circuit measurements or "
        "classically-controlled gates). Rebuilding a purely-quantum "
        "circuit by dropping any instruction that touches a classical "
        "bit; find_cuts() will not see these operations.",
        stripped.num_clbits,
    )
    quantum_only = QuantumCircuit(*stripped.qregs)
    for instr in stripped.data:
        if instr.clbits or getattr(instr.operation, "condition", None) is not None:
            continue
        quantum_only.append(instr.operation, instr.qubits)
    return quantum_only


def _build_optimization(config: AutoFinderConfig) -> OptimizationParameters:
    """Translate `AutoFinderConfig` into a Qiskit `OptimizationParameters`."""
    return OptimizationParameters(
        seed=config.seed,
        max_gamma=config.max_gamma,
        max_backjumps=config.max_backjumps,
        gate_lo=config.gate_lo,
        wire_lo=config.wire_lo,
    )


def _build_constraints(config: AutoFinderConfig) -> DeviceConstraints:
    """Translate `AutoFinderConfig` into a Qiskit `DeviceConstraints`."""
    return DeviceConstraints(qubits_per_subcircuit=config.qubits_per_subcircuit)


def _partition_subcircuits(
    cut_circuit: "QuantumCircuit",
    observables: Optional[Any],
) -> Dict[Hashable, "QuantumCircuit"]:
    """Use `partition_problem` to produce reconstruction-ready subcircuits.

    Design decision: even though `find_cuts()` only returns a single
    `QuantumCircuit` containing `BaseQPDGate` placeholders, downstream
    nodes (Validate, reconstruction, hardware execution) need the
    *partitioned* subcircuits. We perform that partitioning here so that
    the AutoFinder's contract -- "produce subcircuits suitable for later
    reconstruction" -- is met inside this node, not implicitly deferred.

    If partitioning fails (e.g., because the cut scheme did not actually
    separate the circuit, which can happen if the search terminated early),
    we log a warning and return an empty dict rather than aborting; the
    `cut_circuit` itself is still returned and downstream nodes can
    decide how to recover.
    """
    try:
        if observables is not None:
            partitioned = partition_problem(
                circuit=cut_circuit, observables=observables
            )
        else:
            partitioned = partition_problem(circuit=cut_circuit)
        return dict(partitioned.subcircuits)
    except Exception as exc:
        logger.warning(
            "partition_problem() failed: %s. Returning cut_circuit without "
            "partitioned subcircuits.",
            exc,
        )
        return {}


# --------------------------------------------------------------------------- #
# Core logic                                                                   #
# --------------------------------------------------------------------------- #
def run_autofinder(
    circuit_input: CircuitInput,
    config: Optional[AutoFinderConfig] = None,
) -> AutoFinderResult:
    """Run the AutoFinder cut-finding pipeline on a single circuit.

    This is the *core* entry point. It is deliberately framework-agnostic so
    that it can be called from a script, a unit test, a notebook, or
    -- via :func:`autofinder_node` -- a LangGraph node.

    Parameters
    ----------
    circuit_input:
        Either a `QuantumCircuit` or a path to a `.qasm` file.
    config:
        AutoFinder configuration. If `None`, sensible defaults are used.

    Returns
    -------
    AutoFinderResult
        Structured result. `result.success` is the canonical pass/fail flag.
    """
    config = config or AutoFinderConfig()

    # Validate config eagerly so we fail fast with a clean message.
    try:
        config.validate()
    except ValueError as exc:
        logger.error("Invalid AutoFinder configuration: %s", exc)
        return AutoFinderResult(success=False, error=f"Config error: {exc}")

    # ----- Step 1: load the input circuit ---------------------------------- #
    try:
        circuit = _load_circuit(circuit_input)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        logger.error("Failed to load input circuit: %s", exc)
        return AutoFinderResult(success=False, error=f"Load error: {exc}")

    logger.info(
        "Loaded circuit: %d qubits, depth=%d, %d operations.",
        circuit.num_qubits,
        circuit.depth(),
        len(circuit.data),
    )

    # Keep the unmodified, as-loaded circuit around for export/provenance
    # (this is what gets written to original_circuit.qpy/.qasm downstream).
    original_circuit = circuit

    # ----- Step 1b: strip classical registers/bits ------------------------ #
    # find_cuts() hard-requires a purely-quantum circuit. Circuits coming
    # out of parse_circuit_node (and most real QASM) carry final
    # measurements/classical registers, so we strip them here rather than
    # pushing that responsibility onto every caller. This is the fix for
    # the "Circuits input to cut_gates should contain no classical
    # registers or bits" crash.
    if circuit.num_clbits > 0:
        circuit = _strip_classical_bits(circuit)
        logger.info(
            "Stripped classical bits from input circuit before find_cuts() "
            "(was %d qubits/%d clbits, now %d qubits/%d clbits).",
            original_circuit.num_qubits,
            original_circuit.num_clbits,
            circuit.num_qubits,
            circuit.num_clbits,
        )

    # find_cuts() requires every gate to be at most two-qubit; surface a
    # clear error up-front rather than letting it explode inside the search.
    max_gate_width = max((len(instr.qubits) for instr in circuit.data), default=0)
    if max_gate_width > 2:
        msg = (
            f"find_cuts() requires gates of width <= 2, but the input "
            f"circuit contains a {max_gate_width}-qubit gate. Decompose "
            "or transpile before invoking AutoFinder."
        )
        logger.error(msg)
        return AutoFinderResult(
            success=False, original_circuit=original_circuit, error=msg
        )

    # ----- Step 2: build optimization + constraints ------------------------ #
    optimization = _build_optimization(config)
    constraints = _build_constraints(config)
    logger.info(
        "Running find_cuts() with qubits_per_subcircuit=%d, max_gamma=%.1f, "
        "max_backjumps=%s, gate_lo=%s, wire_lo=%s, seed=%s.",
        config.qubits_per_subcircuit,
        config.max_gamma,
        config.max_backjumps,
        config.gate_lo,
        config.wire_lo,
        config.seed,
    )

    # ----- Step 3: invoke find_cuts() -------------------------------------- #
    try:
        cut_circuit, metadata = find_cuts(
            circuit=circuit,
            optimization=optimization,
            constraints=constraints,
        )
    except Exception as exc:
        # We catch broadly here because the underlying search can raise a
        # variety of errors (ValueError for infeasible problems, RuntimeError
        # on internal failures, etc.). We surface the full traceback in
        # `info` for debugging but keep `error` short for routing logic.
        logger.error("find_cuts() raised an exception: %s", exc)
        return AutoFinderResult(
            success=False,
            original_circuit=original_circuit,
            error=f"find_cuts() failed: {exc}",
            info={"traceback": traceback.format_exc()},
        )

    # Normalize metadata into a plain dict for safe serialization.
    metadata_dict: Dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {
        "raw_metadata": repr(metadata)
    }
    logger.info("find_cuts() succeeded. Metadata: %s", metadata_dict)

    # If the search timed out (minimum_reached == False), this is *not* a
    # hard failure -- the returned solution may still be optimal -- but we
    # surface it prominently so the Validate node can choose to re-route.
    minimum_reached = metadata_dict.get("minimum_reached", None)
    if minimum_reached is False:
        logger.warning(
            "find_cuts() terminated before proving optimality "
            "(minimum_reached=False). Consider increasing max_backjumps."
        )

    # ----- Step 4: partition into reconstruction-ready subcircuits --------- #
    subcircuits = _partition_subcircuits(
        cut_circuit, observables=config.partition_observables
    )
    logger.info("Produced %d subcircuit(s) via partition_problem().", len(subcircuits))

    # ----- Step 5: assemble the structured result -------------------------- #
    return AutoFinderResult(
        success=True,
        original_circuit=original_circuit,
        cut_circuit=cut_circuit,
        subcircuits=subcircuits,
        cut_metadata=metadata_dict,
        info={
            "config": asdict(config),
            "minimum_reached": minimum_reached,
        },
    )


# --------------------------------------------------------------------------- #
# LangGraph-facing wrapper                                                     #
# --------------------------------------------------------------------------- #
# The wrapper is intentionally thin. LangGraph nodes receive and return a
# state dict; the only job of this function is to extract our inputs from the
# state, call `run_autofinder`, and merge the result back into the state. By
# keeping `run_autofinder` pure (state-free), we keep the module trivially
# testable and reusable outside LangGraph.

# Conventional keys used by the surrounding graph. Centralizing them as
# module-level constants makes it easy to align with the other nodes.
STATE_KEY_CIRCUIT_INPUT = "circuit_input"     # path or QuantumCircuit
STATE_KEY_CONFIG = "autofinder_config"        # AutoFinderConfig or dict
STATE_KEY_RESULT = "autofinder_result"        # AutoFinderResult


def autofinder_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph-compatible wrapper around :func:`run_autofinder`.

    Reads from ``state``:
        - ``circuit_input``: a path to a ``.qasm`` file or a ``QuantumCircuit``.
        - ``autofinder_config`` (optional): an ``AutoFinderConfig`` or a plain
          dict of overrides.

    Writes back into ``state``:
        - ``autofinder_result``: the ``AutoFinderResult`` produced.
        - For convenience and consistency with downstream nodes, also mirrors
          ``cut_circuit`` and ``subcircuits`` into the top-level state.

    Returns the updated state dictionary so it can be returned directly from a
    LangGraph node callable.
    """
    circuit_input = state.get(STATE_KEY_CIRCUIT_INPUT)
    if circuit_input is None:
        logger.error("State is missing required key '%s'.", STATE_KEY_CIRCUIT_INPUT)
        state[STATE_KEY_RESULT] = AutoFinderResult(
            success=False,
            error=f"Missing state key '{STATE_KEY_CIRCUIT_INPUT}'.",
        )
        return state

    raw_config = state.get(STATE_KEY_CONFIG)
    if isinstance(raw_config, AutoFinderConfig):
        config = raw_config
    elif isinstance(raw_config, Mapping):
        config = AutoFinderConfig(**dict(raw_config))
    elif raw_config is None:
        config = AutoFinderConfig()
    else:
        logger.error(
            "Unsupported type for '%s': %s",
            STATE_KEY_CONFIG,
            type(raw_config).__name__,
        )
        state[STATE_KEY_RESULT] = AutoFinderResult(
            success=False,
            error=f"Invalid '{STATE_KEY_CONFIG}' type: {type(raw_config).__name__}.",
        )
        return state

    result = run_autofinder(circuit_input=circuit_input, config=config)
    state[STATE_KEY_RESULT] = result

    # Mirror the most-used fields up to the top of the state so the next
    # nodes (Validate, etc.) don't need to know AutoFinder's internal shape.
    if result.success:
        state["cut_circuit"] = result.cut_circuit
        state["subcircuits"] = result.subcircuits
        state["cut_metadata"] = result.cut_metadata

    return state


# --------------------------------------------------------------------------- #
# Minimal self-test / demo                                                     #
# --------------------------------------------------------------------------- #
# Running `python auto_finder.py` exercises the module on a small 4-qubit GHZ-
# like circuit so you can confirm the install + wiring are correct without
# needing the rest of the LangGraph pipeline.
if __name__ == "__main__":  # pragma: no cover
    demo = QuantumCircuit(4)
    demo.h(0)
    demo.cx(0, 1)
    demo.cx(1, 2)
    demo.cx(2, 3)
    demo.cx(0, 3)  # long-range gate -- a natural cut candidate

    cfg = AutoFinderConfig(
        qubits_per_subcircuit=2,
        max_gamma=64.0,
        max_backjumps=2_000,
        seed=42,
    )

    result = run_autofinder(demo, cfg)
    print("\n=== AutoFinder summary ===")
    for k, v in result.summary().items():
        print(f"{k}: {v}")
