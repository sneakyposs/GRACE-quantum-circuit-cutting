"""
inspection_export.py
=====================
Dumps the circuits and metadata produced by the cutting nodes
(gate_cutting_node.py / wire_cutting_node.py / auto_finder.py) to disk so
they can be inspected manually and checked for equivalence against the
original circuit.

Why qpy *and* (sometimes) QASM
-------------------------------
`cut_circuit` and the entries in `subcircuits` contain QPD placeholder
instructions (`TwoQubitQPDGate`, `SingleQubitQPDGate`, `CutWire`, `Move`).
These are perfectly normal Qiskit `Instruction` objects, but they are NOT
part of the OpenQASM 2/3 standard gate set, so `qasm2.dump` / `qasm3.dump`
will raise on them. To avoid silently producing broken or empty files:

  * Every circuit is always written as `.qpy` (Qiskit's native binary
    format). This is lossless and is the format to reload for any
    programmatic equivalence check (Statevector/Operator comparison,
    re-running through qiskit-addon-cutting, etc.).
  * A QASM (2, falling back to 3) dump is attempted *in addition*, purely
    as a human-readable convenience. Non-standard instructions sometimes
    come through fine as `opaque gate` declarations, but some QPD
    placeholder types can still fail to export; if that happens it's
    logged (not fatal) and the .qpy / plain-text `.draw()` dump still
    cover you.
  * A plain-text circuit diagram (`.draw(output="text")`) is always
    written too, since that's often the fastest way to eyeball whether a
    cut circuit "looks right" without opening Qiskit at all.

Usage
-----
    from inspection_export import export_cutting_results
    export_cutting_results(final_state)   # final_state = app.invoke(...)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qasm2, qasm3


# Helpers

def _sanitize(label: Any) -> str:
    """Turn an arbitrary (possibly tuple/hashable) partition label into a
    filesystem-safe string."""
    text = str(label)
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text) or "unnamed"


def _json_safe(obj: Any) -> Any:
    """Best-effort recursive conversion of arbitrary objects into something
    `json.dump` can handle. Anything it doesn't recognize is stringified
    rather than dropped, so nothing silently disappears from metadata.json.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, QuantumCircuit):
        # Circuits get their own files; in metadata.json just reference them.
        return f"<QuantumCircuit: {obj.num_qubits} qubits, {len(obj.data)} instructions>"
    # QPDBasis and similar objects: try to pull out the human-readable bits,
    # otherwise fall back to repr() so we never crash the export.
    for attrs in (("kappa", "overhead", "maps"),):
        if all(hasattr(obj, a) for a in attrs[:2]):
            out = {"kappa": float(obj.kappa), "overhead": float(obj.overhead)}
            try:
                out["num_maps"] = len(obj.maps)
            except Exception:
                pass
            return out
    try:
        return str(obj)
    except Exception:
        return repr(type(obj))


def _export_circuit(circuit: QuantumCircuit, base_path: Path) -> dict[str, Any]:
    """Write one circuit out as .qpy (always) + .txt diagram (always) +
    .qasm (best effort). Returns a small dict describing what was written,
    for the run summary.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "num_instructions": len(circuit.data),
        "depth": circuit.depth(),
        "files": {},
    }

    # 1. .qpy -- always succeeds, always lossless. This is the file to
    #    reload for any real equivalence test.
    from qiskit import qpy
    qpy_path = base_path.with_suffix(".qpy")
    with open(qpy_path, "wb") as fh:
        qpy.dump(circuit, fh)
    info["files"]["qpy"] = str(qpy_path)

    # 2. .txt -- always succeeds, fastest way to eyeball the circuit.
    txt_path = base_path.with_suffix(".txt")
    txt_path.write_text(str(circuit.draw(output="text", fold=120)), encoding="utf-8")
    info["files"]["txt"] = str(txt_path)

    # 3. .qasm -- best effort only. QPD placeholder instructions
    #    (TwoQubitQPDGate / SingleQubitQPDGate / CutWire / Move) are not
    #    part of the OpenQASM standard gate set, so this is expected to
    #    fail for cut_circuit / subcircuits and succeed for plain circuits
    #    like original_circuit.
    qasm_path = base_path.with_suffix(".qasm")
    try:
        qasm2.dump(circuit, qasm_path)
        info["files"]["qasm"] = str(qasm_path)
    except Exception:
        try:
            qasm3.dump(circuit, qasm_path)
            info["files"]["qasm"] = str(qasm_path)
        except Exception as exc:
            info["qasm_skipped_reason"] = (
                f"{type(exc).__name__}: circuit likely contains non-standard "
                "QPD/CutWire instructions; see .qpy / .txt instead."
            )

    return info


# Main entry point

def export_cutting_results(
    state: dict[str, Any],
    output_dir: str | Path = "cutting_runs",
    run_name: str | None = None,
) -> Path:
    """Dump every circuit + piece of metadata in `state` to disk for manual
    inspection, and print a concise summary to stdout.

    Parameters
    ----------
    state : dict
        The final AgentState returned by `app.invoke(...)`.
    output_dir : str | Path
        Parent directory for all runs (created if missing).
    run_name : str | None
        Subfolder name for this run. Defaults to a timestamp +
        cutting_strategy so repeated runs don't clobber each other.

    Returns
    -------
    Path
        The directory this run's files were written to.
    """
    strategy = state.get("cutting_strategy") or "unknown_strategy"
    if run_name is None:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"strategy": strategy, "circuits": {}, "subcircuits": {}}

    print("\n" + "=" * 60)
    print(f"EXPORTING CUTTING RESULTS -> {run_dir}")
    print("=" * 60)

    # ---- Single circuits ------------------------------------------------
    for key, label in (
        ("original_circuit", "original_circuit"),
        ("cut_circuit", "cut_circuit"),
        ("marked_circuit", "marked_circuit"),  # populated by auto_finder
    ):
        circuit = state.get(key)
        if circuit is None:
            continue
        info = _export_circuit(circuit, run_dir / label)
        summary["circuits"][label] = info
        qasm_note = "qasm OK" if "qasm" in info["files"] else "qasm skipped (non-standard gates)"
        print(
            f"  [{label}] {info['num_qubits']} qubits, "
            f"{info['num_instructions']} instructions, depth {info['depth']} "
            f"-> {info['files']['qpy']}  ({qasm_note})"
        )

    # ---- Subcircuits (dict[label -> QuantumCircuit]) ---------------------
    subcircuits = state.get("subcircuits")
    if subcircuits:
        sub_dir = run_dir / "subcircuits"
        print(f"\n  [subcircuits] {len(subcircuits)} partition(s):")
        for label, circuit in subcircuits.items():
            safe_label = _sanitize(label)
            info = _export_circuit(circuit, sub_dir / safe_label)
            summary["subcircuits"][str(label)] = info
            print(
                f"    - partition {label!r}: {info['num_qubits']} qubits, "
                f"{info['num_instructions']} instructions -> {info['files']['qpy']}"
            )
    else:
        print("\n  [subcircuits] none present in state (not partitioned, or this "
              "strategy didn't populate them)")

    # ---- Metadata sidecar --------------------------------------------------
    metadata_fields = [
        "cutting_strategy", "strategy_reasoning", "gate_ids", "gate_info","selector_note", "decision_rationale",
        "cut_locations", "cut_info", "qpd_bases", "total_sampling_overhead",
        "partition_labels", "minimum_reached", "metadata",
        "validation_passed", "validation_reason", "validation_attempts",
        "equivalence_passed", "equivalence_detail", "equivalence_seconds",
        "equivalence_skipped",
        # here so the LLM's chosen technique + justification paragraph
        # are captured in metadata.json like every other field, with no
        # change to the run-folder layout needed for other strategies
        # (these two keys are simply absent from metadata.json on runs
        # that didn't use llm_custom_cut).
        "llm_cut_technique", "llm_cut_reasoning", "llm_fallback",
        "error",
    ]
    metadata_out = {k: _json_safe(state.get(k)) for k in metadata_fields if k in state}
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    print(f"\n  [metadata] -> {metadata_path}")

    # The paragraph explaining *why* the LLM cut the circuit the way it
    # did is already captured in metadata.json above via
    # "llm_cut_reasoning", but it's also written out as a standalone,
    # easy-to-skim .txt file -- the same convenience rationale as the
    # plain-text circuit diagrams in _export_circuit(). Only written when
    # this run actually used llm_custom_cut (key present and non-empty).
    llm_cut_reasoning = state.get("llm_cut_reasoning")
    if llm_cut_reasoning:
        reasoning_path = run_dir / "llm_cut_reasoning.txt"
        reasoning_path.write_text(
            f"Technique chosen: {state.get('llm_cut_technique', 'unknown')}\n\n"
            f"{llm_cut_reasoning}\n",
            encoding="utf-8",
        )
        print(f"  [llm reasoning] -> {reasoning_path}")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  [summary]  -> {summary_path}")
    print("=" * 60)

    return run_dir
