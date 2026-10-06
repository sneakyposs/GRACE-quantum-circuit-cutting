"""
llm_custom_cut_node.py
=======================
LangGraph Node 5 — LLM_Custom_cut

An LLM is shown the QASM circuit (plus the upstream circuit/analysis
summaries already in state) and asked to propose a bespoke cutting plan:
which technique to use (QPD gate cutting or QPD wire cutting) and exactly
which gates/wires to cut. That plan is then executed using the SAME
underlying machinery as Node 6 (gate_cutting_node.py) and Node 7
(wire_cutting_node.py) -- this node does no cutting math of its own, it
only chooses targets and delegates.

Mirrors the style/structure of gate_cutting_node.py and wire_cutting_node.py
so it's interchangeable from Strategy_router's / validate_node's point of
view, and is usable both as a LangGraph node and as a standalone module.

Output contract (matches validate_node._run_validation_checks)
----------------------------------------------------------------
validate_node requires:
  1. state["original_circuit"] is not state["cut_circuit"]  (circuit altered)
  2. state["subcircuits"] is a non-empty dict

To guarantee (2) even when the LLM doesn't supply explicit
partition_labels, this node always calls the underlying cutting function
with auto_partition=True whenever no partition_labels were proposed.

Requires:
    pip install 'qiskit-addon-cutting>=0.10' qiskit

Author : Ezekiel Laney
License: Apache-2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from qiskit import QuantumCircuit

from parse_node import AgentState
from llm_client import call_llm

from gate_cutting_node import (
    gate_cutting_node,
    list_cuttable_gates,
    GateCuttingResult,
)
from wire_cutting_node import (
    wire_cutting_node,
    list_cuttable_wires,
    WireCuttingResult,
)


# 1.  Data container

@dataclass
class LLMCustomCutResult:
    """Structured output of the LLM custom-cut node.

    Wraps whichever underlying result (``GateCuttingResult`` or
    ``WireCuttingResult``) was produced, plus the LLM's own justification
    for the plan it chose, so callers don't need to know which technique
    was selected under the hood to consume the result.
    """

    original_circuit: QuantumCircuit
    cut_circuit: QuantumCircuit
    technique: str                      # "qpd_gate_cut" or "qpd_wire_cut"
    llm_reasoning: str                  # the LLM's justification paragraph
    qpd_bases: list
    gate_ids: list[int]
    gate_info: Optional[list[dict[str, Any]]]
    cut_locations: list[dict[str, int]]
    cut_info: Optional[list[dict[str, Any]]]
    total_sampling_overhead: float
    partition_labels: list[int | None] | None = None
    subcircuits: dict | None = None
    marked_circuit: QuantumCircuit | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (e.g. for LangGraph state).

        Keys deliberately match the existing AgentState field names used
        by gate_cutting_node / wire_cutting_node so validate_node and
        inspection_export.py work without modification (aside from the
        two new LLM-specific fields, see "=== " notes in
        parse_node.py / inspection_export.py).
        """
        return {
            "original_circuit": self.original_circuit,
            "cut_circuit": self.cut_circuit,
            "marked_circuit": self.marked_circuit,
            "qpd_bases": self.qpd_bases,
            "gate_ids": self.gate_ids,
            "gate_info": self.gate_info,
            "cut_locations": self.cut_locations,
            "cut_info": self.cut_info,
            "total_sampling_overhead": self.total_sampling_overhead,
            "partition_labels": self.partition_labels,
            "subcircuits": self.subcircuits,
            "metadata": self.metadata,
            # AgentState additions in parse_node.py.
            "llm_cut_technique":  self.technique,
            "llm_cut_reasoning":  self.llm_reasoning,
            "llm_fallback":       self.metadata.get("llm_fallback", False),
        }


# 2.  Core node function (standalone-usable)

def llm_custom_cut_node(
    circuit_input: str | QuantumCircuit,
    circuit_summary: str = "",
    analysis_summary: str = "",
    failed_strategies: Optional[list[str]] = None,
) -> LLMCustomCutResult:
    """Ask an LLM to propose a custom cutting plan, then execute it.

    This is the single entry-point designed for both standalone use and
    LangGraph integration (see ``llm_custom_cut_langgraph_node`` below).

    Parameters
    ----------
    circuit_input : str | QuantumCircuit
        An OpenQASM 2/3 string **or** a Qiskit ``QuantumCircuit``.
    circuit_summary : str
        LLM-readable text produced by parse_circuit_node (optional, used
        to enrich the prompt if available).
    analysis_summary : str
        LLM-readable text produced by analyze_circuit_node (optional).
    failed_strategies : list[str] | None
        Strategy keys already tried and failed this run (from
        validate_node). Only used to warn the LLM off re-proposing a
        technique that's already failed if 'llm_custom_cut' itself is in
        the list (i.e. a retry of this same node).

    Returns
    -------
    LLMCustomCutResult
    """
    circuit = _load_circuit(circuit_input)
    work_circuit = _strip_classical_bits(circuit)

    cuttable_gates = list_cuttable_gates(work_circuit)
    cuttable_wires = list_cuttable_wires(work_circuit)

    plan, llm_reasoning = _invoke_llm_cut_planner(
        circuit=work_circuit,
        circuit_summary=circuit_summary,
        analysis_summary=analysis_summary,
        cuttable_gates=cuttable_gates,
        cuttable_wires=cuttable_wires,
        failed_strategies=failed_strategies or [],
    )

    technique = plan["technique"]
    is_fallback = plan.get("llm_fallback", False)

    if technique == "qpd_gate_cut":
        result: GateCuttingResult = gate_cutting_node(
            circuit_input=work_circuit,
            gate_ids=plan["gate_ids"],
            partition_labels=plan.get("partition_labels"),
            auto_partition=(plan.get("partition_labels") is None),
        )
        return LLMCustomCutResult(
            original_circuit=result.original_circuit,
            cut_circuit=result.cut_circuit,
            technique="qpd_gate_cut",
            llm_reasoning=llm_reasoning,
            qpd_bases=result.qpd_bases,
            gate_ids=result.gate_ids,
            gate_info=result.gate_info,
            cut_locations=[],
            cut_info=None,
            total_sampling_overhead=result.total_sampling_overhead,
            partition_labels=result.partition_labels,
            subcircuits=result.subcircuits,
            marked_circuit=None,
            metadata={
                **result.metadata,
                "llm_proposed_gate_ids": plan["gate_ids"],
                "llm_plan_source": plan.get("source", "llm"),
                "llm_fallback": is_fallback,
            },
        )

    elif technique == "qpd_wire_cut":
        result_w: WireCuttingResult = wire_cutting_node(
            circuit_input=work_circuit,
            cut_locations=plan["cut_locations"],
            partition_labels=plan.get("partition_labels"),
            auto_partition=(plan.get("partition_labels") is None),
        )
        return LLMCustomCutResult(
            original_circuit=result_w.original_circuit,
            cut_circuit=result_w.cut_circuit,
            technique="qpd_wire_cut",
            llm_reasoning=llm_reasoning,
            qpd_bases=result_w.qpd_bases,
            gate_ids=[],
            gate_info=None,
            cut_locations=result_w.cut_locations,
            cut_info=result_w.cut_info,
            total_sampling_overhead=result_w.total_sampling_overhead,
            partition_labels=result_w.partition_labels,
            subcircuits=result_w.subcircuits,
            marked_circuit=result_w.marked_circuit,
            metadata={
                **result_w.metadata,
                "llm_proposed_cut_locations": plan["cut_locations"],
                "llm_plan_source": plan.get("source", "llm"),
                "llm_fallback": is_fallback,
            },
        )

    else:
        raise ValueError(f"Unknown technique '{technique}' in cut plan.")


# 3.  Helpers (mirrors gate_cutting_node.py / wire_cutting_node.py)

def _load_circuit(source: str | QuantumCircuit) -> QuantumCircuit:
    if isinstance(source, QuantumCircuit):
        return source.copy()
    from qiskit import qasm2, qasm3
    try:
        return qasm2.loads(source)
    except Exception:
        pass
    try:
        return qasm3.loads(source)
    except Exception as exc:
        raise ValueError(
            "Could not parse the input as OpenQASM 2 or OpenQASM 3."
        ) from exc


def _strip_classical_bits(circuit: QuantumCircuit) -> QuantumCircuit:
    if circuit.num_clbits == 0 and len(circuit.cregs) == 0:
        # Even without clbits, barriers must still be removed (see below).
        if not any(i.operation.name == "barrier" for i in circuit.data):
            return circuit.copy()
    new_qc = QuantumCircuit(*circuit.qregs)
    for inst in circuit.data:
        # and glue partitions together in partition_problem()'s automatic
        # labelling (the same barrier handling used elsewhere). Index frames
        # stay consistent: list_cuttable_gates reports indices in the frame
        # of the circuit passed to it, and gate_cutting_node remaps them.
        if inst.operation.name in ("measure", "reset", "barrier"):
            continue
        if len(inst.clbits) == 0:
            new_qc.append(inst.operation, inst.qubits)
    return new_qc


# 4.  LLM PLANNER
#     prompts, or parsing behaviour. Mirrors the structure of
#     strategy_router.py's _invoke_llm_strategy_selector().

def _invoke_llm_cut_planner(
    circuit: QuantumCircuit,
    circuit_summary: str,
    analysis_summary: str,
    cuttable_gates: list[dict[str, Any]],
    cuttable_wires: list[dict[str, Any]],
    failed_strategies: list[str],
) -> tuple[dict[str, Any], str]:
    """
    Calls openai/gpt-oss-120b:free (via OpenRouter, see llm_client.py) to
    propose a custom cutting plan for the circuit.

    The LLM is restricted to candidate gates/wires already enumerated by
    ``list_cuttable_gates`` / ``list_cuttable_wires`` (the same candidate
    sets qpd_gate_cut / qpd_wire_cut use) so it cannot propose an
    out-of-range or non-two-qubit gate index. It is instructed to reply
    with a single JSON object and nothing else.

    Falls back to a deterministic heuristic plan (see
    ``_deterministic_fallback_plan``) if the LLM call fails outright, the
    reply isn't valid/parseable JSON, or the proposed plan fails
    validation against the candidate sets -- so this node never crashes
    or produces zero subcircuits solely because the free-tier model is
    unavailable or misbehaves.

    Returns
    -------
    tuple[dict, str]
        ``(plan, reasoning_paragraph)`` where ``plan`` has keys
        ``technique`` ("qpd_gate_cut" | "qpd_wire_cut"), ``gate_ids``,
        ``cut_locations``, and ``partition_labels`` (None unless the LLM
        supplied them), and ``reasoning_paragraph`` is the LLM's own
        prose justification for the plan (or a clearly-labeled fallback
        explanation).
    """
    if not cuttable_gates and not cuttable_wires:
        # Nothing to cut at all (e.g. a 1-qubit or fully-separable circuit).
        return (
            {
                "technique": "qpd_gate_cut",
                "gate_ids": [],
                "cut_locations": [],
                "partition_labels": None,
                "source": "fallback_no_candidates",
            },
            "No two-qubit gates or cuttable wires were found in this "
            "circuit, so no cut could be proposed. The circuit is "
            "trivially separable / too small to cut.",
        )

    prompt = _build_cut_prompt(
        circuit, circuit_summary, analysis_summary, cuttable_gates, cuttable_wires
    )

    failed_note = (
        " Note: 'llm_custom_cut' already failed validation earlier this "
        "run -- propose a DIFFERENT set of cuts than you might have "
        "before."
        if "llm_custom_cut" in failed_strategies else ""
    )

    try:
        raw = call_llm(
            system_prompt=(
                "You are an expert in quantum circuit cutting (quasi-"
                "probability decomposition, QPD), embedded inside an "
                "automated pipeline. You will be given a quantum circuit "
                "and a list of CANDIDATE cut locations. Propose a cutting "
                "plan using ONLY locations from those candidate lists."
                + failed_note +
                " Reply with EXACTLY one JSON object and nothing else "
                "(no markdown fences, no commentary outside the JSON). "
                "The JSON object must have these keys:\n"
                '  "technique": either "qpd_gate_cut" or "qpd_wire_cut"\n'
                '  "gate_ids": list of integers from the candidate gate '
                "indices (required if technique is qpd_gate_cut, else [])\n"
                '  "cut_locations": list of {"qubit": int, '
                '"after_instruction": int} objects taken from the '
                "candidate wire list (required if technique is "
                "qpd_wire_cut, else [])\n"
                '  "reasoning": 2-3 sentences explaining '
                "why you chose this technique and these specific cut "
                "locations.\n"
                "HARD CONSTRAINT: total sampling overhead multiplies ~9x "
                "per gate cut and ~16x per wire cut. Keep total overhead "
                "under ~6500: AT MOST 4 gate cuts or 3 wire cuts. Choose "
                "the FEWEST cuts that still split the circuit into two or "
                "more pieces."
            ),
            user_prompt=prompt,
            temperature=0.2,
            max_tokens=1200,
        )

        plan = _parse_cut_plan(raw, cuttable_gates, cuttable_wires)
        reasoning = plan.pop("reasoning", "") or raw.strip()
        plan["source"] = "llm"
        return plan, reasoning

    except Exception as exc:
        # Covers: missing OPENROUTER_API_KEY, no .env file, no internet,
        # rate limiting, malformed/unparseable response, invalid plan, etc.
        fallback = _deterministic_fallback_plan(
            circuit, cuttable_gates, cuttable_wires)
        fallback["source"] = "fallback_llm_unavailable"
        fallback["llm_fallback"] = True
        return fallback, (
            f"LLM custom cut planning unavailable or returned an invalid "
            f"plan ({exc}); falling back to a deterministic heuristic: "
            f"cutting the first available candidate "
            f"({fallback['technique']})."
        )


def _build_cut_prompt(
    circuit: QuantumCircuit,
    circuit_summary: str,
    analysis_summary: str,
    cuttable_gates: list[dict[str, Any]],
    cuttable_wires: list[dict[str, Any]],
) -> str:
    """Construct a structured prompt describing the circuit and candidate
    cut locations for the LLM cut planner."""

    lines: list[str] = []

    if circuit_summary:
        lines += ["=== CIRCUIT SUMMARY ===", circuit_summary, ""]
    if analysis_summary:
        lines += ["=== ANALYSIS SUMMARY ===", analysis_summary, ""]

    lines += [
        "=== CANDIDATE TWO-QUBIT GATES (for qpd_gate_cut) ===",
        "Each entry: index | gate name | qubits acted on.",
    ]
    if cuttable_gates:
        for g in cuttable_gates:
            lines.append(
                f"  index={g['index']}  name={g['name']}  "
                f"qubits={g['qubit_indices']}"
            )
    else:
        lines.append("  (none available)")

    lines += [
        "",
        "=== CANDIDATE WIRE-CUT LOCATIONS (for qpd_wire_cut) ===",
        "Each entry: qubit | after_instruction | preceding_gate -> following_gate.",
    ]
    if cuttable_wires:
        for w in cuttable_wires:
            lines.append(
                f"  qubit={w['qubit']}  after_instruction={w['after_instruction']}  "
                f"{w['preceding_gate']} -> {w['following_gate']}"
            )
    else:
        lines.append("  (none available)")

    lines += [
        "",
        "Choose exactly one technique and propose cuts using ONLY the "
        "candidate indices/locations listed above. Prefer the smallest "
        "number of cuts that meaningfully partitions the circuit into "
        "two or more pieces (fewer cuts = lower sampling overhead).",
    ]

    return "\n".join(lines)


def _parse_cut_plan(
    raw: str,
    cuttable_gates: list[dict[str, Any]],
    cuttable_wires: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse and validate the LLM's JSON reply into a usable plan dict.

    Tolerant of minor formatting slips (stray markdown fences) since
    small free-tier models occasionally wrap JSON in ```json fences
    despite instructions not to. Raises ValueError on anything that
    can't be salvaged into a valid plan, which the caller catches and
    routes to the deterministic fallback.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` or ``` ... ``` fences.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)  # raises ValueError/json.JSONDecodeError if malformed

    technique = str(data.get("technique", "")).strip().lower()
    if technique not in ("qpd_gate_cut", "qpd_wire_cut"):
        raise ValueError(f"Unrecognized technique in LLM plan: {technique!r}")

    valid_gate_ids = {g["index"] for g in cuttable_gates}
    valid_wire_keys = {
        (w["qubit"], w["after_instruction"]) for w in cuttable_wires
    }

    gate_ids = [int(i) for i in data.get("gate_ids", []) or []]
    raw_locations = data.get("cut_locations", []) or []
    cut_locations: list[dict[str, int]] = []
    for loc in raw_locations:
        q = int(loc["qubit"])
        a = int(loc["after_instruction"])
        cut_locations.append({"qubit": q, "after_instruction": a})

    if technique == "qpd_gate_cut":
        if not gate_ids:
            raise ValueError("technique=qpd_gate_cut but gate_ids is empty.")
        bad = [g for g in gate_ids if g not in valid_gate_ids]
        if bad:
            raise ValueError(f"gate_ids {bad} are not in the candidate list.")
    else:  # qpd_wire_cut
        if not cut_locations:
            raise ValueError("technique=qpd_wire_cut but cut_locations is empty.")
        bad_loc = [
            loc for loc in cut_locations
            if (loc["qubit"], loc["after_instruction"]) not in valid_wire_keys
        ]
        if bad_loc:
            raise ValueError(
                f"cut_locations {bad_loc} are not in the candidate list."
            )

    partition_labels = data.get("partition_labels")
    if partition_labels is not None:
        partition_labels = [
            (None if p is None else int(p)) for p in partition_labels
        ]

    return {
        "technique": technique,
        "gate_ids": gate_ids if technique == "qpd_gate_cut" else [],
        "cut_locations": cut_locations if technique == "qpd_wire_cut" else [],
        "partition_labels": partition_labels,
        "reasoning": str(data.get("reasoning", "")).strip(),
    }


def _deterministic_fallback_plan(
    circuit: QuantumCircuit,
    cuttable_gates: list[dict[str, Any]],
    cuttable_wires: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic backstop plan used when the LLM is unavailable or its
    reply can't be parsed/validated.

    Prefers a MINIMAL gate cut (cheaper, simpler to reconstruct, and — most
    importantly — validatable) if any two-qubit gate is available;
    otherwise falls back to the first candidate wire cut.

    All plans returned by this function carry ``llm_fallback=True`` so
    downstream code (validate_node, batch analysis) can distinguish these
    from genuine LLM-proposed plans.
    """
    if cuttable_gates:
        from strategy_router import _min_cut_gate_ids

        cuttable_idx = {g["index"] for g in cuttable_gates}
        gate_ids = [i for i in _min_cut_gate_ids(circuit) if i in cuttable_idx]
        if not gate_ids:
            gate_ids = [g["index"] for g in cuttable_gates]
        return {
            "technique": "qpd_gate_cut",
            "gate_ids": gate_ids,
            "cut_locations": [],
            "partition_labels": None,
            "llm_fallback": True,
        }
    if cuttable_wires:
        return {
            "technique": "qpd_wire_cut",
            "gate_ids": [],
            "cut_locations": [
                {
                    "qubit": cuttable_wires[0]["qubit"],
                    "after_instruction": cuttable_wires[0]["after_instruction"],
                }
            ],
            "partition_labels": None,
            "llm_fallback": True,
        }
    raise ValueError(
        "No cuttable two-qubit gates or wires found — "
        "circuit cannot be cut by LLM_Custom_cut."
    )


# 5.  LangGraph adapter (thin wrapper)

def llm_custom_cut_langgraph_node(state: AgentState) -> AgentState:
    """LangGraph-compatible node wrapper for Node 5 — LLM_Custom_cut.

    Expects the following keys in *state*:
      - "circuit" : str | QuantumCircuit (required)
      - "circuit_summary" : str (optional)
      - "analysis_summary" : str (optional)
      - "failed_strategies" : list[str] (optional)

    Returns a dict that can be merged directly into the LangGraph state,
    matching the pattern used by gate_cutting_langgraph_node /
    wire_cutting_langgraph_node.
    """
    if state.get("error"):
        return state

    circuit = state.get("circuit")
    if circuit is None:
        return {**state, "error": "No circuit found. Run parse_circuit_node first."}

    try:
        result = llm_custom_cut_node(
            circuit_input=circuit,
            circuit_summary=state.get("circuit_summary", "") or "",
            analysis_summary=state.get("analysis_summary", "") or "",
            failed_strategies=state.get("failed_strategies") or [],
        )
        return {
            **state,
            **result.to_dict(),
            "error": None,
        }
    except Exception as exc:
        return {**state, "error": f"LLM_Custom_cut failed: {exc}"}


# 6.  Standalone self-test / demo  (python llm_custom_cut_node.py [path.qasm])

if __name__ == "__main__":
    import sys

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

    path = sys.argv[1] if len(sys.argv) > 1 else None
    circuit_input = path if path else demo_qasm

    print("=" * 60)
    print("LLM_Custom_cut Node — Self-Test")
    print("=" * 60)

    if path:
        circuit = QuantumCircuit.from_qasm_file(path)
    else:
        circuit = demo_qasm

    result = llm_custom_cut_node(circuit)

    print(f"\nTechnique chosen : {result.technique}")
    print(f"Gate ids cut     : {result.gate_ids}")
    print(f"Wire cuts        : {result.cut_locations}")
    print(f"\nOriginal circuit ({result.original_circuit.num_qubits} qubits):")
    print(result.original_circuit.draw(output="text"))
    print(f"\nCut circuit ({result.cut_circuit.num_qubits} qubits):")
    print(result.cut_circuit.draw(output="text"))

    if result.subcircuits:
        print(f"\nSubcircuits ({len(result.subcircuits)} partition(s)):")
        for label, sub in result.subcircuits.items():
            print(f"  Partition {label!r}: {sub.num_qubits} qubits, {len(sub.data)} instructions")
    else:
        print("\nNo subcircuits were produced (this would fail validate_node).")

    print(f"\nTotal sampling overhead: {result.total_sampling_overhead}")
    print("\n--- LLM REASONING ---")
    print(result.llm_reasoning)

    # ---- Standalone state-dict round-trip, mirroring graph usage --------
    fake_state: AgentState = {
        "qasm_path": path or "",
        "circuit": circuit,
        "circuit_summary": "",
        "analysis_summary": "",
        "cutting_strategy": "llm_custom_cut",
        "failed_strategies": [],
        "error": None,
    }
    out_state = llm_custom_cut_langgraph_node(fake_state)
    print("\n--- STATE KEYS WRITTEN ---")
    for k in result.to_dict().keys():
        print(f"  {k}: {'OK' if out_state.get(k) is not None else 'None'}")

    print("\n✓ Self-test complete.")
