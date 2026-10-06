"""
strategy_router_node.py
-----------------------
LangGraph Node 3 — Strategy Router

Consumes the structured `analysis` dict produced by analyze_circuit_node
and the parsed `circuit` from parse_circuit_node, then decides which
circuit-cutting technique to employ.

Current behaviour (demo mode):
    Always routes to QPD gate cutting (the only completed cutting node).

To wire in a real LLM later, replace ONLY the body of
`_invoke_llm_strategy_selector()`.  Everything else — the node function,
conditional edge, gate-id selector, state fields, and subgraph wiring —
stays exactly the same.

Node flow position:
    parse_circuit → analyze_circuit → [THIS NODE] → qpd_gate_cut (or others)
"""

from __future__ import annotations

import os
from typing import Any, Optional
from qiskit import QuantumCircuit

from parse_node import AgentState

# openai/gpt-oss-120b:free via OpenRouter. See _invoke_llm_strategy_selector()
# below for where it's actually used.
from llm_client import call_llm


# 1.  STRATEGY CONSTANTS
#     These are the four cutting strategies this
#     router knows about.  Values MUST match the
#     node names used in workflow.add_node() in
#     subgraph.py exactly.

# Maps strategy keys → node names registered in the subgraph.
# Add new entries here as new cutting nodes are completed.
STRATEGY_NODE_MAP: dict[str, str] = {
    "auto_finder":    "auto_finder",     # Node 4 – AutoFinder
    "llm_custom_cut": "llm_custom_cut",  # Node 5 – LLM Custom Cut
    "qpd_gate_cut":   "qpd_gate_cut",   # Node 6 – QPD Gate Cut
}

# Fallback / hardcoded strategy while LLM is not yet wired in.
# Change this to test-drive a different default once more nodes are built.
_DEFAULT_STRATEGY: str = "qpd_gate_cut"

# failed validation this run (see _pick_alternative_strategy below).
# llm_custom_cut is deliberately excluded: it is ALWAYS the first strategy
# attempted (see the first-attempt policy in strategy_router_node), so by
# the time a retry/fallback pick happens it has already failed validation;
# retries choose among the remaining three techniques only.
_ACTIVE_STRATEGY_PRIORITY: list[str] = ["qpd_gate_cut", "auto_finder"]


def _pick_alternative_strategy(failed_strategies: list[str]) -> str:
    """
    Return the highest-priority strategy in ``_ACTIVE_STRATEGY_PRIORITY``
    that is NOT in ``failed_strategies``.

    This is the deterministic backstop that guarantees the graph never
    re-routes to a strategy that's already failed validation this run —
    even if the LLM selector ignores the failed-strategies note in its
    prompt (small/free models sometimes do).

    If every active strategy has already failed, there's nothing safe
    left to try; we return ``_DEFAULT_STRATEGY`` anyway and let
    validate_node's MAX_VALIDATION_LOOPS backstop end the graph instead
    of looping forever.
    """
    for candidate in _ACTIVE_STRATEGY_PRIORITY:
        if candidate not in failed_strategies:
            return candidate
    return _DEFAULT_STRATEGY


def remaining_active_strategies(failed_strategies: list[str]) -> list[str]:
    """Active retry strategies (in ``_ACTIVE_STRATEGY_PRIORITY`` order) that
    are NOT yet in ``failed_strategies``.

    Consumed by ``validate_node`` to decide whether an equivalence-infeasible
    attempt still has a leaner strategy to reroute to before conceding a
    terminal VALIDATION_INFEASIBLE. Returns [] when every active strategy has
    already been tried and marked failed. (Mirrors the selection space of
    ``_pick_alternative_strategy``; ``llm_custom_cut`` is intentionally
    excluded because it is the forced first attempt only.)
    """
    failed = set(failed_strategies or [])
    return [s for s in _ACTIVE_STRATEGY_PRIORITY if s not in failed]


# 2.  NODE FUNCTION
#     Signature required by LangGraph:
#       fn(state: State) -> State

def strategy_router_node(state: AgentState) -> AgentState:
    """
    Node 3 – Strategy Router
    Reads `analysis` (structured metrics) and `circuit` (QuantumCircuit)
    from shared state, selects a cutting strategy, identifies which gates or
    wires to target, and writes the routing decision back into state.

    Writes to state
    ---------------
    cutting_strategy  : str        – key from STRATEGY_NODE_MAP
    strategy_reasoning: str        – LLM-readable explanation of the choice
    gate_ids          : list[int]  – instruction indices to cut (qpd_gate_cut)
    auto_partition    : bool       – whether the cutting node should auto-partition
    partition_labels  : list|None  – explicit per-qubit labels (None = let node decide)
    error             : str|None   – non-None if something went wrong
    """
    # Short-circuit: propagate upstream (parse/analyze) errors without
    # masking them.  BUT only on the first attempt — during retries a
    # stale "error" left by a cutting node (e.g. llm_custom_cut raising
    # an exception) must NOT prevent the router from computing fresh
    # gate_ids and cutting_strategy for the next strategy.
    failed_strategies_so_far: list[str] = list(state.get("failed_strategies") or [])
    is_retry = bool(failed_strategies_so_far) or bool(state.get("validation_attempts"))
    if state.get("error") and not is_retry:
        return state

    analysis: Optional[dict] = state.get("analysis")
    if analysis is None:
        return {
            **state,
            "error": "No analysis found in state.  Run analyze_circuit_node first.",
        }

    circuit: Optional[QuantumCircuit] = state.get("circuit")
    if circuit is None:
        return {
            **state,
            "error": "No circuit found in state.  Run parse_circuit_node first.",
        }

    try:
        # validation this run (written by validate_node). Defaults to []
        # on the first pass through the graph.
        failed_strategies: list[str] = list(state.get("failed_strategies") or [])

        # ── Step 1: Choose a cutting strategy ───────────────────────────────
        # POLICY: the FIRST attempt of every run always uses llm_custom_cut.
        # Only if Validate rejects its output do subsequent retries choose
        # freely among auto_finder / qpd_gate_cut via the
        # existing LLM-selector + deterministic-fallback logic below
        # (llm_custom_cut will then be in failed_strategies, so neither the
        # selector override nor _pick_alternative_strategy can re-pick it).
        no_llm = os.environ.get("GRACE_NO_LLM", "") == "1"
        first_attempt = (
            not failed_strategies
            and not (state.get("validation_attempts") or 0)
        )
        if first_attempt and not no_llm:
            strategy = "llm_custom_cut"
            llm_reasoning = (
                "Policy: the first cutting attempt always uses "
                "llm_custom_cut; the LLM-selector is only consulted on "
                "retries after Validate rejects its output."
            )
        elif failed_strategies:
            # (qpd_gate_cut → auto_finder) instead of the LLM selector.
            # auto_finder for circuits where qpd_gate_cut (with its
            # _min_cut_gate_ids) would succeed with fewer cuts. This
            # caused 21 easy circuits to fail in LLM mode that passed
            # in no-LLM mode, where the deterministic fallback always
            # picks qpd_gate_cut first.
            strategy = _pick_alternative_strategy(failed_strategies)
            llm_reasoning = (
                f"Retry: deterministic fallback chose '{strategy}' "
                f"(failed so far: {failed_strategies})."
            )
        else:
            # _invoke_llm_strategy_selector() now returns
            # (strategy, llm_reasoning) instead of just `strategy`, so the
            strategy, llm_reasoning = _invoke_llm_strategy_selector(
                analysis        = analysis,
                circuit_summary = state.get("circuit_summary",   ""),
                analysis_summary= state.get("analysis_summary",  ""),
                failed_strategies = failed_strategies,
            )

        # Guard: ensure the strategy is one the subgraph knows how to route to
        if strategy not in STRATEGY_NODE_MAP:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid options: {sorted(STRATEGY_NODE_MAP.keys())}"
            )

        # already in failed_strategies (e.g. it ignored the prompt note, or
        # we're still on the hardcoded _DEFAULT_STRATEGY fallback path),
        # override it with the next strategy that hasn't failed yet. This is
        # what actually guarantees the graph never re-routes to a failed
        # technique, regardless of what the LLM does.
        if strategy in failed_strategies:
            fallback_strategy = _pick_alternative_strategy(failed_strategies)
            llm_reasoning = (
                f"NOTE: selector chose '{strategy}', which already failed "
                f"validation this run (failed so far: {failed_strategies}). "
                f"Overriding to '{fallback_strategy}'.\n\n{llm_reasoning}"
            )
            strategy = fallback_strategy

        # ── Step 2: Identify which instructions/wires to cut ────────────────
        gate_ids: list[int] = _select_gate_ids(circuit, strategy)

        # ── Step 3: Generate a SEPARATE decision rationale ──────────────────
        # A second, dedicated LLM call justifies *this specific strategy
        # choice* for the circuit. This is intentionally distinct from
        # analyze_circuit_node's `reasoning_notes` (which characterise the
        # circuit's cuttability without naming a strategy) and from the
        # DECISION RATIONALE section just copied `reasoning_notes`.
        decision_rationale = _invoke_llm_decision_rationale(
            strategy           = strategy,
            gate_ids           = gate_ids,
            analysis           = analysis,
            selector_reasoning = llm_reasoning,
            failed_strategies  = failed_strategies,
        )

        # ── Step 4: Build a human-readable routing summary ──────────────────
        # router's own decision rationale through to the summary. ===
        routing_summary = _format_routing_decision(
            strategy, gate_ids, analysis, llm_reasoning, decision_rationale
        )

        return {
            **state,
            "cutting_strategy":  strategy,
            "strategy_reasoning": routing_summary,
            "selector_note":     llm_reasoning,
            "decision_rationale": decision_rationale,
            "gate_ids":          gate_ids,
            "cut_locations":     [],
            "auto_partition":    True,   # default: let the cutting node auto-partition
            "partition_labels":  None,   # None → auto_partition controls the split
            "failed_strategies": failed_strategies,
            "error":             None,
        }

    except Exception as exc:


        return {**state, "error": f"Strategy router failed: {exc}"}


# 3.  LLM STRATEGY SELECTOR (stub)
#
#     CHANGE WHEN ADDING A REAL LLM.
#
#     Keep the signature identical.
#     The caller validates the return value
#     against STRATEGY_NODE_MAP automatically.

def _invoke_llm_strategy_selector(
    analysis: dict,
    circuit_summary: str,
    analysis_summary: str,
    failed_strategies: Optional[list[str]] = None,
) -> tuple[str, str]:
    """
    
    Calls openai/gpt-oss-120b:free (via OpenRouter, see llm_client.py) to
    choose a cutting strategy for the circuit, using the prompt built by
    `_build_strategy_prompt()` below (which was already written for this
    purpose in the selector — only this function's body changed).

    Deviation from the original docstring's "keep this signature
    identical" note
    -----------------------------------------------------------------
    This now returns ``tuple[str, str]`` -- (strategy, llm_reasoning) --
    instead of just ``str``. The selector discarded any reasoning
    text the LLM might produce; returning it too lets the LLM's actual
    justification appear in `state["strategy_reasoning"]` instead of only
    a bare strategy key. `strategy_router_node()` has been updated to
    unpack this tuple accordingly.

    ``failed_strategies`` parameter
    -----------------------------------------------------------------
    List of strategy keys that have already failed validation earlier
    in this run (written by validate_node). When non-empty, it's folded
    into the system + user prompt so the LLM is explicitly told not to
    re-select those techniques. This is advisory only -- the hard
    guarantee that a failed strategy is never re-used lives in
    strategy_router_node's deterministic override (see
    _pick_alternative_strategy), since small free-tier models don't
    reliably follow prompt instructions.

    Behavior
    --------
    The LLM is instructed to reply in a fixed two-line format
    ("STRATEGY: ...", "REASON: ...") so the reply can be parsed reliably.
    `_parse_strategy_response()` also tolerates minor deviations from
    that format, since small open-weight models occasionally add stray
    text despite instructions.

    Falls back to ``_DEFAULT_STRATEGY`` (or, if that's already failed,
    to the next strategy in ``_pick_alternative_strategy``) with a
    clearly-labeled fallback reasoning string if the LLM call fails
    outright (missing API key, no internet, rate limit, etc.) or returns
    something that can't be mapped to a known strategy key, so the demo
    never crashes solely because the LLM is unavailable.

    Parameters
    ----------
    analysis : dict
        The structured ``CircuitAnalysis`` dict from analyze_circuit_node.
    circuit_summary : str
        LLM-readable text produced by parse_circuit_node.
    analysis_summary : str
        LLM-readable text produced by analyze_circuit_node.
    failed_strategies : list[str] | None
        Strategy keys already tried and failed this run. Defaults to [].

    Returns
    -------
    tuple[str, str]
        ``(strategy, reasoning)`` where ``strategy`` is a key from
        ``STRATEGY_NODE_MAP`` and ``reasoning`` is the LLM's own
        explanation (or a fallback-explanation string).
    """
    failed_strategies = failed_strategies or []

    def _safe_default() -> str:
        # Don't fall back to a strategy that's already known to fail.
        if _DEFAULT_STRATEGY in failed_strategies:
            return _pick_alternative_strategy(failed_strategies)
        return _DEFAULT_STRATEGY

    try:
        prompt = _build_strategy_prompt(
            analysis, circuit_summary, analysis_summary, failed_strategies
        )

        failed_note = (
            f" The following strategies were already tried and FAILED "
            f"validation this run -- do NOT select them again: "
            f"{', '.join(failed_strategies)}."
            if failed_strategies else ""
        )

        raw = call_llm(
            system_prompt=(
                "You are an expert in quantum circuit partitioning, embedded "
                "inside an automated pipeline. Given a circuit's summary and "
                "metrics, choose exactly ONE circuit-cutting strategy and "
                "briefly justify the choice." + failed_note + " You MUST reply "
                "in EXACTLY this two-line format and nothing else. (no "
                "markdown, no extra commentary.):\n"
                "STRATEGY: <one of auto_finder, llm_custom_cut, qpd_gate_cut>\n"
                "REASON: <one or two sentences of justification>"
            ),
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=400,
        )

        strategy, reasoning = _parse_strategy_response(raw)

        if strategy not in STRATEGY_NODE_MAP:
            # The LLM said something we don't recognize -- fall back safely
            # rather than letting an unmapped key crash the conditional edge.
            fallback = _safe_default()
            return fallback, (
                f"LLM reply could not be mapped to a known strategy key "
                f"(raw reply: {raw!r}). Falling back to '{fallback}'."
            )

        return strategy, reasoning

    except Exception as exc:
        # Covers: missing OPENROUTER_API_KEY, no .env file, no internet,
        # rate limiting, malformed response, etc.
        fallback = _safe_default()
        return fallback, (
            f"LLM strategy selection unavailable ({exc}); "
            f"falling back to default strategy '{fallback}'."
        )


def _parse_strategy_response(raw: str) -> tuple[str, str]:
    """
    
    Parses the LLM's "STRATEGY: ...\\nREASON: ..." reply into
    (strategy_key, reasoning_text).

    Tolerant of minor format deviations: if no line starts with
    "STRATEGY:", it falls back to scanning the raw text for any known
    strategy key as a substring. This matters in practice because small
    free-tier models sometimes ignore formatting instructions.
    """
    strategy = ""
    reason = ""

    for line in raw.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("strategy:"):
            strategy = stripped.split(":", 1)[1].strip().lower()
        elif lower.startswith("reason:"):
            reason = stripped.split(":", 1)[1].strip()

    # Strip stray punctuation/quotes/markdown the model might wrap the key in.
    strategy = strategy.strip("`'\" .*")

    if strategy not in STRATEGY_NODE_MAP:
        # Fallback: scan the whole reply for any known key as a substring.
        lower_all = raw.lower()
        for key in STRATEGY_NODE_MAP:
            if key in lower_all:
                strategy = key
                break

    if not reason:
        reason = raw.strip()

    return strategy, reason


# 4.  PROMPT BUILDER
#     Ready-to-use when an LLM is wired in.
#     Pass the returned string to your LLM call
#     inside _invoke_llm_strategy_selector().

def _build_strategy_prompt(
    analysis: dict,
    circuit_summary: str,
    analysis_summary: str,
    failed_strategies: Optional[list[str]] = None,
) -> str:
    """
    Construct a structured prompt for an LLM to choose a cutting strategy.

    This function is not called in demo mode.  When an LLM is added,
    call it inside ``_invoke_llm_strategy_selector`` and pass the result
    to your LLM invocation.

    The prompt describes the available strategies and the circuit metrics,
    then asks the LLM to reply with exactly one strategy key.

    if ``failed_strategies`` is non-empty, a note listing them is
    appended so the LLM avoids re-selecting a technique that's already
    failed validation earlier in this run.
    """
    failed_strategies = failed_strategies or []

    strategy_descriptions = (
        "  auto_finder    : Automatically discovers the optimal cut locations.  "
        "Best when no prior knowledge about the circuit structure is available.\n"
        "  llm_custom_cut : Uses a second LLM call to craft a bespoke cutting plan.  "
        "Best for highly irregular or research-specific circuits.\n"
        "  qpd_gate_cut   : Quasi-Probability Decomposition applied to selected two-qubit gates.  "
        "Best when there are few cross-partition two-qubit gates."
    )

    failed_block = ""
    if failed_strategies:
        failed_block = (
            "\n=== ALREADY-FAILED STRATEGIES (do not select again) ===\n"
            f"{', '.join(failed_strategies)}\n"
        )

    return f"""You are an expert in quantum circuit partitioning.
Your task is to choose the best circuit-cutting strategy for the circuit described below.

=== AVAILABLE STRATEGIES ===
{strategy_descriptions}
{failed_block}
=== CIRCUIT SUMMARY ===
{circuit_summary}

=== CIRCUIT ANALYSIS ===
{analysis_summary}

=== KEY METRICS ===
- Qubits             : {analysis['num_qubits']}
- Depth              : {analysis['depth']}
- Two-qubit gates    : {analysis['two_qubit_gates']}
- Connectivity density: {analysis['connectivity_density']}
- Dense?             : {analysis['is_dense']}
- Deep?              : {analysis['is_deep']}
- Non-Clifford gates : {analysis['non_clifford_count']}

=== REASONING NOTES (from analyze_circuit_node) ===
{analysis.get('reasoning_notes', 'None')}

Respond with ONLY one of the following strategy keys (no other text):
auto_finder | llm_custom_cut | qpd_gate_cut
"""


# 5.  GATE / WIRE INDEX SELECTION
#     Given a decided strategy, identify which
#     specific instructions in circuit.data
#     should be targeted for cutting.

def _select_gate_ids(circuit: QuantumCircuit, strategy: str) -> list[int]:
    """
    Return a list of ``circuit.data`` indices to cut.

    For ``qpd_gate_cut``:
        All two-qubit gates are selected.  The gate-cutting node
        handles measurements and barriers itself, but we exclude them
        here as a belt-and-suspenders measure.

    For all other strategies:
        Returns an empty list.  Those cutting nodes are responsible
        for determining their own cut locations (AutoFinder finds its own
        optimal cuts; wire-cut and LLM-custom nodes have their own logic).

    When an LLM is integrated, you can extend ``_invoke_llm_strategy_selector``
    to also return a list of gate indices (e.g. via structured JSON output)
    and unpack them here instead of using the heuristic below.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parsed quantum circuit from ``state["circuit"]``.
    strategy : str
        A key from ``STRATEGY_NODE_MAP``.

    Returns
    -------
    list[int]
        Indices into ``circuit.data`` of the instructions to cut.
    """
    if strategy == "qpd_gate_cut":
        return _min_cut_gate_ids(circuit)

    # Other strategies find their own targets.
    if strategy in ("auto_finder", "llm_custom_cut"):
        return []

    return []


# overhead by 9 per cut, which explodes past any usable budget on dense
# circuits (e.g. 6 cuts -> overhead ~5e5). Instead, choose the MINIMUM set
# of gates whose removal bipartitions the qubit-interaction graph: brute
# force over bipartitions (cheap for <= 14 qubits), minimizing crossing
# gate count and, as a tiebreak, partition imbalance.
_MAX_BRUTE_FORCE_QUBITS = 14


def _min_cut_gate_ids(circuit: QuantumCircuit) -> list[int]:
    _SKIP = {"measure", "barrier", "reset"}
    twoq = [(idx, tuple(sorted(circuit.find_bit(q).index
                               for q in inst.qubits)))
            for idx, inst in enumerate(circuit.data)
            if len(inst.qubits) == 2 and inst.operation.name not in _SKIP]
    if not twoq:
        return []

    # Idle qubits (and secondary components) otherwise admit a degenerate
    # "cut" with zero crossings - idle qubits on one side, the circuit on
    # the other - so nothing gets cut and a trivial single-partition
    # export results (observed on Bernstein-Vazirani). Restrict the
    # search to the largest connected component of the interaction
    # graph; other components separate for free anyway.
    adj: dict[int, set[int]] = {}
    for _idx, (a, b) in twoq:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    components: list[set[int]] = []
    unseen = set(adj)
    while unseen:
        stack = [unseen.pop()]
        comp = set(stack)
        while stack:
            for nb in adj[stack.pop()]:
                if nb in unseen:
                    unseen.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        components.append(comp)
    comp = max(components, key=len)
    if len(comp) < 2:
        return []
    if len(comp) > _MAX_BRUTE_FORCE_QUBITS:
        # legacy behaviour for circuits too big to brute force
        return _find_two_qubit_gate_ids(circuit)

    qubits = sorted(comp)
    pos = {q: i for i, q in enumerate(qubits)}
    comp_gates = [(idx, pos[a], pos[b]) for idx, (a, b) in twoq
                  if a in comp and b in comp]
    m = len(qubits)
    best_key: tuple | None = None
    best_ids: list[int] = []
    for mask in range(1, 1 << (m - 1)):  # last component qubit pinned to B
        ids = [idx for idx, a, b in comp_gates
               if ((mask >> a) & 1) != ((mask >> b) & 1)]
        size_a = bin(mask).count("1")
        key = (len(ids), abs(m - 2 * size_a))
        if best_key is None or key < best_key:
            best_key, best_ids = key, ids
    return best_ids


def _find_two_qubit_gate_ids(circuit: QuantumCircuit) -> list[int]:
    """
    Return the indices of every two-qubit gate in the circuit.

    Measurements, barriers, and resets are skipped because they either
    act on classical bits (which gate-cutting cannot handle) or are not
    genuinely entangling operations.
    """
    _SKIP = {"measure", "barrier", "reset"}
    return [
        idx
        for idx, inst in enumerate(circuit.data)
        if len(inst.qubits) == 2 and inst.operation.name not in _SKIP
    ]


# 6.  CONDITIONAL EDGE ROUTING FUNCTION
#     Import this into subgraph.py and pass it
#     to workflow.add_conditional_edges().
#
#     Contains NO routing logic — the decision
#     already lives in strategy_router_node.
#     This is intentional: swapping the LLM in
#     only touches Section 3 above.

def route_to_cutter(state: AgentState) -> str:
    """
    LangGraph conditional-edge function for the strategy router.

    Reads ``state["cutting_strategy"]`` (written by ``strategy_router_node``)
    and returns it as the routing key.  LangGraph maps that key to the
    appropriate cutting node via the dict passed to
    ``workflow.add_conditional_edges()``.

    Returns ``_DEFAULT_STRATEGY`` (or an alternative that hasn't failed
    yet -- see below) if the state contains an error or the strategy key
    is missing — the cutting node's own error guard will then short-circuit
    and the error propagates cleanly to the graph end.

    this fallback now also avoids ``state["failed_strategies"]``.
    In normal operation strategy_router_node already guarantees
    ``cutting_strategy`` isn't a failed one (see
    ``_pick_alternative_strategy``), so this is a second, independent
    line of defense rather than the primary mechanism.

    Usage in subgraph.py
    --------------------
    ::

        from strategy_router_node import strategy_router_node, route_to_cutter

        workflow.add_conditional_edges(
            "strategy_router",
            route_to_cutter,
            {
                "qpd_gate_cut":   "qpd_gate_cut",

                "auto_finder":    END,            # not yet implemented
                "llm_custom_cut": END,            # not yet implemented
            },
        )
    """
    strategy = state.get("cutting_strategy")
    failed_strategies: list[str] = list(state.get("failed_strategies") or [])

    # If the state has an error, no strategy was set, or the strategy
    # already failed validation this run, fall back to a strategy that
    # hasn't failed yet so we never re-route to a known-bad technique.
    if not strategy or strategy not in STRATEGY_NODE_MAP or strategy in failed_strategies:
        return _pick_alternative_strategy(failed_strategies)

    return strategy


# 6b. DECISION RATIONALE (separate LLM reasoning)
#
#     Produces the Strategy_router's OWN rationale
#     for the chosen strategy — distinct from the
#     analyze_circuit_node "reasoning notes", which
#     describe the circuit rather than justify the
#     routing decision. This is what makes the
#     DECISION RATIONALE section stand on its own
#     instead of copying the analyze node's notes.

def _invoke_llm_decision_rationale(
    strategy: str,
    gate_ids: list[int],
    analysis: dict,
    selector_reasoning: str,
    failed_strategies: Optional[list[str]] = None,
) -> str:
    """
    Ask the LLM to write a short, self-contained paragraph explaining WHY
    ``strategy`` was selected for this circuit.

    This is deliberately a different question from the one answered by
    analyze_circuit_node's ``reasoning_notes`` (which characterise the
    circuit's cuttability without naming a strategy). Here the strategy is
    already fixed, so the model justifies *that specific choice* in the
    context of the circuit's metrics and the number of cuts being made.

    Falls back to a deterministic rationale (built from the metrics and the
    selector's own one-liner) if the LLM call fails for any reason, so the
    graph never crashes just because the LLM is unavailable.
    """
    failed_strategies = failed_strategies or []
    try:
        prompt = _build_rationale_prompt(
            strategy, gate_ids, analysis, selector_reasoning, failed_strategies
        )
        rationale = call_llm(
            system_prompt=(
                "You are an expert in quantum circuit partitioning, embedded "
                "in an automated pipeline. A cutting strategy has ALREADY been "
                "chosen for the circuit described to you. Write ONE short "
                "paragraph (2-4 sentences) that justifies why this particular "
                "strategy is a sensible choice for this circuit, referring to "
                "its metrics (connectivity, depth, two-qubit-gate count, "
                "non-Clifford content) and the resulting sampling overhead "
                "where relevant. Do NOT suggest a different strategy and do "
                "NOT restate the metrics as a list. Reply with prose only — "
                "no headers, no bullet points."
            ),
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=600,
        )
        return rationale
    except Exception as exc:
        return (
            f"{_deterministic_decision_rationale(strategy, gate_ids, analysis)} "
            f"[LLM rationale unavailable, used fallback: {exc}]"
        )


def _build_rationale_prompt(
    strategy: str,
    gate_ids: list[int],
    analysis: dict,
    selector_reasoning: str,
    failed_strategies: list[str],
) -> str:
    """User-turn prompt for _invoke_llm_decision_rationale()."""
    failed_block = (
        f"\nStrategies already tried and failed this run: "
        f"{', '.join(failed_strategies)}."
        if failed_strategies else ""
    )
    n_cuts = len(gate_ids) if gate_ids else "to be determined by the cutting node"
    return f"""A circuit-cutting strategy has been selected for the circuit below.

Chosen strategy       : {strategy}
Number of planned cuts: {n_cuts}
Selector's one-line note: {selector_reasoning or 'N/A'}{failed_block}

=== CIRCUIT METRICS ===
- Qubits              : {analysis['num_qubits']}
- Depth               : {analysis['depth']}
- Two-qubit gates     : {analysis['two_qubit_gates']}
- Unique qubit pairs  : {analysis['num_unique_pairs']}
- Connectivity density: {analysis['connectivity_density']}
- Dense?              : {analysis['is_dense']}
- Deep?               : {analysis['is_deep']}
- Non-Clifford gates  : {analysis['non_clifford_count']}

Explain, in 2-4 sentences, why '{strategy}' is an appropriate cutting
strategy for a circuit with these characteristics.
"""


def _deterministic_decision_rationale(
    strategy: str,
    gate_ids: list[int],
    analysis: dict,
) -> str:
    """Fallback rationale used when the LLM call cannot complete."""
    n_cuts = len(gate_ids) if gate_ids else "an automatically determined number of"
    basis = {
        "qpd_gate_cut": (
            f"Gate cutting was chosen to sever {n_cuts} two-qubit gate(s); with "
            f"{analysis['two_qubit_gates']} two-qubit gate(s) and connectivity "
            f"density {analysis['connectivity_density']}, replacing individual "
            "entangling gates keeps the quasi-probability overhead manageable."
        ),
        "auto_finder": (
            "AutoFinder was chosen so the optimiser can search the circuit graph "
            "for the minimum-overhead cut configuration rather than committing to "
            "a hand-picked set of cuts."
        ),
        "llm_custom_cut": (
            "LLM custom cutting was chosen so a bespoke, circuit-specific cut plan "
            "can be proposed for a structure that does not fit the standard "
            "gate- or wire-cut heuristics."
        ),
    }.get(strategy, f"Strategy '{strategy}' was selected for this circuit.")

    if analysis["non_clifford_count"] > 0:
        basis += (
            f" Note that {analysis['non_clifford_count']} non-Clifford gate(s) "
            "will raise the classical reconstruction cost of any cut."
        )
    return basis


# 7.  DECISION FORMATTER
#     Mirrors the _format_*_for_llm() helpers
#     in parse_node.py and analyze_node.py.

def _format_routing_decision(
    strategy: str,
    gate_ids: list[int],
    analysis: dict,
    llm_reasoning: str,
    decision_rationale: str,
) -> str:
    """Return a structured, LLM-readable summary of the routing decision."""
    lines: list[str] = []

    lines += [
        "=== STRATEGY ROUTER DECISION ===",
        f"Chosen strategy     : {strategy}",
        f"Target gate indices : {gate_ids if gate_ids else 'N/A (node selects its own)'}",
        f"Number of cuts      : {len(gate_ids) if gate_ids else 'TBD'}",
        "",
        "=== SELECTOR NOTE (why this strategy over others) ===",
        f"  {llm_reasoning.strip() if llm_reasoning else 'No selector note provided.'}",
        "",
        "=== DECISION RATIONALE ===",
    ]

    # Use the router's OWN LLM-generated rationale (see
    # _invoke_llm_decision_rationale). This is distinct from
    # analyze_circuit_node's reasoning_notes, which describe the circuit
    # rather than justify the routing decision.
    if decision_rationale:
        lines.append(f"  {decision_rationale.strip()}")
    else:
        lines.append("  No decision rationale provided.")

    lines += [
        "",
        "=== STRATEGY NOTES ===",
        {
            "qpd_gate_cut":   "QPD gate cutting replaces selected two-qubit gates with "
                               "TwoQubitQPDGate placeholders.  Overhead scales as "
                               "∏ kappa_i² per cut gate.",
            "auto_finder":    "AutoFinder searches the circuit graph for the minimum-overhead "
                               "cut configuration automatically.",
            "llm_custom_cut": "An LLM analyses the circuit and proposes a bespoke cutting plan.",
        }.get(strategy, "No strategy notes available."),
    ]

    return "\n".join(lines)


# 8.  QUICK SMOKE-TEST  (python strategy_router_node.py)
#     Runs parse → analyze → strategy_router
#     without needing the full subgraph.

if __name__ == "__main__":
    import sys
    import json

    from parse_node   import parse_circuit_node
    from analyze_node import analyze_circuit_node

    path = sys.argv[1] if len(sys.argv) > 1 else "test.qasm"

    # Build a fully-initialised state (all keys must be present)
    state: AgentState = {
        "qasm_path":          path,
        "circuit":            None,
        "circuit_summary":    None,
        "analysis":           None,
        "analysis_summary":   None,
        "cutting_strategy":   None,
        "strategy_reasoning": None,
        "gate_ids":           None,
        "cut_locations":      None,
        "auto_partition":     False,
        "partition_labels":   None,
        "cut_result":         None,
        "failed_strategies":  [],
        "error":              None,
    }

    # Run the three completed nodes in sequence
    state = parse_circuit_node(state)
    state = analyze_circuit_node(state)
    state = strategy_router_node(state)

    if state["error"]:
        print(f"ERROR: {state['error']}")
        sys.exit(1)

    print("=" * 60)
    print("STRATEGY ROUTER OUTPUT")
    print("=" * 60)
    print(state["strategy_reasoning"])

    print("\n" + "=" * 60)
    print("ROUTING DECISION (for subgraph conditional edge)")
    print("=" * 60)
    print(f"  cutting_strategy : {state['cutting_strategy']}")
    print(f"  gate_ids         : {state['gate_ids']}")
    print(f"  auto_partition   : {state['auto_partition']}")
    print(f"  route_to_cutter  → '{route_to_cutter(state)}'")
