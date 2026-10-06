"""
validate_node.py
----------------
LangGraph Node 8 — Validate

Current responsibilities:
1. Confirm the circuit was altered (a cut circuit exists and differs from the original).
2. Confirm that subcircuits were created.
3. Confirm that actual cuts were made -- an uncut circuit is REFUSED as a
   "result" (fails loudly and retries another strategy). The one sanctioned
   exception is AutoFinder legitimately determining that no cuts are needed;
   that outcome is accepted but annotated in the run's metadata
   (``no_cut_note``).
4. Run the strategy's offline EQUIVALENCE validator on the produced
   sub-circuits (gate_validate / wire_validate / auto_finder_validate /
   validate_llm_custom_cut) and require it to PASS -- i.e. the reconstructed
   expectation values must match the original circuit within the shared
   validation tolerance (cutting_runs/validation_tolerance.py). Validation
   only succeeds when subcircuits exist AND the equivalence validator passes
   AND the tolerance is satisfied.
   Exception: when the estimated validation cost (validation_cost_model.py)
   exceeds the equivalence budget, the equivalence step is SKIPPED and
   recorded as ``equivalence_skipped='infeasible'``.
5. Re-route back to Strategy_router if validation fails.
   Infeasibility (see point 4) is NON-TERMINAL: an over-cut first attempt
   whose exact equivalence check exceeds budget is rerouted to a leaner
   strategy (the min-cut qpd_gate_cut usually lands under the exact
   ceiling) whenever an untried active strategy and a retry remain. Only
   when nothing leaner is left is a terminal VALIDATION_INFEASIBLE
   conceded, emitted as an accepted-infeasible state so the batch harness
   classifies it as VALIDATION_INFEASIBLE (see _handle_infeasible).
6. Stop retrying after a configurable maximum number of attempts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from parse_node import AgentState


# Configuration

# Change this value later if you want more/fewer retries.
MAX_VALIDATION_LOOPS = 3

_PROJECT_ROOT = Path(__file__).resolve().parent

# Strategy name -> offline equivalence validator script (same mapping the
# batch harness uses).
EQUIV_VALIDATORS: dict[str, Path] = {
    "qpd_gate_cut":   _PROJECT_ROOT / "cutting_runs" / "gate_validate.py",
    "auto_finder":    _PROJECT_ROOT / "cutting_runs" / "auto_finder_validate.py",
    "llm_custom_cut": _PROJECT_ROOT / "cutting_runs" / "validate_llm_custom_cut.py",
}

# Wall-clock budget (seconds) for the in-graph equivalence check. Used both
# as the feasibility threshold for the cost model AND as the subprocess
# timeout. The batch harness overrides this via the environment so it stays
# consistent with --validator-timeout.
DEFAULT_EQUIV_BUDGET_SECONDS = 900.0


def _equiv_budget_seconds() -> float:
    try:
        return float(os.environ.get("GRACE_EQUIV_BUDGET_SECONDS", ""))
    except (TypeError, ValueError):
        return DEFAULT_EQUIV_BUDGET_SECONDS


def validate_node(state: AgentState) -> AgentState:
    """
    Node 8 – Validate

    Reads the cutting output written by a cutting node and determines
    whether the produced partition is acceptable.

    Writes:
        validation_passed : bool
        validation_reason : str
        validation_attempts : int
    """

    attempts = state.get("validation_attempts", 0)
    # Copy (don't mutate state's list in place) so we can append safely.
    failed_strategies: list[str] = list(state.get("failed_strategies") or [])

    passed, reason, updates = _run_validation_checks(state)

    # ── Option A: equivalence-infeasible is NON-TERMINAL ────────────────
    # When the cost model says this strategy's exact equivalence check is
    # unaffordable within budget, we do NOT accept the (unverified) cut as a
    # result. Instead reroute to a leaner strategy — the min-cut
    # qpd_gate_cut usually uses far fewer cuts and lands under the exact
    # ceiling — as long as an untried active strategy remains and we still
    # have retries left. Only when nothing leaner is left do we concede a
    # terminal VALIDATION_INFEASIBLE. See _handle_infeasible for how the
    # terminal state is shaped so the batch harness classifies it correctly.
    if not passed and updates.get("equivalence_skipped") == "infeasible":
        return _handle_infeasible(
            state, attempts, failed_strategies, reason, updates
        )

    #print if passed
    if passed:
        print(f"[VALIDATE] SUCCESS: {reason}")
    else:
        print(f"[VALIDATE] FAILED: {reason}")
        print(
            f"[VALIDATE] Retry "
            f"{attempts + 1}/{MAX_VALIDATION_LOOPS} "
            f"-> Routing back to Strategy_router"
        )

    if passed:
        # Flag runs where the LLM was unavailable/hit token limit and a
        # deterministic heuristic was silently substituted. These passes
        # are real (equivalence was verified) but should NOT be counted
        # as genuine LLM-proposed successes in batch analysis.
        llm_fallback = state.get("llm_fallback", False)
        is_llm_strategy = (state.get("cutting_strategy") == "llm_custom_cut")
        if llm_fallback and is_llm_strategy:
            reason = f"PASS_LLM_FALLBACK: {reason}"
            print("[VALIDATE] NOTE: this pass used a deterministic fallback, "
                  "NOT a genuine LLM-proposed plan.")

        return {
            **state,
            **updates,
            "validation_passed": True,
            "validation_reason": reason,
            "validation_attempts": attempts,
            "failed_strategies": failed_strategies,
        }

    # not to pick it again on retry. Without this, the router has no way
    # of knowing a strategy was already tried and failed, so it (or its
    # fallback) keeps re-selecting the same failing strategy until
    # MAX_VALIDATION_LOOPS is exhausted.
    failed_strategy = state.get("cutting_strategy")
    if failed_strategy and failed_strategy not in failed_strategies:
        failed_strategies.append(failed_strategy)
        print(f"[VALIDATE] Marking strategy '{failed_strategy}' as failed. "
              f"Failed so far: {failed_strategies}")

    attempts += 1

    return {
        **state,
        **updates,
        "validation_passed": False,
        "validation_reason": reason,
        "validation_attempts": attempts,
        "failed_strategies": failed_strategies,
        # Without this, strategy_router's early-return guard
        #   (if state.get("error"): return state)
        # short-circuits on every subsequent retry, preventing fresh
        # gate_ids / cutting_strategy computation for the next strategy.
        "error": None,
    }


def _handle_infeasible(
    state: AgentState,
    attempts: int,
    failed_strategies: list[str],
    reason: str,
    updates: dict[str, Any],
) -> AgentState:
    """Decide what to do when this strategy's exact equivalence check is
    infeasible within budget (Option A: infeasibility is non-terminal).

    Reroute to a leaner strategy when one is still untried AND a retry
    remains; otherwise concede a terminal VALIDATION_INFEASIBLE.

    The terminal concede is emitted as an ACCEPTED-infeasible state
    (``validation_passed=True`` with ``equivalence_skipped='infeasible'``),
    which is the same shape the pre-Option-A code produced. That matters:
    the batch harness classifies VALIDATION_INFEASIBLE from
    ``equivalence_skipped`` only AFTER its in-graph gate lets the record
    through, and that gate rejects any run whose ``validation_passed`` is
    not True (recording it as PIPELINE_VALIDATION_FAILED instead). Keeping
    the concede as an accepted state is therefore what preserves the
    INFEASIBLE classification without touching the harness.

    "Last attempt decides": if some earlier attempt failed equivalence
    *within* budget while the final attempt is infeasible, the run is still
    conceded INFEASIBLE here — the terminal outcome follows whatever the
    last attempt was. A feasible equivalence FAIL on the final attempt does
    not reach this function (it takes the genuine-failure path in
    validate_node) and so ends as a FAIL, as intended.
    """
    current = state.get("cutting_strategy")

    # What failed_strategies would look like once we mark the current one.
    would_be_failed = list(failed_strategies)
    if current and current not in would_be_failed:
        would_be_failed.append(current)

    try:
        from strategy_router import remaining_active_strategies
        untried = remaining_active_strategies(would_be_failed)
    except Exception as exc:  # never let a router import break validation
        # Fail safe by conceding: a correctly-labelled INFEASIBLE beats an
        # unbounded reroute loop if the router can't be consulted.
        print(f"[VALIDATE] Could not determine remaining strategies "
              f"({exc}); conceding infeasible.")
        untried = []

    # A reroute only leads to another cutting attempt if route_after_validation
    # won't immediately end the graph — i.e. the post-increment attempt count
    # is still below MAX_VALIDATION_LOOPS.
    retries_remain = (attempts + 1) < MAX_VALIDATION_LOOPS

    if untried and retries_remain:
        if current and current not in failed_strategies:
            failed_strategies.append(current)
        attempts += 1
        print(
            f"[VALIDATE] INFEASIBLE for '{current}': exact equivalence check "
            f"exceeds budget. Non-terminal -> rerouting to a leaner strategy "
            f"(untried: {untried}). "
            f"Retry {attempts}/{MAX_VALIDATION_LOOPS} -> Strategy_router."
        )
        return {
            **state,
            **updates,
            "validation_passed": False,
            "validation_reason": (
                f"Equivalence infeasible for '{current}'; rerouting to a "
                f"leaner strategy. {reason}"
            ),
            "validation_attempts": attempts,
            "failed_strategies": failed_strategies,
            # short-circuit on the retry (same fix as the normal-failure
            # path above).
            "error": None,
        }

    # Terminal concede: nothing leaner remains (or retries exhausted). Emit
    # the accepted-infeasible state (see docstring) so the harness records
    # VALIDATION_INFEASIBLE.
    print(
        f"[VALIDATE] INFEASIBLE for '{current}' and no leaner strategy remains "
        f"(untried={untried}, attempts={attempts}). Conceding "
        f"VALIDATION_INFEASIBLE."
    )
    return {
        **state,
        **updates,
        "validation_passed": True,
        "validation_reason": (
            f"Validation conceded infeasible: no strategy could certify "
            f"equivalence within budget. {reason}"
        ),
        "validation_attempts": attempts,
        "failed_strategies": failed_strategies,
    }


def _run_validation_checks(state: dict[str, Any]) -> tuple[bool, str, dict]:
    """
    Central validation function.

    Order of requirements (ALL must hold for success):
      1. structural checks: a modified circuit exists, one or more valid
         subcircuits were produced, the partition is non-trivial;
      2. actual cuts were made (uncut "results" are refused, with the
         single annotated AutoFinder no-cut exception);
      3. the strategy's equivalence validator passes within the shared
         validation tolerance (skipped only when the cost model says the
         check is infeasible within the budget, or for the sanctioned
         no-cut case where there is nothing to reconstruct).

    Returns
    -------
    (passed, reason, updates)
        ``updates`` is a dict of extra state fields to merge (equivalence
        results, annotated metadata).
    """
    updates: dict[str, Any] = {
        "equivalence_passed": None,
        "equivalence_detail": None,
        "equivalence_seconds": None,
        "equivalence_skipped": None,
    }

    structural_checks = [
        _has_cut_circuit,
        _has_subcircuits,
        _partition_is_nontrivial,
    ]
    for check in structural_checks:
        passed, reason = check(state)
        if not passed:
            return False, reason, updates

    # ── Requirement: actual cuts were made ─────────────────────────────
    cuts_ok, cuts_reason, legit_no_cut, meta_update = _check_cuts_were_made(state)
    if meta_update is not None:
        updates["metadata"] = meta_update
    if not cuts_ok:
        return False, cuts_reason, updates

    if legit_no_cut:
        # Nothing was cut, legitimately -- there are no subcircuit
        # decompositions to reconstruct, so the equivalence check is
        # vacuous. Accept, with the note recorded in metadata.
        updates["equivalence_skipped"] = "no_cut"
        updates["equivalence_detail"] = cuts_reason
        return True, f"Validation passed (no-cut). {cuts_reason}", updates

    # ── Requirement: equivalence validator must PASS ────────────────────
    eq = _check_equivalence(state)
    updates.update(eq)

    if eq.get("equivalence_skipped") == "infeasible":
        # Estimated validation cost exceeds the budget for THIS strategy.
        # This is no longer a terminal pass: validate_node decides whether to
        # reroute to a leaner strategy or concede a terminal
        # VALIDATION_INFEASIBLE (see validate_node / _handle_infeasible). We
        # report it as not-passed here and leave equivalence_skipped set in
        # ``updates`` so validate_node can recognise the infeasible case.
        return False, (
            f"Equivalence check infeasible for strategy "
            f"'{state.get('cutting_strategy')}': {eq.get('equivalence_detail')}"
        ), updates

    if eq.get("equivalence_passed"):
        return True, (
            f"Validation passed. Equivalence validator PASSED in "
            f"{eq.get('equivalence_seconds')}s."
        ), updates

    return False, (
        f"Equivalence validation failed for strategy "
        f"'{state.get('cutting_strategy')}': {eq.get('equivalence_detail')}"
    ), updates


def _count_actual_cuts(state: dict[str, Any]) -> int:
    """Best-effort count of how many cuts were actually performed.

    Combines the strategy-specific target lists with a direct scan of the
    produced circuits for QPD / cut placeholder instructions, taking the
    maximum so a stale or missing field can't hide a real cut (or fake one:
    every source counts *placed* cut artifacts, not intentions).
    """
    strategy = (state.get("cutting_strategy") or "").lower()
    meta = state.get("metadata") or {}
    counts: list[int] = []

    if strategy in ("qpd_gate_cut", "llm_custom_cut"):
        counts.append(len(state.get("gate_ids") or []))
    if strategy == "llm_custom_cut":
        counts.append(len(state.get("cut_locations") or []))
    if strategy == "auto_finder":
        cuts = meta.get("cuts")
        if isinstance(cuts, list):
            counts.append(len(cuts))
        num_cuts = meta.get("num_cuts")
        if isinstance(num_cuts, (int, float)):
            counts.append(int(num_cuts))

    marked = state.get("marked_circuit")
    if marked is not None:
        try:
            counts.append(sum(
                1 for inst in marked.data
                if inst.operation.name == "cut_wire"
            ))
        except Exception:
            pass

    cut = state.get("cut_circuit")
    if cut is not None:
        try:
            counts.append(sum(
                1 for inst in cut.data
                if inst.operation.name.lower().startswith(("qpd", "cut_", "move"))
            ))
        except Exception:
            pass

    return max(counts) if counts else 0


def _check_cuts_were_made(
    state: dict[str, Any],
) -> tuple[bool, str, bool, dict | None]:
    """Refuse to emit an uncut circuit as a "result".

    Returns
    -------
    (passed, reason, legit_no_cut, metadata_update)
        ``legit_no_cut`` is True only for the sanctioned AutoFinder
        zero-cut outcome; ``metadata_update`` carries the annotated
        metadata dict in that case (None otherwise).
    """
    n_cuts = _count_actual_cuts(state)
    if n_cuts > 0:
        return True, f"{n_cuts} cut(s) were made.", False, None

    strategy = (state.get("cutting_strategy") or "").lower()
    meta = state.get("metadata") or {}

    if strategy == "auto_finder":
        # Legitimate no-cut outcome: find_cuts() may correctly determine
        # that the circuit already fits the device constraints and return
        # zero cuts. Accepted, but annotated in metadata so it is never
        # mistaken for a genuine cut result downstream.
        cut = state.get("cut_circuit")
        no_cuts_reported = meta.get("cuts") == [] if "cuts" in meta else None
        has_qpd_ops = any(
            inst.operation.name.startswith(("qpd", "cut_", "move"))
            for inst in getattr(cut, "data", [])
        ) if cut is not None else False
        if no_cuts_reported or (no_cuts_reported is None and not has_qpd_ops):
            note = (
                "NO CUTS MADE: AutoFinder determined no cuts are needed "
                "(circuit already fits the configured constraints). "
                "Accepted as a legitimate no-cut outcome; the 'subcircuits' "
                "are the original circuit, not a cut decomposition."
            )
            annotated = dict(meta)
            annotated["no_cut_note"] = note
            return True, note, True, annotated

    return False, (
        f"NO CUTS WERE MADE (strategy='{strategy or 'unknown'}'): refusing "
        f"to emit an uncut circuit as a result. Failing loudly so another "
        f"strategy can be retried."
    ), False, None


def _check_equivalence(state: dict[str, Any]) -> dict[str, Any]:
    """Run the strategy's offline equivalence validator on this attempt.

    The current state is exported to a temporary run directory (the same
    artifact layout the validators already consume) and the matching
    validator script is executed in a subprocess. Exit codes follow the
    validators' CI convention: 0 = pass, 1 = equivalence FAIL, anything
    else = validator error (treated as a failure -- an unverifiable cut is
    not a certified cut).

    Feasibility guard: when the fitted cost model predicts the check
    cannot finish within the budget, it is skipped and recorded as
    ``equivalence_skipped='infeasible'`` (the batch harness then reports
    VALIDATION_INFEASIBLE, exactly as it did before this check moved
    in-graph).
    """
    out: dict[str, Any] = {
        "equivalence_passed": None,
        "equivalence_detail": None,
        "equivalence_seconds": None,
        "equivalence_skipped": None,
    }

    strategy = (state.get("cutting_strategy") or "").lower()
    script = EQUIV_VALIDATORS.get(strategy)
    if script is None or not script.exists():
        out["equivalence_passed"] = False
        out["equivalence_detail"] = (
            f"no equivalence validator found for strategy {strategy!r}"
        )
        return out

    budget = _equiv_budget_seconds()

    # ── Feasibility guard (adaptive, cost-model based) ──────────────────
    try:
        from validation_cost_model import estimated_validation_seconds
        original = state.get("original_circuit") or state.get("circuit")
        num_qubits = int(getattr(original, "num_qubits", 0) or 0)
        n_cuts = _count_actual_cuts(state)
        est = estimated_validation_seconds(n_cuts, num_qubits, strategy)
        if est > budget:
            out["equivalence_skipped"] = "infeasible"
            out["equivalence_detail"] = (
                f"estimated validation cost {est:.0f}s (cuts={n_cuts}, "
                f"qubits={num_qubits}, model 6^cuts * per_variant_cost) "
                f"exceeds budget {budget:.0f}s"
            )
            print(f"[VALIDATE] Equivalence check skipped: "
                  f"{out['equivalence_detail']}")
            return out
    except Exception as exc:  # cost model must never break validation
        print(f"[VALIDATE] Cost-model estimate unavailable ({exc}); "
              f"attempting equivalence check anyway.")

    # ── Export current attempt to a temp run dir and dispatch ───────────
    tmp_dir = tempfile.mkdtemp(prefix="grace_equiv_")
    t0 = time.monotonic()
    try:
        from inspection_export import export_cutting_results
        run_dir = export_cutting_results(state, output_dir=tmp_dir)

        cmd = [sys.executable, str(script), str(run_dir)]
        print(f"[VALIDATE] Running equivalence validator: {script.name} "
              f"(budget {budget:.0f}s)")
        try:
            proc = subprocess.run(
                cmd, cwd=str(script.parent), capture_output=True,
                text=True, timeout=budget,
            )
        except subprocess.TimeoutExpired:
            out["equivalence_passed"] = False
            out["equivalence_detail"] = (
                f"equivalence validator timed out after {budget:.0f}s"
            )
            return out

        tail = ((proc.stdout or "") + (proc.stderr or "")).strip()[-1500:]
        out["equivalence_detail"] = tail
        if proc.returncode == 0:
            out["equivalence_passed"] = True
        elif proc.returncode == 1:
            out["equivalence_passed"] = False
        else:
            out["equivalence_passed"] = False
            out["equivalence_detail"] = (
                f"validator errored (exit {proc.returncode}): {tail}"
            )
        return out
    except Exception as exc:
        out["equivalence_passed"] = False
        out["equivalence_detail"] = f"equivalence check could not run: {exc}"
        return out
    finally:
        out["equivalence_seconds"] = round(time.monotonic() - t0, 3)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _has_cut_circuit(state: dict[str, Any]) -> tuple[bool, str]:
    """Confirm a modified circuit exists."""

    original = state.get("original_circuit")
    cut = state.get("cut_circuit")

    if original is None or cut is None:
        return False, "Missing original or cut circuit."

    # Simple demo heuristic:
    # If they are the exact same object, nothing was altered.
    if original is cut:
        return False, "Circuit was not modified."

    return True, "Circuit was altered."


def _has_subcircuits(state: dict[str, Any]) -> tuple[bool, str]:
    """Confirm at least one subcircuit was generated."""

    subcircuits = state.get("subcircuits")

    if not subcircuits:
        return False, "No subcircuits were generated."

    return True, "Subcircuits exist."


def _partition_is_nontrivial(state: dict[str, Any]) -> tuple[bool, str]:
    """Confirm the partitioning actually did something.

     a run could "pass" with a single subcircuit that was
    just the whole original circuit (this happened when a leftover barrier
    kept all qubits connected, so partition_problem produced one partition
    spanning every qubit). The entire point of circuit cutting is to reduce
    circuit width, so require either (a) more than one partition, or
    (b) every subcircuit strictly narrower than the original circuit.
    """
    subcircuits = state.get("subcircuits")
    original = state.get("original_circuit")
    if not subcircuits or original is None:
        # _has_subcircuits / _has_cut_circuit already cover the missing case.
        return True, "Partition triviality check skipped (nothing to check)."

    try:
        n_orig = original.num_qubits
        widths = [sc.num_qubits for sc in subcircuits.values()]
    except Exception as exc:
        return False, f"Could not inspect subcircuit widths: {exc}"

    if len(widths) > 1:
        return True, f"Partitioned into {len(widths)} subcircuits (widths {widths})."
    if widths and widths[0] < n_orig:
        return True, (f"Single subcircuit but width reduced "
                      f"{n_orig} -> {widths[0]}.")

    # Legitimate no-cut outcome: auto_finder's find_cuts() may correctly
    # determine that the circuit already fits the device constraints and
    # return zero cuts (metadata {'cuts': [], ...}). That is a valid result,
    # not a failure — only flag a trivial partition when cuts were actually
    # supposed to have been made.
    if state.get("cutting_strategy") == "auto_finder":
        cut = state.get("cut_circuit")
        meta = state.get("metadata") or {}
        no_cuts_reported = meta.get("cuts") == [] if "cuts" in meta else None
        has_qpd_ops = any(
            inst.operation.name.startswith(("qpd", "cut_", "move"))
            for inst in getattr(cut, "data", [])
        ) if cut is not None else False
        if no_cuts_reported or (no_cuts_reported is None and not has_qpd_ops):
            return True, (
                "AutoFinder determined no cuts are needed (circuit already "
                "fits constraints); trivial partition accepted."
            )

    return False, (
        f"Trivial partition: 1 subcircuit spanning all {n_orig} qubit(s) — "
        f"the cut did not actually separate the circuit."
    )


def route_after_validation(state: dict[str, Any]) -> str:
    """
    Conditional edge router.

    Returns:
        'end'              -> validation succeeded
        'strategy_router'  -> retry another cutting strategy
        'end'              -> retry limit exceeded
    """

    if state.get("validation_passed"):
        return "end"

    attempts = state.get("validation_attempts", 0)

    if attempts >= MAX_VALIDATION_LOOPS:
        print(
            f"[VALIDATE] Maximum retries "
            f"({MAX_VALIDATION_LOOPS}) reached. Ending workflow."
        )
        return "end"

    return "strategy_router"
