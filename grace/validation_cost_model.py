"""
validation_cost_model.py
========================
Adaptive feasibility model for equivalence validation, replacing the old
fixed --max-cuts limit.

Model
-----
    estimated_seconds ~= 6 ** num_cuts * per_variant_cost(num_qubits,
                                                          num_observables)
    per_variant_cost  =  K * num_observables * G ** num_qubits

Fit provenance
--------------
K and G were fit by least squares on log(validator_seconds / 6**num_cuts
/ num_observables) vs num_qubits over the 85 PASS / FAIL_EQUIVALENCE
records in batch_results_llm_v2/results.jsonl that carry
validator_seconds, num_cuts and num_qubits (82 qpd_gate_cut +
3 auto_finder; num_observables reproduced per-strategy exactly as the
validators build their observable batteries). Fit quality on that data:
median actual/predicted = 1.06; 90% of runs fall within [0.4x, 2.6x] of
the prediction. The --validator-timeout backstop still catches the
underestimating tail.

Note: the wire validator was also optimised (8192-shot regime instead of
100k shots), so this model -- fit on pre-optimisation data -- tends to
OVERestimate wire-cut validation cost. That errs on the safe (skip)
side.
"""

from __future__ import annotations

# Fitted constants (see "Fit provenance" above).
COST_K: float = 0.068301   # seconds per variant per observable at nq=0
COST_G: float = 1.0487     # per-qubit multiplicative growth
VARIANTS_PER_CUT: float = 6.0


def estimated_num_observables(num_qubits: int, strategy: str | None) -> int:
    """Reproduce the observable-battery size each validator builds.

    gate / llm_custom (gate path): 3 globals + 3n singles + 2(n-1)
    nearest-neighbour pairs + up to min(6, 2n) random strings.
    wire / auto_finder: capped at 24 (3n + 3 structured, random-filled
    up to the 24 cap).
    """
    n = max(int(num_qubits), 1)
    if strategy in ("qpd_gate_cut", "llm_custom_cut"):
        return 3 + 3 * n + 2 * (n - 1) + min(6, 2 * n)
    return min(24, 3 * n + 3 + 24)


def per_variant_cost(num_qubits: int, num_observables: int) -> float:
    """Estimated seconds to run ONE QPD variant's sub-experiments."""
    return COST_K * max(int(num_observables), 1) * COST_G ** max(int(num_qubits), 0)


def estimated_validation_seconds(
    num_cuts: int,
    num_qubits: int,
    strategy: str | None = None,
    num_observables: int | None = None,
) -> float:
    """Estimated wall-clock seconds for exact equivalence validation."""
    if num_observables is None:
        num_observables = estimated_num_observables(num_qubits, strategy)
    return (VARIANTS_PER_CUT ** max(int(num_cuts), 0)
            * per_variant_cost(num_qubits, num_observables))
