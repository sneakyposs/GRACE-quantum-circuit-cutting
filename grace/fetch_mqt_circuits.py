"""
fetch_mqt_circuits.py
---------------------
Generate benchmark circuits from the Munich Quantum Toolkit Benchmark
Library (mqt.bench) and write them as OpenQASM 2.0 files that GRACE's
parse_node (QuantumCircuit.from_qasm_file) can load.

Requires:  pip install mqt.bench

Examples
--------
# Small smoke set (default benchmarks, 3-6 qubits):
python fetch_mqt_circuits.py --out benchmarks_smoke --min-qubits 3 --max-qubits 6

# Full sweep of every benchmark family from 3 to 12 qubits:
python fetch_mqt_circuits.py --out benchmarks --all --min-qubits 3 --max-qubits 12

Files are named  <benchmark>_n<qubits>.qasm  and are skipped if they
already exist, so re-running is cheap and idempotent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mqt.bench import BenchmarkLevel, get_benchmark
from mqt.bench.benchmarks import get_available_benchmark_names
from qiskit import qasm2

# A sensible default subset: structurally diverse, well-behaved at small
# qubit counts, and interesting for cutting (entangling structure varies).
DEFAULT_BENCHMARKS = [
    "ghz", "dj", "graphstate", "qft", "qftentangled",
    "wstate", "vqe_two_local", "qaoa", "ae", "qpeexact",
    "grover", "qwalk", "randomcircuit", "bv", "qnn",
]


def _normalize(qc):
    """Make an MQT circuit friendly to the GRACE pipeline + validators.

    - remove terminal measurements and barriers (validators compare Pauli
      observable expectation values, so these carry no information and
      only distort metadata gate indexing),
    - decompose composite/custom gates (e.g. dj's 'Oracle') down to
      standard gates so QASM2 round-trips cleanly through parse_node.
    """
    from qiskit.transpiler.passes import RemoveBarriers

    qc = qc.remove_final_measurements(inplace=False)
    qc = RemoveBarriers()(qc)

    # The whole cutting toolchain (find_cuts, cut_gates, cut_wires)
    # requires gates of width <= 2, but several MQT families (adders,
    # qwalk, grover, ...) contain ccx/rccx/mcx. Decompose ONLY the wide
    # gates, repeatedly, until none remain.
    for _ in range(8):
        wide = {inst.operation.name for inst in qc.data
                if inst.operation.num_qubits > 2}
        if not wide:
            break
        qc = qc.decompose(list(wide))

    standard = None
    for _ in range(6):  # bounded: decompose until only basis gates remain
        try:
            import qiskit.circuit.library as _lib  # noqa: F401
            qasm2.dumps(qc)
            custom = [inst.operation.name for inst in qc.data
                      if inst.operation.definition is not None
                      and not inst.operation.name.startswith(("u", "r", "c", "h",
                                                              "x", "y", "z", "s",
                                                              "t", "p", "swap",
                                                              "id"))]
            if not custom:
                standard = qc
                break
        except Exception:
            pass
        qc = qc.decompose()
    return standard if standard is not None else qc


def generate_one(name: str, n_qubits: int, level: BenchmarkLevel,
                 keep_measurements: bool = False) -> str | None:
    """Return QASM2 text for one benchmark instance, or None on failure."""
    try:
        qc = get_benchmark(name, level=level, circuit_size=n_qubits)
        if not keep_measurements:
            qc = _normalize(qc)
    except Exception as exc:
        print(f"  [skip] {name} n={n_qubits}: generation failed ({exc})")
        return None
    try:
        return qasm2.dumps(qc)
    except Exception as exc:
        print(f"  [skip] {name} n={n_qubits}: QASM2 export failed ({exc})")
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="benchmarks", help="Output directory.")
    p.add_argument("--benchmarks", nargs="*", default=None,
                   help="Benchmark names (default: curated subset).")
    p.add_argument("--all", action="store_true",
                   help="Use every benchmark mqt.bench provides.")
    p.add_argument("--min-qubits", type=int, default=3)
    p.add_argument("--max-qubits", type=int, default=8)
    p.add_argument("--level", choices=["alg", "indep"], default="indep",
                   help="'indep' = target-independent (basic gates, plays well "
                        "with QASM2 + cutting). 'alg' = raw algorithm level.")
    p.add_argument("--keep-measurements", action="store_true",
                   help="Keep raw MQT circuits (measurements, barriers, "
                        "composite gates). Default strips/decomposes them "
                        "into pure-unitary standard-gate circuits, which is "
                        "what the equivalence validators expect.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate files that already exist.")
    args = p.parse_args(argv)

    level = BenchmarkLevel.INDEP if args.level == "indep" else BenchmarkLevel.ALG

    if args.all:
        names = get_available_benchmark_names()
    elif args.benchmarks:
        names = args.benchmarks
    else:
        names = DEFAULT_BENCHMARKS

    available = set(get_available_benchmark_names())
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"warning: unknown benchmarks ignored: {unknown}")
        names = [n for n in names if n in available]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = 0
    for name in names:
        for n in range(args.min_qubits, args.max_qubits + 1):
            dest = out_dir / f"{name}_n{n}.qasm"
            if dest.exists() and not args.force:
                skipped += 1
                continue
            qasm = generate_one(name, n, level,
                                keep_measurements=args.keep_measurements)
            if qasm is None:
                failed += 1
                continue
            dest.write_text(qasm)
            written += 1
            print(f"  [ok]   {dest}")

    print(f"\nDone: {written} written, {skipped} already existed, "
          f"{failed} failed to generate -> {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
