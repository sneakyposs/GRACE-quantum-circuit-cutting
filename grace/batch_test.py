"""
batch_test.py
-------------
Robust, resumable batch tester for the GRACE circuit-cutting pipeline.

For every QASM circuit it:
  1. runs the full LangGraph workflow (Parse -> Analyze -> Strategy_router
     -> cutting node -> Validate) in an ISOLATED SUBPROCESS with a timeout.
     The Validate node itself now runs the matching offline equivalence
     validator (gate_validate / wire_validate / auto_finder_validate /
     validate_llm_custom_cut) on every attempt, so a run only passes
     in-graph validation when its subcircuits were certified equivalent
     within the shared tolerance (or the check was provably infeasible),
  2. exports the run artifacts into batch_results/runs/<circuit>/,
  3. reads the in-graph equivalence verdict from the worker record
     (the offline validator is only dispatched here as a legacy fallback
     when the worker carried no verdict, so nothing is validated twice),
  4. appends one JSON record to batch_results/results.jsonl.

PAUSE / RESUME
--------------
Results are appended to results.jsonl one line per circuit, written and
flushed immediately. Kill the process at any point (Ctrl+C, crash, power
loss) and simply re-run the same command: circuits that already have a
record are skipped. Nothing is ever lost mid-run because no record is
written until a circuit fully finishes.

Because both the pipeline and the validator run in child processes, a
segfault or infinite loop on one pathological circuit cannot take down
the batch — it is recorded as TIMEOUT_* or *_ERROR and the run moves on.

USAGE
-----
# Generate circuits first (see fetch_mqt_circuits.py), then:
python batch_test.py --circuits benchmarks

# Common options:
python batch_test.py --circuits benchmarks \
    --results-dir batch_results \
    --pipeline-timeout 300 --validator-timeout 600 \
    --max-qubits 12 --limit 50

# LLM unavailable / offline run (forces deterministic fallbacks, fast):
python batch_test.py --circuits benchmarks --no-llm

# Re-run only circuits that failed equivalence or errored:
python batch_test.py --circuits benchmarks --rerun failed
python batch_test.py --circuits benchmarks --rerun errors

# Print an aggregate report (also writes summary.csv):
python batch_test.py --summarize --results-dir batch_results

STATUS VOCABULARY
-----------------
PASS                        pipeline ran, subcircuits equivalent to original
FAIL_EQUIVALENCE            offline validator found a mismatch (legacy
                            fallback path only; equivalence failures are
                            normally retried in-graph and end up as
                            PIPELINE_VALIDATION_FAILED when every strategy
                            fails)
PIPELINE_VALIDATION_FAILED  in-graph Validate node never accepted a cut
                            (structural failure, no cuts made, or
                            equivalence mismatch on every attempt)
PIPELINE_ERROR              exception inside the LangGraph workflow
VALIDATOR_ERROR             offline validator crashed / couldn't validate
TIMEOUT_PIPELINE            workflow exceeded --pipeline-timeout
TIMEOUT_VALIDATOR           validator exceeded --validator-timeout
SKIPPED_TOO_LARGE           circuit exceeds --max-qubits (statevector cap)
NO_VALIDATOR                unknown strategy name (shouldn't happen)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WORKER = PROJECT_ROOT / "batch_worker.py"
RESULT_MARKER = "@@RESULT@@"

# Strategy name -> offline equivalence validator script.
VALIDATORS = {
    "qpd_gate_cut":   PROJECT_ROOT / "cutting_runs" / "gate_validate.py",
    "auto_finder":    PROJECT_ROOT / "cutting_runs" / "auto_finder_validate.py",
    "llm_custom_cut": PROJECT_ROOT / "cutting_runs" / "validate_llm_custom_cut.py",
}

# Extra CLI args passed to each validator. Edit here if you want e.g.
# different shot counts or tolerances for a sweep. All validators use
# exit codes 0=pass, 1=fail, 2=error, which is what the harness reads.
VALIDATOR_EXTRA_ARGS: dict[str, list[str]] = {
    "qpd_gate_cut":   [],
    "auto_finder":    [],
    "llm_custom_cut": [],
}

_QREG_RE = re.compile(r"^\s*qreg\s+\w+\s*\[\s*(\d+)\s*\]", re.MULTILINE)


# Interrupt-insulated child processes
# Ctrl+C in the console is delivered to the WHOLE process group, so without
# isolation an in-flight pipeline/validator child dies with KeyboardInterrupt
# and gets recorded as a phantom PIPELINE_ERROR / VALIDATOR_ERROR. Children
# are therefore started in their own process group / session: the parent's
# graceful-interrupt handler stays in charge, in-flight circuits finish and
# are recorded, and a second Ctrl+C explicitly kills the registered children.

_ACTIVE_PROCS: set = set()
_PROC_LOCK = threading.Lock()

if os.name == "nt":
    _ISOLATE = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
else:
    _ISOLATE = {"start_new_session": True}


def _run_child(cmd, cwd, env, timeout):
    """subprocess.run equivalent, isolated from console Ctrl+C.

    Returns (returncode, stdout, stderr); raises subprocess.TimeoutExpired
    after killing the child on timeout.
    """
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **_ISOLATE)
    with _PROC_LOCK:
        _ACTIVE_PROCS.add(proc)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        return proc.returncode, out, err
    finally:
        with _PROC_LOCK:
            _ACTIVE_PROCS.discard(proc)


def _kill_active_children():
    with _PROC_LOCK:
        procs = list(_ACTIVE_PROCS)
    for proc in procs:
        try:
            proc.kill()
        except Exception:
            pass


# Helpers

def count_qubits(qasm_path: Path) -> int | None:
    """Cheap qubit count from qreg declarations (no qiskit parse needed)."""
    try:
        text = qasm_path.read_text(errors="replace")
    except OSError:
        return None
    sizes = [int(m) for m in _QREG_RE.findall(text)]
    return sum(sizes) if sizes else None


def discover_circuits(paths: list[str]) -> list[Path]:
    """Expand dirs / globs / files into a sorted, de-duplicated list."""
    found: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.qasm")))
        elif any(ch in p for ch in "*?["):
            found.extend(sorted(Path().glob(p)))
        elif path.is_file():
            found.append(path)
        else:
            print(f"warning: {p} not found, ignoring")
    seen, unique = set(), []
    for f in found:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(f)
    return unique


def load_previous(results_file: Path) -> dict[str, dict]:
    """Latest record per circuit key from the append-only JSONL log."""
    records: dict[str, dict] = {}
    if not results_file.exists():
        return records
    with results_file.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records[rec["circuit"]] = rec  # later lines override earlier
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a hard kill
    return records


def append_record(results_file: Path, record: dict) -> None:
    """Append one finished-circuit record, flushed to disk immediately."""
    with results_file.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


ERROR_STATUSES = {"PIPELINE_ERROR", "VALIDATOR_ERROR",
                  "TIMEOUT_PIPELINE", "TIMEOUT_VALIDATOR", "NO_VALIDATOR"}
FAIL_STATUSES = {"FAIL_EQUIVALENCE", "PIPELINE_VALIDATION_FAILED"}


def should_skip(prev: dict | None, rerun: str) -> bool:
    if prev is None:
        return False
    if rerun == "all":
        return False
    status = prev.get("status", "")
    if rerun == "failed":
        return status not in (FAIL_STATUSES | ERROR_STATUSES)
    if rerun == "errors":
        return status not in ERROR_STATUSES
    return True  # rerun == "none": anything recorded is done


# Per-circuit processing

def run_pipeline(qasm: Path, export_dir: Path, timeout: int,
                 env: dict) -> tuple[dict | None, str, str]:
    """Run batch_worker.py; return (worker_result_or_None, status, detail)."""
    cmd = [sys.executable, str(WORKER), str(qasm), "--export-dir", str(export_dir)]
    try:
        _rc, _out, _err = _run_child(cmd, str(PROJECT_ROOT), env, timeout)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT_PIPELINE", f"exceeded {timeout}s"

    m = re.search(f"{RESULT_MARKER}(.*?){RESULT_MARKER}", _out, re.DOTALL)
    if not m:
        tail = (_err or _out or "")[-2000:]
        return None, "PIPELINE_ERROR", f"worker produced no result record:\n{tail}"

    result = json.loads(m.group(1))
    if result.get("pipeline_error"):
        return result, "PIPELINE_ERROR", str(result["pipeline_error"])[-2000:]
    if result.get("in_graph_validation_passed") is not True:
        eq_note = ""
        if result.get("equivalence_detail"):
            eq_note = ("; last equivalence result: "
                       + str(result["equivalence_detail"])[-600:])
        return result, "PIPELINE_VALIDATION_FAILED", (
            f"in-graph Validate rejected all attempts "
            f"(attempts={result.get('validation_attempts')}, "
            f"failed_strategies={result.get('failed_strategies')})"
            + eq_note)
    if not result.get("run_dir"):
        return result, "PIPELINE_ERROR", "no run_dir exported"
    return result, "OK", ""


def run_validator(strategy: str, run_dir: str, timeout: int,
                  env: dict) -> tuple[str, str, float]:
    """Dispatch the strategy's validator; return (status, detail, seconds)."""
    script = VALIDATORS.get(strategy)
    if script is None or not script.exists():
        return "NO_VALIDATOR", f"no validator for strategy {strategy!r}", 0.0

    cmd = [sys.executable, str(script), run_dir,
           *VALIDATOR_EXTRA_ARGS.get(strategy, [])]
    t0 = time.monotonic()
    try:
        rc, out, err = _run_child(cmd, str(script.parent), env, timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT_VALIDATOR", f"exceeded {timeout}s", time.monotonic() - t0
    secs = time.monotonic() - t0

    output = (out or "") + (err or "")
    tail = output.strip()[-1500:]
    if rc == 0:
        return "PASS", tail, secs
    if rc == 1:
        return "FAIL_EQUIVALENCE", tail, secs
    return "VALIDATOR_ERROR", tail, secs


# Summary

def summarize(results_file: Path, csv_out: Path | None = None) -> None:
    records = load_previous(results_file)
    if not records:
        print(f"No records in {results_file}.")
        return

    by_status = Counter(r.get("status", "?") for r in records.values())
    by_strategy: dict[str, Counter] = defaultdict(Counter)
    for r in records.values():
        by_strategy[r.get("strategy") or "-"][r.get("status", "?")] += 1

    total = len(records)
    passed = by_status.get("PASS", 0)
    no_cut = by_status.get("PASS_NO_CUT", 0)
    print(f"\n{'=' * 64}\nBATCH SUMMARY  ({results_file})\n{'=' * 64}")
    print(f"Circuits recorded : {total}")
    print(f"Genuine PASS      : {passed}  ({100 * passed / total:.1f}%)")
    if no_cut:
        print(f"PASS_NO_CUT       : {no_cut}  (auto_finder no-cut bypasses, "
              f"excluded from pass rate)")
    print("By status:")
    for status, n in by_status.most_common():
        print(f"  {status:<28} {n}")
    print("\nBy strategy:")
    for strat, counts in sorted(by_strategy.items()):
        n = sum(counts.values())
        p = counts.get("PASS", 0)
        print(f"  {strat:<16} {p}/{n} passed   {dict(counts)}")

    if csv_out:
        fields = ["circuit", "status", "strategy", "num_qubits", "depth",
                  "in_graph_validation_passed", "validation_attempts",
                  "pipeline_seconds", "validator_seconds", "sampling_overhead", "num_cuts", "run_dir",
                  "timestamp", "detail"]
        with csv_out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for key in sorted(records):
                rec = dict(records[key])
                rec["detail"] = (rec.get("detail") or "").replace("\n", " | ")[:500]
                w.writerow(rec)
        print(f"\nWrote {csv_out}")


# Main loop

def process_one(qasm: Path, args, env: dict, results_dir: Path) -> dict:
    """Full treatment of one circuit: pipeline subprocess + validator
    subprocess. Returns the finished record (does not write it)."""
    record = {
        "circuit": qasm.name,
        "qasm_path": str(qasm),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": None, "strategy": None,
        "num_qubits": None, "depth": None,
        "in_graph_validation_passed": None, "validation_attempts": None,
        "pipeline_seconds": None, "validator_seconds": None,
        "sampling_overhead": None, "num_cuts": None,
        "run_dir": None, "detail": "",
    }

    nq = count_qubits(qasm)
    record["num_qubits"] = nq
    if nq is not None and nq > args.max_qubits:
        record["status"] = "SKIPPED_TOO_LARGE"
        record["detail"] = f"{nq} qubits > --max-qubits {args.max_qubits}"
        return record

    export_dir = (results_dir / "runs" / qasm.stem).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    # The Validate node now runs the equivalence validator INSIDE the
    # pipeline subprocess (up to 1 + MAX_VALIDATION_LOOPS = 4 attempts),
    # so the equivalence budget must be added to the pipeline timeout.
    budget = args.validation_budget or args.validator_timeout
    pipeline_timeout = args.pipeline_timeout + 4 * budget
    worker, status, detail = run_pipeline(
        qasm.resolve(), export_dir, pipeline_timeout, env)

    if worker:
        record["strategy"] = worker.get("strategy")
        record["num_qubits"] = worker.get("num_qubits") or nq
        record["depth"] = worker.get("depth")
        record["in_graph_validation_passed"] = worker.get(
            "in_graph_validation_passed")
        record["validation_attempts"] = worker.get("validation_attempts")
        record["pipeline_seconds"] = worker.get("pipeline_seconds")
        record["run_dir"] = worker.get("run_dir")

    if status != "OK":
        record["status"], record["detail"] = status, detail
        return record

    # ---- feasibility gate: refuse hopeless validations instantly --------
    # Exact QPD equivalence validation enumerates every decomposition
    # variant; past ~4-5 cuts that is hours-to-days of work that will only
    # ever end in TIMEOUT_VALIDATOR or a memory crash. metadata.json is
    # written before validation, so read the overhead and classify.
    try:
        meta = json.loads(
            (Path(record["run_dir"]) / "metadata.json").read_text())
        record["sampling_overhead"] = meta.get("total_sampling_overhead")
    except Exception:
        pass
    # auto_finder nests its numbers one level down (metadata["metadata"])
    nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    if record["sampling_overhead"] is None:
        record["sampling_overhead"] = (meta.get("sampling_overhead")
                                       or nested.get("total_sampling_overhead")
                                       or nested.get("sampling_overhead"))
    ovh = record["sampling_overhead"]
    n_cuts = None
    try:
        _cut_list = meta.get("cuts") or nested.get("cuts")
        n_cuts = (len(meta.get("gate_ids") or [])
                  or len(meta.get("cut_locations") or [])
                  or (len(_cut_list) if isinstance(_cut_list, list) else None)
                  or meta.get("num_cuts") or nested.get("num_cuts") or None)
    except Exception:
        pass
    record["num_cuts"] = n_cuts
    # ---- in-graph equivalence verdict (preferred path) -------------------
    # validate_node already ran the strategy's equivalence validator on the
    # accepted attempt; consume its verdict instead of re-running the exact
    # same check offline (which would double the wall-clock per circuit).
    eq_passed = worker.get("equivalence_passed")
    eq_skipped = worker.get("equivalence_skipped")
    if eq_passed is True:
        record["status"] = "PASS"
        record["validator_seconds"] = worker.get("equivalence_seconds")
        record["detail"] = str(worker.get("equivalence_detail") or "")[-1500:]
        return record
    if eq_skipped == "no_cut":
        record["status"] = "PASS_NO_CUT"
        record["sampling_overhead"] = 1.0
        record["num_cuts"] = 0
        record["detail"] = ("no-cut run (see metadata no_cut_note): "
                            + str(worker.get("equivalence_detail") or ""))[:1500]
        return record
    if eq_skipped == "infeasible":
        record["status"] = "VALIDATION_INFEASIBLE"
        record["detail"] = str(worker.get("equivalence_detail") or "")[-1500:]
        return record

    # ---- legacy fallback: worker carried no in-graph verdict -------------
    # (e.g. records produced by an older batch_worker). Apply the ADAPTIVE
    # validation-cost gate -- which replaces the old fixed --max-cuts limit
    # -- and only then dispatch the offline validator.
    if ovh is not None and ovh > args.max_overhead:
        record["status"] = "VALIDATION_INFEASIBLE"
        record["detail"] = (
            f"sampling overhead {ovh:.4g} > --max-overhead "
            f"{args.max_overhead:g} (cuts={n_cuts}); exact equivalence "
            f"validation is computationally infeasible at this cut count")
        return record
    if n_cuts is not None:
        try:
            from validation_cost_model import estimated_validation_seconds
            est = estimated_validation_seconds(
                n_cuts, record["num_qubits"] or nq or 0, record["strategy"])
        except Exception:
            est = None
        if est is not None and est > budget:
            record["status"] = "VALIDATION_INFEASIBLE"
            record["detail"] = (
                f"estimated validation cost {est:.0f}s (cuts={n_cuts}, "
                f"qubits={record['num_qubits']}, model 6^cuts * "
                f"per_variant_cost fitted on batch_results_llm_v2) exceeds "
                f"the validation budget {budget:.0f}s")
            return record

    v_status, v_detail, v_secs = run_validator(
        record["strategy"], record["run_dir"], args.validator_timeout, env)
    record["status"] = v_status
    record["detail"] = v_detail
    record["validator_seconds"] = round(v_secs, 3)
    return record


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Resumable batch tester for the GRACE cutting pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--circuits", nargs="+", default=["benchmarks"],
                   help="Directories, files, or globs of .qasm circuits.")
    p.add_argument("--results-dir", default="batch_results",
                   help="Where results.jsonl, summary.csv and run exports go.")
    p.add_argument("--pipeline-timeout", type=int, default=600,
                   help="Seconds allowed per circuit for the LangGraph run.")
    p.add_argument("--validator-timeout", type=int, default=900,
                   help="Seconds allowed per circuit for equivalence check.")
    p.add_argument("--validation-budget", type=float, default=None,
                   help="Seconds the equivalence check is allowed to cost. "
                        "Replaces the old fixed --max-cuts limit with an "
                        "ADAPTIVE one: a run is VALIDATION_INFEASIBLE when "
                        "estimated_cost = 6^cuts * per_variant_cost(qubits, "
                        "observables) exceeds this budget (constants fitted "
                        "on batch_results_llm_v2 validator_seconds; see "
                        "validation_cost_model.py). Default: the value of "
                        "--validator-timeout. Also passed to the in-graph "
                        "Validate node via GRACE_EQUIV_BUDGET_SECONDS.")
    p.add_argument("--max-overhead", type=float, default=20000,
                   help="Skip equivalence validation when the run's total "
                        "sampling overhead exceeds this (recorded as "
                        "VALIDATION_INFEASIBLE). Exact QPD validation cost "
                        "grows with the variant count (~6^cuts for CX), so "
                        "past ~4-5 cuts it cannot finish in any reasonable "
                        "timeout. 6561 = 4 CX cuts; 59049 = 5.")
    p.add_argument("--shuffle", action="store_true",
                   help="Process circuits in a fixed pseudorandom order "
                        "instead of alphabetically, so early progress "
                        "samples all families representatively.")
    p.add_argument("--max-qubits", type=int, default=12,
                   help="Skip circuits above this size (exact statevector "
                        "reconstruction cost grows exponentially).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N not-yet-done circuits this session.")
    p.add_argument("--rerun", choices=["none", "failed", "errors", "all"],
                   default="none",
                   help="Which recorded circuits to redo.")
    p.add_argument("--no-llm", action="store_true",
                   help="Blank OPENROUTER_API_KEY in child processes so every "
                        "node takes its instant deterministic fallback "
                        "(useful offline; avoids slow network retries).")
    p.add_argument("--workers", type=int, default=1,
                   help="Circuits processed concurrently. Each circuit "
                        "already spawns its own subprocesses, so a good "
                        "value is the number of physical cores (validator "
                        "statevector work is CPU-bound).")
    p.add_argument("--smoke", action="store_true",
                   help="One-command smoke test: auto-generates a tiny MQT "
                        "set (ghz/graphstate/wstate at 3-4 qubits) into "
                        "benchmarks_smoke/ if missing, runs it into "
                        "batch_results_smoke/, and prints a verdict. "
                        "Use before every large run.")
    p.add_argument("--summarize", action="store_true",
                   help="Only print/write the aggregate report, run nothing.")
    args = p.parse_args(argv)

    if args.smoke:
        # Self-contained preflight check: tiny circuit set, separate results
        # dir so it never pollutes real benchmark data.
        args.circuits = ["benchmarks_smoke"]
        if args.results_dir == "batch_results":
            args.results_dir = "batch_results_smoke"
        args.max_qubits = min(args.max_qubits, 4)
        smoke_dir = Path("benchmarks_smoke")
        if not any(smoke_dir.glob("*.qasm")):
            print("[smoke] Generating smoke circuits into benchmarks_smoke/ ...")
            import fetch_mqt_circuits as fmc
            fmc.main(["--out", "benchmarks_smoke",
                      "--benchmarks", "ghz", "graphstate", "wstate",
                      "--min-qubits", "3", "--max-qubits", "4"])

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.jsonl"

    if args.summarize:
        summarize(results_file, results_dir / "summary.csv")
        return 0

    circuits = discover_circuits(args.circuits)
    if not circuits:
        print("No circuits found. Generate some first, e.g.:\n"
              "  python fetch_mqt_circuits.py --out benchmarks")
        return 2

    previous = load_previous(results_file)
    todo = [c for c in circuits
            if not should_skip(previous.get(c.name), args.rerun)]
    if args.shuffle:
        import random
        random.Random(2026).shuffle(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"Discovered {len(circuits)} circuit(s); "
          f"{len(previous)} already recorded; "
          f"{len(todo)} to process this session "
          f"(rerun policy: {args.rerun}).")
    if not todo:
        print("Nothing to do. Use --rerun or add circuits.")
        summarize(results_file)
        return 0

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # Keep the in-graph equivalence check (validate_node) on the same
    # feasibility budget / timeout the harness uses.
    env["GRACE_EQUIV_BUDGET_SECONDS"] = str(
        args.validation_budget or args.validator_timeout)
    if args.no_llm:
        # llm_client's lazy init raises immediately on an empty key, so all
        # nodes drop straight to their deterministic fallbacks. Setting the
        # variable (rather than deleting it) also stops load_dotenv() from
        # re-loading the real key out of .env in the child process.
        env["OPENROUTER_API_KEY"] = ""
        env["GRACE_NO_LLM"] = "1"

    # Graceful Ctrl+C: no partial records are ever written. Interrupted
    # circuits simply have no record yet, so the next invocation of the
    # same command picks them up again.
    interrupted = threading.Event()

    def _sigint(_sig, _frm):
        if interrupted.is_set():
            raise KeyboardInterrupt  # second Ctrl+C: hard exit
        interrupted.set()
        print("\n[batch] Interrupt received - no new circuits will start; "
              "in-flight circuits finish and are recorded. Press Ctrl+C "
              "again to abort immediately. Progress is saved; re-run the "
              "same command to resume.")

    signal.signal(signal.SIGINT, _sigint)

    session_counts: Counter = Counter()
    t_session = time.monotonic()
    write_lock = threading.Lock()
    done_n = 0

    def _task(qasm: Path) -> dict | None:
        if interrupted.is_set():
            return None
        return process_one(qasm, args, env, results_dir)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_task, q): q for q in todo}
        try:
            for fut in as_completed(futures):
                qasm = futures[fut]
                try:
                    record = fut.result()
                except Exception as exc:  # defensive: never lose the batch
                    record = {
                        "circuit": qasm.name, "qasm_path": str(qasm),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "PIPELINE_ERROR",
                        "detail": f"harness exception: {exc}",
                    }
                if record is None:
                    continue  # skipped due to interrupt: stays unrecorded
                with write_lock:
                    append_record(results_file, record)
                    session_counts[record["status"]] += 1
                    done_n += 1
                    print(f"[{done_n}/{len(todo)}] {record['circuit']:<28} "
                          f"-> {record['status']}"
                          f"  (strategy={record.get('strategy')}, "
                          f"pipe={record.get('pipeline_seconds')}s, "
                          f"val={record.get('validator_seconds')}s)")
        except KeyboardInterrupt:
            print("[batch] Hard abort - killing in-flight child processes.")
            for f in futures:
                f.cancel()
            _kill_active_children()

    elapsed = time.monotonic() - t_session
    print(f"\n{'=' * 64}\nSession finished in {elapsed:.1f}s: "
          f"{dict(session_counts)}")
    summarize(results_file, results_dir / "summary.csv")

    if args.smoke:
        recs = load_previous(results_file)
        bad = {k: r["status"] for k, r in recs.items()
               if r.get("status") in (FAIL_STATUSES | ERROR_STATUSES)}
        if bad:
            print(f"\n[smoke] VERDICT: NOT READY - investigate before a "
                  f"large run: {bad}")
            return 1
        print("\n[smoke] VERDICT: READY - pipeline, export, and validators "
              "all healthy. Safe to launch the full batch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
