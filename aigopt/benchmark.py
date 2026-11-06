"""Local, circuit-held-out comparisons with raw outcomes and real tool costs."""
import argparse
from collections import defaultdict
import json
import math
import platform
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
from time import perf_counter
from .abc import ABC, ABCError, Action, actions, fingerprint
from .evaluator import Evaluator, reward
from .graph import Graph
from .search import search

RESYN2 = ("balance", "rewrite", "refactor", "balance", "rewrite", "rewrite -z",
          "balance", "refactor -z", "rewrite -z", "balance")
PRIMARY_METHODS = ("resyn2", "random-mapped", "structural", "legacy-neural", "neural-root", "neural-best", "neural-wide", "mapped-uct")
METHODS = (*PRIMARY_METHODS, "structural-wide", "previous-wide")


class StructuralEvaluator:
    def predict(self, features):
        return {"area": max(1, features[0]), "delay": max(1, features[1])}


class MappedEvaluator:
    def __init__(self, backend, initial, metrics):
        self.backend = backend
        self.cache = {Graph.read(initial).digest(): metrics}

    def evaluate(self, state, graph):
        key = graph.digest()
        if key not in self.cache:
            self.cache[key] = self.backend.map(state)
        return {k: self.cache[key][k] for k in ("area", "delay")}


def assert_disjoint(model, cases, dataset=None):
    report = model.payload.get("training_report", {})
    seen = set(report.get("train_circuits", [])) | set(report.get("validation_circuits", []))
    if "train_circuits" not in report:
        if dataset is None or fingerprint(dataset) != model.payload.get("dataset_sha256"):
            raise ValueError("Older checkpoints need their hash-matching dataset to audit circuit overlap")
        seen = {r["circuit"] for r in json.loads(Path(dataset).read_text())["rows"]}
    overlap = seen & {fingerprint(p) for p in cases}
    if overlap:
        raise ValueError("Benchmark circuits overlap checkpoint training/validation circuits")


def random_search(backend, initial, evaluator, space, directory, budget, depth, seed):
    rng = random.Random(seed)
    states = [(initial, 0, [])]
    base_graph = Graph.read(initial)
    baseline = evaluator.evaluate(initial, base_graph)
    best, best_metrics, best_plan = initial, baseline, []
    failures = []
    for i in range(budget):
        parent, parent_depth, plan = rng.choice([s for s in states if s[1] < depth])
        action = rng.choice(space)
        plan = [*plan, action.command]
        try:
            state = backend.apply(parent, action, directory / f"random-{i}.aig")
            metrics = evaluator.evaluate(state, Graph.read(state))
        except (ABCError, subprocess.TimeoutExpired) as error:
            failures.append({"command": action.command, "timeout": isinstance(error, subprocess.TimeoutExpired),
                             "error": f"{type(error).__name__}: {str(error)[-1500:]}"})
            continue
        states.append((state, parent_depth + 1, plan))
        if reward(metrics, best_metrics) > 0:
            best, best_metrics, best_plan = state, metrics, plan
    return best, {"plan": best_plan, "visited_max_depth": max(s[1] for s in states), "failed_actions": failures}


def run_case(abc_path, library, path, method, model, legacy, budget, depth, seed, timeout=5):
    backend = ABC(abc_path, library, timeout=timeout)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="aig-benchmark-") as directory:
        work = Path(directory)
        initial = backend.prepare(path, work / "input.aig")
        before_features = Graph.read(initial).features()
        baseline = backend.map(initial)
        backend.verify(initial, baseline["netlist"])
        setup_seconds = perf_counter() - started
        search_started = perf_counter()
        mapping_before = backend.mapping_calls
        space = actions(seed)
        if method == "resyn2":
            state = initial
            plan = []
            for i in range(budget):
                command = RESYN2[i % len(RESYN2)]
                state = backend.apply(state, Action(command, command), work / f"pass-{i}.aig")
                plan.append(command)
            trace = {"plan": plan, "visited_max_depth": budget}
        elif method == "random-mapped":
            evaluator = MappedEvaluator(backend, initial, baseline)
            state, trace = random_search(backend, initial, evaluator,
                                         space, work, budget, depth, seed)
        else:
            evaluator = (MappedEvaluator(backend, initial, baseline) if method == "mapped-uct" else
                         StructuralEvaluator() if method in ("structural", "structural-wide") else
                         legacy if method in ("legacy-neural", "previous-wide") else model)
            chosen, trace = search(backend, initial, evaluator, space, work, budget, depth,
                                   seed=seed, randomize=method != "legacy-neural",
                                   choice="root" if method in ("legacy-neural", "neural-root") else "best",
                                   record_frames=False, widening=1. if method in ("neural-wide", "structural-wide", "previous-wide") else None)
            state = chosen.state
        search_seconds = perf_counter() - search_started
        search_mapping_calls = backend.mapping_calls - mapping_before
        # Cache direct-mapped candidates so reporting does not add artificial
        # mapping cost. Final verification is identical for every method.
        if method in ("mapped-uct", "random-mapped"):
            final_graph = Graph.read(state)
            candidate = evaluator.cache[final_graph.digest()]
        else:
            candidate = None
        if candidate is None:
            candidate = backend.map(state)
        backend.verify(initial, state)
        backend.verify(initial, candidate["netlist"])
        raw_reward = reward(candidate, baseline)
        accepted = raw_reward >= 0
        return {"circuit": Path(path).stem, "input_sha256": fingerprint(path),
                "method": method, "seed": seed, "candidate_budget": budget,
                "depth_limit": depth if method != "resyn2" else None,
                "baseline": {k: baseline[k] for k in ("area", "delay")},
                "candidate": {k: candidate[k] for k in ("area", "delay")},
                "raw_reward": raw_reward, "accepted_reward": max(0, raw_reward),
                "fallback": not accepted, "equivalence_proven": True,
                "before_features": before_features, "candidate_features": Graph.read(state).features(),
                "setup_seconds": setup_seconds, "search_seconds": search_seconds,
                "total_seconds": perf_counter() - started,
                "transformations": backend.transform_calls, "mapping_calls": backend.mapping_calls,
                "eda_commands": backend.eda_commands,
                "search_mapping_calls": search_mapping_calls, "verification_calls": backend.verification_calls,
                "abc_process_seconds": backend.process_seconds, "trace": trace}


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    output = {}
    for method, records in grouped.items():
        # Equal circuit weight, even if a method has a missing/failed seed.
        circuits = defaultdict(list)
        for row in records:
            if "error" in row:
                row = dict(row, accepted_reward=0., fallback=True)
            circuits[row["circuit"]].append(row)
        successful = [r for r in records if "error" not in r]
        raw_circuits = defaultdict(list)
        for row in successful:
            raw_circuits[row["circuit"]].append(row)
        output[method] = {"successful_runs": len(successful), "attempted_runs": len(records), "circuits": len(circuits),
            "mean_accepted_reward": statistics.mean(statistics.mean(r["accepted_reward"] for r in group) for group in circuits.values()),
            "mean_raw_reward": statistics.mean(statistics.mean(r["raw_reward"] for r in group) for group in raw_circuits.values()) if raw_circuits else None,
            "geomean_accepted_adp_ratio": math.exp(statistics.mean(statistics.mean(
                math.log(1 - r["accepted_reward"]) for r in group) for group in circuits.values())),
            "fallback_rate": statistics.mean(r.get("fallback", True) for r in records),
            "failure_rate": 1 - len(successful) / len(records),
            "median_search_seconds": statistics.median(r["search_seconds"] for r in successful) if successful else None,
            "median_total_seconds": statistics.median(r["total_seconds"] for r in records),
            "mean_transformations": statistics.mean(r["transformations"] for r in successful) if successful else None,
            "mean_eda_commands": statistics.mean(r["eda_commands"] for r in successful) if successful else None,
            "mean_mapping_calls": statistics.mean(r["mapping_calls"] for r in successful) if successful else None,
            "rejected_actions": sum(len(r.get("trace", {}).get("failed_actions", [])) for r in successful),
            "per_circuit": {name: {"mean_reward": statistics.mean(r["accepted_reward"] for r in group),
                "min_reward": min(r["accepted_reward"] for r in group),
                "max_reward": max(r["accepted_reward"] for r in group)} for name, group in circuits.items()}}
    return {"methods": output, "failures": [r for r in rows if "error" in r]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--legacy-model")
    parser.add_argument("--legacy-dataset")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5, help="Seconds per ABC invocation, including final checks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(PRIMARY_METHODS))
    args = parser.parse_args()
    if args.budget < 1 or args.depth < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Budget and depth must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("Seeds and methods must be unique")
    root = Path(args.benchmark_root)
    manifest = json.loads(Path(args.manifest).read_text())
    cases = [root / p for p in manifest["test"]]
    if not cases or len({p.stem for p in cases}) != len(cases):
        raise ValueError("Test suite must have unique, nonempty circuit names")
    model = Evaluator.load(args.model)
    legacy = Evaluator.load(args.legacy_model) if args.legacy_model else None
    if set(args.methods) & {"legacy-neural", "previous-wide"} and legacy is None:
        parser.error("The requested previous/legacy policy requires --legacy-model")
    library_hash = fingerprint(args.library)
    for evaluator in (model, legacy):
        if evaluator:
            if evaluator.payload["library_sha256"] != library_hash:
                raise ValueError("Checkpoint/library mismatch")
            assert_disjoint(evaluator, cases, args.legacy_dataset if evaluator is legacy else None)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = {"abc_sha256": fingerprint(args.abc), "library_sha256": library_hash,
              "python_version": platform.python_version(), "machine": platform.machine(),
              "source_sha256": {p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "model_sha256": fingerprint(args.model), "legacy_model_sha256": fingerprint(args.legacy_model) if legacy else None,
              "legacy_dataset_sha256": fingerprint(args.legacy_dataset) if args.legacy_dataset else None,
              "manifest_sha256": fingerprint(args.manifest), "budget": args.budget,
              "depth": args.depth, "seeds": args.seeds, "methods": args.methods,
              "per_command_timeout_seconds": args.timeout,
              "input_sha256": {p.name: fingerprint(p) for p in cases},
              "methodology": "Equal candidate action budgets; macros contain multiple EDA commands. Resyn2 has unrestricted sequence depth. Timings are sequential local wall times, including subprocess overhead. Training cost is excluded. Raw and baseline-fallback outcomes are both recorded."}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different benchmark configuration")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    raw = output / "runs.jsonl"
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    completed = {(r["circuit"], r["method"], r["seed"]) for r in rows}
    jobs = [(path, method, seed) for path in cases for seed in args.seeds for method in args.methods]
    # Fixed shuffle reduces warm-cache/order bias, without simultaneous timing.
    random.Random(731).shuffle(jobs)
    for path, method, seed in jobs:
        if (path.stem, method, seed) in completed:
            continue
        try:
            started = perf_counter()
            row = run_case(args.abc, args.library, path, method, model, legacy, args.budget, args.depth, seed, args.timeout)
            print(f"{path.stem:6} {method:14} seed={seed} improvement={row['accepted_reward']:.2%} raw={row['raw_reward']:.2%} time={row['total_seconds']:.2f}s", flush=True)
        except Exception as error:
            row = {"circuit": path.stem, "method": method, "seed": seed,
                   "total_seconds": perf_counter() - started,
                   "error": f"{type(error).__name__}: {error}"}
            print(f"FAILED {path.stem} {method} seed={seed}: {error}", flush=True)
        with raw.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row)
        completed.add((path.stem, method, seed))
        (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
