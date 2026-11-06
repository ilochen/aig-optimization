"""Evaluate surrogate errors and within-circuit candidate rankings on held-out graphs."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
from .abc import ABC, ABCError, actions, fingerprint
from .benchmark import assert_disjoint
from .evaluator import Evaluator
from .graph import Graph


def ranking_metrics(rows):
    score, comparable, ties = 0., 0, 0
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            actual = left["area"] * left["delay"] / (right["area"] * right["delay"]) - 1
            if abs(actual) < 1e-4:
                continue  # Rounded mapping ties contain no ranking signal.
            predicted = left["prediction"]["area"] * left["prediction"]["delay"] / (
                right["prediction"]["area"] * right["prediction"]["delay"]) - 1
            comparable += 1
            if abs(predicted) < 1e-6:
                score += .5
                ties += 1
            elif (actual > 0) == (predicted > 0):
                score += 1
    return {"examples": len(rows), "comparable_pairs": comparable,
            "predicted_tied_pairs": ties, "pair_accuracy": score / comparable if comparable else None,
            "area_mape": statistics.mean(abs(r["prediction"]["area"] / r["area"] - 1) for r in rows),
            "delay_mape": statistics.mean(abs(r["prediction"]["delay"] / r["delay"] - 1) for r in rows),
            "adp_mape": statistics.mean(abs(r["prediction"]["area"] * r["prediction"]["delay"] / (r["area"] * r["delay"]) - 1) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint JSON files")
    parser.add_argument("--legacy-dataset", help="Original hash-matching dataset for older checkpoints")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.timeout <= 0:
        parser.error("Samples must be positive")
    cases = [Path(args.benchmark_root) / p for p in json.loads(Path(args.manifest).read_text())["test"]]
    models = {Path(p).stem: Evaluator.load(p) for p in args.models}
    if len(models) != len(args.models):
        raise ValueError("Model filenames must have unique stems")
    for model in models.values():
        if model.payload["library_sha256"] != fingerprint(args.library):
            raise ValueError("Checkpoint/library mismatch")
        assert_disjoint(model, cases, args.legacy_dataset)
    rows, per_circuit, failures = [], {}, []
    for index, source in enumerate(cases):
        backend = ABC(args.abc, args.library, timeout=args.timeout)
        rng = random.Random(args.seed + index)
        space = actions(args.seed + index)
        seen, candidates = set(), []
        with tempfile.TemporaryDirectory(prefix="aig-probes-") as directory:
            work = Path(directory)
            initial = backend.prepare(source, work / "initial.aig")
            circuit_models={name:model.for_reference(Graph.read(initial)) if hasattr(model,'for_reference') else model for name,model in models.items()}
            states = [(initial, 0, [])]
            for sample in range(args.samples):
                if sample == 0:
                    state, depth, sequence = initial, 0, []
                else:
                    parent, parent_depth, parent_sequence = states[0] if sample <= len(space) else rng.choice([s for s in states if s[1] < 3])
                    action = space[sample - 1] if sample <= len(space) else rng.choice(space)
                    try:
                        state = backend.apply(parent, action, work / f"probe-{sample}.aig")
                    except (ABCError, subprocess.TimeoutExpired) as error:
                        failures.append({"circuit": source.stem, "sample": sample,
                            "command": action.command, "error": f"{type(error).__name__}: {error}"})
                        continue
                    depth = parent_depth + 1
                    sequence = [*parent_sequence, action.command]
                    states.append((state, depth, sequence))
                graph = Graph.read(state)
                digest = graph.digest()
                if digest in seen:
                    continue
                seen.add(digest)
                try:
                    mapped = backend.map(state)
                    backend.verify(initial, state)
                    backend.verify(initial, mapped["netlist"])
                except (ABCError, subprocess.TimeoutExpired) as error:
                    failures.append({"circuit": source.stem, "sample": sample,
                        "error": f"{type(error).__name__}: {error}"})
                    continue
                features = graph.features()
                row = {"circuit": source.stem, "input_sha256": fingerprint(source),
                       "graph_sha256": digest, "features": features,
                       "sequence": sequence,
                       "evaluator_features": {name: m.encode(graph) if hasattr(m, "encode") else features for name, m in circuit_models.items()},
                       "area": mapped["area"], "delay": mapped["delay"],
                       "predictions": {name: m.evaluate(state, graph) if hasattr(m, "evaluate") else m.predict(features) for name, m in circuit_models.items()},
                       "domains": {name: m.domain_graph(graph) if hasattr(m, "domain_graph") else m.domain(features) for name, m in circuit_models.items()}}
                rows.append(row)
                candidates.append(row)
        per_circuit[source.stem] = {name: ranking_metrics([
            {**row, "prediction": row["predictions"][name]} for row in candidates]) for name in models}
        print(f"{source.stem}: {len(candidates)} unique verified probes", flush=True)
    summary = {}
    for name in models:
        metrics = [c[name] for c in per_circuit.values()]
        summary[name] = {}
        for key in ("area_mape", "delay_mape", "adp_mape", "pair_accuracy"):
            values = [m[key] for m in metrics if m[key] is not None]
            summary[name][key] = statistics.mean(values) if values else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model_hashes": {Path(p).stem: fingerprint(p) for p in args.models},
        "library_sha256": fingerprint(args.library), "abc_sha256": fingerprint(args.abc),
        "manifest_sha256": fingerprint(args.manifest), "samples_per_circuit": args.samples,
        "per_command_timeout_seconds": args.timeout, "failed_actions": failures,
        "source_sha256": {p.name: fingerprint(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "seed": args.seed, "methodology": "Exact graph deduplication; within-circuit ADP comparisons; true relative ties <1e-4 excluded, predicted ties <1e-6 earn half credit; summary gives circuits equal weight.",
        "summary": summary, "per_circuit": per_circuit, "rows": rows}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
