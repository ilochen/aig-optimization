"""Offline collection → supervised training → surrogate-guided search."""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from .abc import ABC, actions, fingerprint
from .evaluator import Evaluator, reward
from .graph import Graph, FEATURE_VERSION
from .search import search
from .objective import Objective
from .bundle import write_bundle, verify_bundle
from .finalists import pool_from_trace, finalize, retain_artifacts, summary as finalist_summary


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def collect(args):
    backend = ABC(args.abc, args.library, timeout=getattr(args, "timeout", 60))
    rows = []
    rng = random.Random(args.seed)
    duplicates = 0
    partial = Path(args.output).with_suffix(".partial.json")

    def dataset(processed, complete):
        return {"feature_version": FEATURE_VERSION, "library_sha256": backend.library_sha256,
                "label_source": "ABC mapping with recorded timing profile", "mapping_profile": backend.mapping_profile,
                "abc_sha256": fingerprint(backend.executable), "seed": args.seed,
                "attempted_candidates": args.samples * processed,
                "duplicate_candidates": duplicates, "complete": complete, "rows": rows}
    with tempfile.TemporaryDirectory(prefix="aig-labels-") as directory:
        work = Path(directory)
        for index, source in enumerate(args.inputs):
            circuit = fingerprint(source)
            state = baseline = backend.prepare(source, work / f"circuit-{index}.aig")
            states = [(baseline, [])]
            seen = set()
            space = list(actions(args.seed + index))
            rng.shuffle(space)
            for sample in range(args.samples):
                if sample:
                    if sample <= len(space):
                        parent, sequence = states[0]
                        action = space[sample - 1]
                    else:
                        parent, sequence = rng.choice([s for s in states if len(s[1]) < getattr(args, "depth", 3)])
                        action = rng.choice(space)
                    state = backend.apply(parent, action, work / f"circuit-{index}-{sample}.aig")
                    sequence = [*sequence, action.command]
                    states.append((state, sequence))
                else:
                    sequence = []
                graph = Graph.read(state)
                digest = graph.digest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                backend.verify(baseline, state)
                mapped = backend.map(state)
                backend.verify(state, mapped["netlist"])
                rows.append({"circuit": circuit, "source_name": Path(source).name,
                             "sequence": list(sequence), "aig_sha256": fingerprint(state),
                             "graph_sha256": digest, "features": graph.features(),
                             "area": mapped["area"], "delay": mapped["delay"]})
            print(f"{Path(source).name}: {len(seen)} unique graphs / {args.samples} attempts", flush=True)
            write_json(partial, dataset(index + 1, False))
    write_json(args.output, dataset(len(args.inputs), True))
    partial.unlink(missing_ok=True)
    print(f"Collected {len(rows)} verified mapping examples → {args.output}")


def train(args):
    dataset = json.loads(Path(args.dataset).read_text())
    if dataset.get("complete") is False:
        raise ValueError("Label collection is incomplete; finish collection before training")
    if dataset["feature_version"] != FEATURE_VERSION:
        raise ValueError("Dataset feature schema mismatch")
    model = Evaluator.train(dataset["rows"], dataset["library_sha256"], args.epochs, args.seed,
                            scaled=not getattr(args, "absolute_targets", False))
    model.payload["mapping_profile"] = dataset.get("mapping_profile")
    model.payload["dataset_sha256"] = fingerprint(args.dataset)
    model.payload["abc_sha256"] = dataset.get("abc_sha256")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(json.dumps(model.payload["training_report"], indent=2))


def optimize(args):
    backend = ABC(args.abc, args.library, timeout=getattr(args, "timeout", 60))
    evaluator = Evaluator.load(args.model)
    if evaluator.payload["library_sha256"] != backend.library_sha256:
        raise ValueError("Checkpoint was trained for a different cell library; collect labels and retrain")
    backend.validate_profile(evaluator.payload)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    objective = Objective(getattr(args, 'objective', 'adp'), getattr(args, 'max_area', None), getattr(args, 'max_delay', None))
    finalist_limit=getattr(args,'finalists',1)
    proof_budget=getattr(args,'verification_budget',None)
    advanced=finalist_limit>1 or proof_budget is not None or getattr(args,'retain_finalists',False) or objective != Objective()
    # Keep all trial states temporary; export only the fully checked final result.
    with tempfile.TemporaryDirectory(prefix="aig-search-") as directory:
        work = Path(directory)
        initial = state = backend.prepare(args.input, work / "initial.aig")
        initial_features = Graph.read(initial).features()
        baseline = backend.map(initial)
        if proof_budget is None:
            backend.verify(initial, baseline["netlist"])
        else:
            backend.verify(initial, baseline["netlist"],timeout=proof_budget)
        initial_graph = Graph.read(initial)
        if hasattr(evaluator,'for_reference'):
            evaluator=evaluator.for_reference(initial_graph)
        base_prediction = evaluator.evaluate(initial, initial_graph) if hasattr(evaluator, "evaluate") else evaluator.predict(initial_features)
        decisions = [];pool=[];prefix=()
        for step in range(args.steps):
            decision_input=state
            stepdir = work / f"decision-{step}"
            stepdir.mkdir()
            calls_before = backend.mapping_calls
            chosen, trace = search(backend, state, evaluator, actions(args.seed + step),
                                   stepdir, args.iterations, args.depth, args.exploration,
                                   baseline=base_prediction, seed=args.seed + step,
                                   choice=getattr(args, "choice", "root"),
                                   record_frames=not getattr(args, "compact_trace", False),
                                   widening=getattr(args, "widening", None), objective=objective)
            if backend.mapping_calls != calls_before:
                raise RuntimeError("Mapping must not run inside the search loop")
            if advanced:
                local_pool=pool_from_trace(trace,decision_input,stepdir,prefix=prefix,decision=step)
                pool.extend(local_pool)
                primary=next(c for c in local_pool if c.key==f'{step}:{chosen.id}')
                prefix=(*prefix,*trace['plan'])
            state = chosen.state
            decisions.append(trace)
        finalization=None
        if advanced:
            finalization=finalize(backend,initial,baseline,initial_graph.digest(),pool,primary,
                limit=finalist_limit,verification_budget=proof_budget,baseline_verified=True,objective=objective)
            final_state,final_metrics=finalization['state'],finalization['metrics']
            primary_row=next(r for r in finalization['records'] if r['key']==primary.key)
            candidate=primary_row.get('mapped')
            accepted=finalization['key']!='baseline'
        else:
            backend.verify(initial, state)
            candidate = backend.map(state)
            backend.verify(initial, candidate["netlist"])
            accepted = reward(candidate, baseline) >= 0
            final_state, final_metrics = (state, candidate) if accepted else (initial, baseline)
            backend.verify(initial, final_state)
        report = {"input_sha256": fingerprint(args.input),
                  "model_sha256": fingerprint(args.model), "library_sha256": backend.library_sha256,
                  "seed": args.seed, "equivalence_proven": True, "mapping_profile": backend.mapping_profile,
                  "objective": objective.payload(), "objective_result": objective.describe(final_metrics),
                  "result_status": "constraints_unmet" if not objective.feasible(final_metrics) else "improved" if objective.better(final_metrics, baseline) else "baseline_retained",
                  "baseline_objective": objective.describe(baseline),
                  "objective_reward": objective.reward(final_metrics, baseline),
                  "candidate_accepted": accepted,
                  "fallback_reason": None if accepted else "No verified improvement within finalization budgets" if advanced else "Predicted candidate worsened real mapped area-delay product",
                  "baseline_mapped": {k: baseline[k] for k in ("area", "delay")},
                  "candidate_mapped": {k: candidate[k] for k in ("area", "delay")} if candidate is not None else None,
                  "final_mapped": {k: final_metrics[k] for k in ("area", "delay")},
                  "mapped_reward": reward(final_metrics, baseline),
                  "mapping_calls": backend.mapping_calls, "search_mapping_calls": 0,
                  "input_domain": evaluator.domain_graph(initial_graph) if hasattr(evaluator, "domain_graph") else evaluator.domain(initial_features) if hasattr(evaluator, "domain") else None,
                  "baseline_features": initial_features,
                  "final_features": Graph.read(final_state).features(), "decisions": decisions}
        # Build the complete export beside its destination. A failed write or
        # verification leaves no partially published result directory.
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".aig-export-", dir=output.parent) as staging:
            staged = Path(staging) / "result"
            staged.mkdir()
            shutil.copyfile(final_state, staged / "final.aig")
            shutil.copyfile(final_metrics["netlist"], staged / "final.mapped.blif")
            if finalization is not None:
                if getattr(args,'retain_finalists',False):
                    retain_artifacts(finalization,initial,args.library,staged/'finalists')
                report['finalization']=finalist_summary(finalization)
            write_json(staged / "trace.json", report)
            settings = {name: getattr(args, name, None) for name in (
                'seed', 'steps', 'iterations', 'depth', 'exploration', 'choice', 'widening',
                'timeout', 'finalists', 'verification_budget', 'compact_trace', 'retain_finalists')}
            settings['objective'] = objective.payload()
            write_bundle(staged, reference=initial, source=args.input, library=args.library,
                         model=args.model, backend=backend, settings=settings)
            if output.exists():
                raise FileExistsError(output)
            staged.rename(output)
        print(json.dumps({k: v for k, v in report.items() if k != "decisions"}, indent=2))


def verify(args):
    result = verify_bundle(args.result, args.abc, timeout=args.timeout,
                           allow_different_abc=args.allow_different_abc)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "optimize"):
        command = commands.add_parser(name)
        command.add_argument("--abc", required=True, help="Path to a built Berkeley ABC executable")
        command.add_argument("--library", required=True, help="ABC GENLIB or Liberty (.lib) cell library")
        command.add_argument("--seed", type=int, default=314159)
        command.add_argument("--timeout", type=float, default=60, help="Seconds per ABC invocation")
        command.add_argument("--output", required=True)
        if name == "collect":
            command.add_argument("inputs", nargs="+")
            command.add_argument("--samples", type=int, default=12)
            command.add_argument("--depth", type=int, default=3)
            command.set_defaults(func=collect)
        else:
            command.add_argument('--objective', choices=('adp', 'area', 'delay'), default='adp')
            command.add_argument('--max-area', type=float, help='Maximum measured mapped area in library units')
            command.add_argument('--max-delay', type=float, help='Maximum measured mapped delay in library units')
            command.add_argument("--input", required=True)
            command.add_argument("--model", required=True)
            command.add_argument("--iterations", type=int, default=16)
            command.add_argument("--depth", type=int, default=3)
            command.add_argument("--steps", type=int, default=1)
            command.add_argument("--exploration", type=float, default=.8)
            command.add_argument("--choice", choices=("root", "best"), default="root")
            command.add_argument("--compact-trace", action="store_true", help="Omit quadratic per-phase tree snapshots")
            command.add_argument("--widening", type=float, help="Optional coefficient for sqrt-visit progressive widening")
            command.add_argument('--finalists',type=int,default=1,help='Final mapping budget: primary and predicted area/delay extremes')
            command.add_argument('--verification-budget',type=float,help='Total proof seconds, separate from the ABC transformation timeout')
            command.add_argument('--retain-finalists',action='store_true',help='Save diagnostic finalists with explicit verification status')
            command.set_defaults(func=optimize)
    command = commands.add_parser('verify', help='Independently check a portable result bundle')
    command.add_argument('--result', required=True)
    command.add_argument('--abc', required=True)
    command.add_argument('--timeout', type=float, default=60)
    command.add_argument('--allow-different-abc', action='store_true')
    command.set_defaults(func=verify)
    command = commands.add_parser("train")
    command.add_argument("--dataset", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--epochs", type=int, default=1500)
    command.add_argument("--seed", type=int, default=314159)
    command.add_argument("--absolute-targets", action="store_true", help="Ablation: train absolute log costs rather than scaled residuals")
    command.set_defaults(func=train)
    args = parser.parse_args()
    if any(getattr(args, name, 1) < 1 for name in ("samples", "steps", "iterations", "depth", "epochs", "finalists")):
        parser.error("Sample counts, steps and budgets must be positive")
    if not math.isfinite(getattr(args, "timeout", 1)) or getattr(args, "timeout", 1) <= 0:
        parser.error("Timeout must be positive")
    proof_budget=getattr(args,'verification_budget',None)
    if proof_budget is not None and (not math.isfinite(proof_budget) or proof_budget<=0):
        parser.error('Verification budget must be finite and positive')
    if args.command == 'optimize':
        try:
            Objective(args.objective, args.max_area, args.max_delay)
        except ValueError as error:
            parser.error(str(error))
    args.func(args)


if __name__ == "__main__":
    main()
