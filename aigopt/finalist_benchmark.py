"""Matched mapping budgets for post-search finalist selection and certification."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter
from .abc import ABC, actions, fingerprint
from .benchmark import StructuralEvaluator, assert_disjoint
from .evaluator import Evaluator, reward
from .finalists import pool_from_trace, finalize, retain_artifacts, summary
from .graph import Graph
from .search import search

METHODS={'structural-1':('structural',1),'neural-1':('neural',1),
    'structural-3':('structural',3),'neural-3':('neural',3),'ranker-3':('ranker',3)}


def run_case(args,path,method,seed,model,ranker,artifact_dir=None):
    backend=ABC(args.abc,args.library,timeout=args.timeout)
    started=perf_counter();name,limit=METHODS[method]
    evaluator=StructuralEvaluator() if name=='structural' else model if name=='neural' else ranker
    with tempfile.TemporaryDirectory(prefix='finalist-benchmark-') as directory:
        work=Path(directory)
        initial=backend.prepare(path,work/'input.aig');graph=Graph.read(initial)
        baseline=backend.map(initial)
        backend.verify(initial,baseline['netlist'],timeout=args.verification_budget)
        search_started=perf_counter();before_maps=backend.mapping_calls
        chosen,trace=search(backend,initial,evaluator,actions(seed),work,args.budget,args.depth,
            seed=seed,choice='best',widening=1.,record_frames=False)
        search_seconds=perf_counter()-search_started
        if backend.mapping_calls!=before_maps:raise RuntimeError('Surrogate search invoked mapping')
        pool=pool_from_trace(trace,initial,work)
        primary=next(c for c in pool if c.key==f'0:{chosen.id}')
        result=finalize(backend,initial,baseline,graph.digest(),pool,primary,limit=limit,
            verification_budget=args.verification_budget,baseline_verified=True)
        if artifact_dir is not None:
            retain_artifacts(result,initial,args.library,artifact_dir/'finalists')
        ledger=[{k:r[k] for k in ('id','parent','command','graph_sha256')} |
            {'error_type':r['error'].split(':')[0] if r['error'] else None} for r in trace['candidates']]
        digest=hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest()
        primary_row=next(r for r in result['records'] if r['key']==primary.key)
        verified=result['metrics']
        return {'circuit':path.stem,'input_sha256':fingerprint(path),'method':method,'seed':seed,
            'accepted_reward':reward(verified,baseline),'equivalence_proven':True,
            'baseline':{k:baseline[k] for k in ('area','delay')},
            'final':{k:verified[k] for k in ('area','delay')},
            'primary_measured':primary_row.get('mapped'),'baseline_retained':result['key']=='baseline',
            'search_ledger_sha256':digest,'trace':trace,'finalization':summary(result),
            'search_seconds':search_seconds,'total_seconds':perf_counter()-started,
            'mapping_calls':backend.mapping_calls,'search_mapping_calls':0,
            'verification_calls':backend.verification_calls,'verification_seconds':backend.verification_seconds,
            'transformations':backend.transform_calls,'eda_commands':backend.eda_commands}


def aggregate(rows):
    import statistics
    grouped=defaultdict(list)
    for row in rows:grouped[row['method']].append(row)
    out={}
    for method,records in grouped.items():
        cases=defaultdict(list)
        for row in records:cases[row['circuit']].append(row)
        valid=[r for r in records if 'error' not in r]
        out[method]={'runs':len(records),'failures':len(records)-len(valid),
            'mean_accepted_reward':statistics.mean(statistics.mean(r.get('accepted_reward',0.) for r in case) for case in cases.values()),
            'mean_mapping_calls':statistics.mean(r['mapping_calls'] for r in valid) if valid else None,
            'median_total_seconds':statistics.median(r['total_seconds'] for r in records),
            'mean_proof_seconds':statistics.mean(r['verification_seconds'] for r in valid) if valid else None,
            'unproven_finalists':sum(sum(c['status']=='unproven' for c in r['finalization']['records']) for r in valid),
            'baseline_retained':sum(r['baseline_retained'] for r in valid),
            'per_circuit':{name:statistics.mean(r.get('accepted_reward',0.) for r in case) for name,case in cases.items()}}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--abc',required=True);p.add_argument('--library',required=True)
    p.add_argument('--model',required=True);p.add_argument('--ranker',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--benchmark-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',default='test')
    p.add_argument('--methods',choices=list(METHODS),nargs='+',default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37])
    p.add_argument('--budget',type=int,default=24);p.add_argument('--depth',type=int,default=3)
    p.add_argument('--timeout',type=float,default=5);p.add_argument('--verification-budget',type=float,default=30)
    p.add_argument('--retain',action='store_true')
    args=p.parse_args()
    if min(args.budget,args.depth,args.timeout,args.verification_budget)<=0:p.error('Budgets must be positive')
    suite=json.loads(Path(args.manifest).read_text())
    paths=[Path(args.benchmark_root)/name for name in suite[args.split]]
    model=Evaluator.load(args.model);ranker=Evaluator.load(args.ranker)
    if args.split=='test':
        for candidate in (model,ranker):assert_disjoint(candidate,paths)
    for candidate in (model,ranker):
        if candidate.payload['library_sha256']!=fingerprint(args.library):raise ValueError('Cell-library mismatch')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config={'abc_sha256':fingerprint(args.abc),'library_sha256':fingerprint(args.library),
        'model_sha256':fingerprint(args.model),'ranker_sha256':fingerprint(args.ranker),
        'manifest_sha256':fingerprint(args.manifest),'split':args.split,
        'methods':args.methods,'seeds':args.seeds,'budget':args.budget,'depth':args.depth,
        'timeout':args.timeout,'verification_budget':args.verification_budget,
        'input_sha256':{path.name:fingerprint(path) for path in paths},
        'source_sha256':{path.name:fingerprint(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
        'selection':'primary first; predicted area and delay extremes; unique-graph dedup; remaining predicted ADP',
        'verification':'baseline proved; primary certified first; only better measured finalists can replace a verified incumbent'}
    configpath=out/'config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=config:raise ValueError('Benchmark configuration changed')
    configpath.write_text(json.dumps(config,indent=2)+'\n')
    raw=out/'runs.jsonl';rows=[json.loads(x) for x in raw.read_text().splitlines()] if raw.exists() else []
    complete={(r['circuit'],r['method'],r['seed']) for r in rows}
    jobs=[(path,method,seed) for path in paths for method in args.methods for seed in args.seeds]
    random.Random(971).shuffle(jobs)
    for path,method,seed in jobs:
        if (path.stem,method,seed) in complete:continue
        started=perf_counter()
        try:
            artifact_dir=out/'artifacts'/f'{path.stem}-{method}-{seed}' if args.retain else None
            row=run_case(args,path,method,seed,model,ranker,artifact_dir)
            print(f"{path.stem} {method} seed={seed} reward={row['accepted_reward']:.2%} maps={row['mapping_calls']} proof={row['verification_seconds']:.2f}s",flush=True)
        except Exception as error:
            row={'circuit':path.stem,'method':method,'seed':seed,'total_seconds':perf_counter()-started,
                'error':f'{type(error).__name__}: {error}'}
            print(f'FAILED {path.stem} {method} seed={seed}: {error}',flush=True)
        with raw.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);complete.add((path.stem,method,seed))
        (out/'summary.json').write_text(json.dumps(aggregate(rows),indent=2)+'\n')


if __name__=='__main__':main()
