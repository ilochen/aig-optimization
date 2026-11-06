"""Frozen test on nine untouched ISCAS85 designs; no fitting in this runner."""
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aigopt.abc import ABC,ABCError,actions,fingerprint
from aigopt.benchmark import StructuralEvaluator,assert_disjoint
from aigopt.evaluator import Evaluator
from aigopt.finalist_benchmark import run_case,aggregate
from aigopt.graph import Graph
from aigopt.load_descriptors import describe
from aigopt.probes import ranking_metrics


def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--abc',required=True)
    p.add_argument('--selection',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--inputs',default=str(ROOT/'build/iscas-inputs'));p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    suite_path=ROOT/'benchmarks/suite.json';suite=json.loads(suite_path.read_text());protocol=suite['protocol']
    paths=[Path(args.inputs)/row['path'] for row in suite['test']]
    for path,record in zip(paths,suite['test']):
        if fingerprint(path)!=record['sha256']:raise ValueError('Reserved input changed')
    library=ROOT/protocol['library'];backend=ABC(args.abc,library)
    if backend.library_sha256!=protocol['library_sha256']:raise ValueError('Library mismatch')
    selection=Path(args.selection);data=json.loads(Path(args.dataset).read_text())
    model_paths={'previous-neural':ROOT/'benchmarks/checkpoints/baseline.json',
        'data-only-neural':selection/'data-only.json','selected-ranker':selection/'selected.json'}
    models={name:Evaluator.load(path) for name,path in model_paths.items()}
    for model in models.values():
        assert_disjoint(model,paths);backend.validate_profile(model.payload)
    out=Path(args.output)
    if out.exists() and not args.resume:raise FileExistsError('Use a new test output directory')
    out.mkdir(parents=True,exist_ok=True)
    config={'suite_sha256':fingerprint(suite_path),'selection_sha256':fingerprint(selection/'selection.json'),
        'model_sha256':{name:fingerprint(path) for name,path in model_paths.items()},'dataset_sha256':fingerprint(args.dataset),
        'library_sha256':backend.library_sha256,'mapping_profile':backend.mapping_profile,'abc_sha256':fingerprint(args.abc),
        'source_sha256':{path.name:fingerprint(path) for path in sorted((ROOT/'aigopt').glob('*.py'))},
        'runner_sha256':fingerprint(__file__),'protocol':protocol,'test_outcomes_used_for_selection':False}
    freeze=out/'test-freeze.json'
    if freeze.exists() and json.loads(freeze.read_text())!=config:raise ValueError('Frozen test changed')
    write(freeze,config);shutil.copyfile(suite_path,out/'suite.json')
    source=out/'source';source.mkdir(exist_ok=True)
    for path in (ROOT/'aigopt').glob('*.py'):shutil.copyfile(path,source/path.name)
    shutil.copyfile(__file__,source/'run_iscas_study.py')
    # Preflight normalizes logic without mapping it or evaluating any model.
    graphs={}
    with tempfile.TemporaryDirectory(prefix='iscas-preflight-') as directory:
        for i,path in enumerate(paths):
            state=backend.prepare(path,Path(directory)/f'initial-{i}.aig');graphs[path.stem]=Graph.read(state).digest()
    if set(graphs.values()) & {r['graph_sha256'] for r in data['rows']}:
        raise ValueError('Reserved graph overlaps development labels')
    write(out/'graph-preflight.json',{'source_graphs':graphs,'development_overlap':False})
    benchmark=SimpleNamespace(abc=args.abc,library=library,timeout=10,verification_budget=30,budget=24,depth=3)
    raw=out/'runs.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    complete={(r['circuit'],r['method'],r['seed']) for r in rows}
    jobs=[(path,method,seed) for path in paths for method in protocol['test_methods'] for seed in protocol['search_seeds']]
    random.Random(971).shuffle(jobs)
    for path,method,seed in jobs:
        if (path.stem,method,seed) in complete:continue
        started=perf_counter()
        try:
            model=models.get(method,models['previous-neural'])
            row=run_case(benchmark,path,'structural-3' if method=='structural' else 'neural-3',seed,model,model)
            row['method']=method;row['model_sha256']=config['model_sha256'].get(method)
            print(f'{len(rows)+1}/{len(jobs)} {path.stem} {method} {seed}: {row["accepted_reward"]:.2%}',flush=True)
        except (RuntimeError,ValueError,subprocess.TimeoutExpired) as error:
            row={'circuit':path.stem,'method':method,'seed':seed,'input_sha256':fingerprint(path),
                'error':f'{type(error).__name__}: {error}','accepted_reward':0.,'total_seconds':perf_counter()-started}
            print(f'FAILED {path.stem} {method} {seed}: {error}',flush=True)
        with raw.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);write(out/'summary.json',aggregate(rows))
    probe_path=out/'probes.json'
    if probe_path.exists() and json.loads(probe_path.read_text()).get('complete'):return
    probe_rows=[];failures=[];per_circuit={}
    for index,path in enumerate(paths):
        b=ABC(args.abc,library,timeout=10);seen=set();local=[]
        with tempfile.TemporaryDirectory(prefix='iscas-probes-') as directory:
            work=Path(directory);initial=b.prepare(path,work/'initial.aig')
            circuit_models={name:model.for_reference(Graph.read(initial)) if hasattr(model,'for_reference') else model for name,model in models.items()}
            for sample,action in enumerate((None,*actions(83+index))):
                try:
                    state=initial if action is None else b.apply(initial,action,work/f'probe-{sample}.aig')
                    graph=Graph.read(state);digest=graph.digest()
                    if digest in seen:continue
                    b.verify(initial,state);mapped=b.map(state);b.verify(initial,mapped['netlist'])
                    seen.add(digest);features=graph.features()
                    predictions={'structural':StructuralEvaluator().predict(features)}
                    for name,model in circuit_models.items():
                        predictions[name]=model.evaluate(state,graph) if hasattr(model,'evaluate') else model.predict(features)
                    local.append({'circuit':path.stem,'input_sha256':fingerprint(path),'graph_sha256':digest,
                        'features':features,'rank_features':describe(graph),'area':mapped['area'],'delay':mapped['delay'],
                        'predictions':predictions})
                except (ABCError,subprocess.TimeoutExpired) as error:
                    failures.append({'circuit':path.stem,'sample':sample,'error':f'{type(error).__name__}: {error}'})
        per_circuit[path.stem]={method:ranking_metrics([{**row,'prediction':row['predictions'][method]} for row in local]) for method in protocol['test_methods']} if local else {}
        probe_rows.extend(local)
        write(probe_path,{'complete':False,'rows':probe_rows,'per_circuit':per_circuit,'failures':failures,
            'model_sha256':config['model_sha256'],'test_freeze_sha256':fingerprint(freeze)})
        print(f'Probes {path.stem}: {len(local)} verified graphs',flush=True)
    probes=json.loads(probe_path.read_text());probes['complete']=True;write(probe_path,probes)
    print('Completed frozen ISCAS85 evaluation.',flush=True)


if __name__=='__main__':main()
