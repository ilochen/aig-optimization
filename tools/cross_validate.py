"""Select relative-cost training/features on previously used controls only."""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aigopt.abc import fingerprint
from aigopt.centered_training import train
from aigopt.evaluator import Evaluator
from aigopt.probes import ranking_metrics
from aigopt.ranker import RankEvaluator
from aigopt.relative_evaluator import RelativeEvaluator


def evaluate(model,rows):
    groups=defaultdict(list)
    roots={r['circuit']:r['rank_features'] for r in rows if not r['sequence']}
    for row in rows:
        if isinstance(model,RelativeEvaluator):
            bound=model.for_reference_features(roots[row['circuit']][:len(model.names)])
            vector=[v-ref for v,ref in zip(row['rank_features'][:len(model.names)],bound.reference)]
            prediction=bound.predict(vector)
        else:
            vector=row['rank_features'][:len(model.names)] if isinstance(model,RankEvaluator) else row['features']
            prediction=model.predict(vector)
        groups[row['source_name']].append(dict(row,prediction=prediction))
    out={}
    for source,group in groups.items():
        chosen=min(group,key=lambda r:r['prediction']['area']*r['prediction']['delay'])
        best=min(r['area']*r['delay'] for r in group)
        out[source]=dict(ranking_metrics(group),selection_regret=chosen['area']*chosen['delay']/best-1)
    return out


def metric(values):
    paired=[r['pair_accuracy'] for r in values if r['pair_accuracy'] is not None]
    return {'mean_selection_regret':statistics.mean(r['selection_regret'] for r in values),
            'mean_pair_accuracy':statistics.mean(paired) if paired else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();data=json.loads(Path(args.dataset).read_text())
    if not data['complete']:raise ValueError('Incomplete development data')
    suite=json.loads((ROOT/'benchmarks/suite.json').read_text());protocol=suite['protocol']
    forbidden={r['sha256'] for r in suite['test']}
    if forbidden & {r['circuit'] for r in data['rows']}:raise ValueError('Reserved sources in development data')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    frozen={'dataset_sha256':fingerprint(args.dataset),'suite_sha256':fingerprint(ROOT/'benchmarks/suite.json'),
        'protocol':protocol,'source_sha256':{p.name:fingerprint(p) for p in (ROOT/'aigopt').glob('*.py')},
        'runner_sha256':fingerprint(__file__),'test_costs_used':False}
    freeze=out/'selection-freeze.json'
    if freeze.exists() and json.loads(freeze.read_text())!=frozen:raise ValueError('Selection configuration changed')
    freeze.write_text(json.dumps(frozen,indent=2)+'\n')
    candidates=[];absolute_fold={};original_fold={};structural_fold={}
    previous=Evaluator.load(ROOT/'benchmarks/checkpoints/baseline.json')
    for fold,names in enumerate(protocol['cv_folds']):
        val=[r for r in data['rows'] if Path(r['source_name']).stem in names]
        fitting=[r for r in data['rows'] if Path(r['source_name']).stem not in names]
        assert not {r['circuit'] for r in fitting}&{r['circuit'] for r in val}
        path=out/f'absolute4-fold{fold}.json'
        if path.exists():absolute=Evaluator.load(path)
        else:
            absolute=Evaluator.train(fitting,data['library_sha256'],epochs=1500,seed=314159);absolute.save(path)
        absolute_fold.update(evaluate(absolute,val));original_fold.update(evaluate(previous,val))
        # A disabled residual is the exact structural control, including ties.
        for feature in ('summary4','topology28','load40'):
            start=perf_counter();path=out/f'reference-{feature}-fold{fold}.json'
            if path.exists():model=Evaluator.load(path)
            else:
                model=train(fitting,data['library_sha256'],features=feature,reference_relative=True)
                model.payload['mapping_profile']=data['mapping_profile']
                model.payload['training_report']['validation_circuits']=sorted({r['circuit'] for r in val})
                model.save(path)
            for blend in protocol['blends']:
                model.payload['blend']=blend;values=evaluate(model,val)
                if blend==0:structural_fold.update(values)
                candidates.append({'feature':feature,'fold':fold,'blend':blend,'per_circuit':values})
            print(f'fold{fold} reference-{feature}: {perf_counter()-start:.1f}s',flush=True)
            (out/'cv-progress.json').write_text(json.dumps(candidates,indent=2)+'\n')
    aggregated=[]
    for feature in ('summary4','topology28','load40'):
        for blend in protocol['blends']:
            cases={k:v for row in candidates if row['feature']==feature and row['blend']==blend for k,v in row['per_circuit'].items()}
            aggregated.append({'feature':feature,'blend':blend,**metric(list(cases.values())),'per_circuit':cases})
    winner=min(aggregated,key=lambda c:(c['mean_selection_regret'],-c['mean_pair_accuracy'],c['blend'],{'summary4':4,'topology28':28,'load40':40}[c['feature']]))
    absolute=Evaluator.train(data['rows'],data['library_sha256'],epochs=1500,seed=314159)
    absolute.payload.update(mapping_profile=data['mapping_profile'],dataset_sha256=fingerprint(args.dataset))
    if data.get('source_artifact_sha256'):absolute.payload['original_dataset_sha256']=data['source_artifact_sha256']
    absolute.save(out/'data-only.json')
    selected=train(data['rows'],data['library_sha256'],features=winner['feature'],reference_relative=True)
    selected.payload.update(blend=winner['blend'],mapping_profile=data['mapping_profile'],dataset_sha256=fingerprint(args.dataset),selection_report={k:v for k,v in winner.items() if k!='per_circuit'})
    selected.payload['training_report']['development_circuits_used_for_selection']=sorted({r['circuit'] for r in data['rows']})
    selected.save(out/'selected.json')
    selection={'candidates':aggregated,'winner':winner,'controls':{
        'structural':metric(list(structural_fold.values())),'previous-neural':metric(list(original_fold.values())),
        'data-only-neural':metric(list(absolute_fold.values()))},'test_labels_used':False}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    print('Frozen selection:',{k:v for k,v in winner.items() if k!='per_circuit'},'controls',selection['controls'],flush=True)


if __name__=='__main__':main()
