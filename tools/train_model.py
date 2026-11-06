"""Select a conservative neural correction to the already trained evaluator."""
import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from aigopt.abc import fingerprint
from aigopt.evaluator import Evaluator
from aigopt.centered_training import train
from aigopt.probes import ranking_metrics
from aigopt.hybrid_evaluator import HybridEvaluator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    p.add_argument('--cv',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    data=json.loads(Path(args.dataset).read_text());suite=json.loads((ROOT/'benchmarks/suite.json').read_text())
    cv_freeze=json.loads((Path(args.cv)/'selection-freeze.json').read_text())
    if cv_freeze['dataset_sha256']!=fingerprint(args.dataset):raise ValueError('CV checkpoints were trained on a different development dataset')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol={'dataset_sha256':fingerprint(args.dataset),'cv_sha256':fingerprint(Path(args.cv)/'selection.json'),
        'correction_features':['summary4','topology28','load40'],'correction_blends':[.25,.5,1.],
        'mixes':[0.,.25,.5,.75,1.],'folds':suite['protocol']['cv_folds'],
        'selection':'minimum out-of-fold mean circuit choice regret; pair accuracy; smaller correction weight; fewer features',
        'test_outcomes_used':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    previous=Evaluator.load(ROOT/'benchmarks/checkpoints/baseline.json');cache={}
    for fold,names in enumerate(protocol['folds']):
        val=[r for r in data['rows'] if Path(r['source_name']).stem in names]
        roots={r['circuit']:r for r in val if not r['sequence']}
        for feature in protocol['correction_features']:
            model=Evaluator.load(Path(args.cv)/f'reference-{feature}-fold{fold}.json')
            for blend in protocol['correction_blends']:
                model.payload['blend']=blend
                for row in val:
                    root=roots[row['circuit']];ref=root['rank_features'][:len(model.names)]
                    bound=model.for_reference_features(ref)
                    delta=[v-r for v,r in zip(row['rank_features'][:len(model.names)],ref)]
                    corrected=bound.predict(delta);base=previous.predict(row['features']);base_root=previous.predict(root['features'])
                    normalized={k:base[k]/base_root[k]*max(1,root['features'][i]) for i,k in enumerate(('area','delay'))}
                    cache[feature,blend,row['circuit'],row['graph_sha256']]=(row,normalized,corrected)
    candidates=[]
    for feature in protocol['correction_features']:
        for blend in protocol['correction_blends']:
            rows=[v for key,v in cache.items() if key[:2]==(feature,blend)]
            for mix in protocol['mixes']:
                groups={}
                for row,base,correction in rows:
                    prediction={k:math.exp((1-mix)*math.log(base[k])+mix*math.log(correction[k])) for k in ('area','delay')}
                    groups.setdefault(row['source_name'],[]).append(dict(row,prediction=prediction))
                per_case={}
                for case,group in groups.items():
                    pick=min(group,key=lambda r:r['prediction']['area']*r['prediction']['delay'])
                    best=min(r['area']*r['delay'] for r in group)
                    per_case[case]=dict(ranking_metrics(group),selection_regret=pick['area']*pick['delay']/best-1)
                candidates.append({'feature':feature,'blend':blend,'mix':mix,
                    'mean_selection_regret':statistics.mean(c['selection_regret'] for c in per_case.values()),
                    'mean_pair_accuracy':statistics.mean(c['pair_accuracy'] for c in per_case.values() if c['pair_accuracy'] is not None),
                    'per_circuit':per_case})
    winner=min(candidates,key=lambda c:(c['mean_selection_regret'],-c['mean_pair_accuracy'],c['mix'],{'summary4':4,'topology28':28,'load40':40}[c['feature']],c['blend']))
    (out/'selection.json').write_text(json.dumps({'protocol':protocol,'candidates':candidates,'winner':winner},indent=2)+'\n')
    # Keep the best-choice model as the reference. This separate candidate
    # targets candidate ordering, with a half-point regret guard.
    baseline=min(c['mean_selection_regret'] for c in candidates if c['mix']==0)
    eligible=[c for c in candidates if c['mean_selection_regret']<=baseline+.005]
    ranking=min(eligible,key=lambda c:(-c['mean_pair_accuracy'],c['mean_selection_regret'],c['mix'],c['blend']))
    print('Best-choice reference:',{k:v for k,v in winner.items() if k!='per_circuit'},flush=True)
    print('Guarded ranking candidate:',{k:v for k,v in ranking.items() if k!='per_circuit'},flush=True)
    relative=train(data['rows'],data['library_sha256'],features=ranking['feature'],reference_relative=True)
    relative.payload.update(blend=ranking['blend'],mapping_profile=data['mapping_profile'],dataset_sha256=fingerprint(args.dataset))
    if data.get('source_artifact_sha256'):relative.payload['original_dataset_sha256']=data['source_artifact_sha256']
    relative.payload['training_report']['development_circuits_used_for_selection']=sorted({r['circuit'] for r in data['rows']})
    payload={'kind':'hybrid_reference_mlp_v1','library_sha256':data['library_sha256'],
        'mapping_profile':data['mapping_profile'],'previous':previous.payload,'relative':relative.payload,
        'mix':ranking['mix'],'training_report':relative.payload['training_report'],
        'dataset_sha256':fingerprint(args.dataset),'prediction_semantics':'fixed-reference cost-ratio proxies; not absolute physical costs',
        'selection_report':{k:v for k,v in ranking.items() if k!='per_circuit'}}
    if data.get('source_artifact_sha256'):payload['original_dataset_sha256']=data['source_artifact_sha256']
    HybridEvaluator(payload).save(out/'selected.json')
    import shutil
    shutil.copyfile(Path(args.cv)/'data-only.json',out/'data-only.json')
    selection={'protocol':{**protocol,'ranking_policy':'maximize pair accuracy subject to <=0.5 percentage point additional mean development top-choice regret'},
        'candidates':candidates,'best_choice_reference':winner,'winner':ranking,'test_outcomes_used':False}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')


if __name__=='__main__':main()
