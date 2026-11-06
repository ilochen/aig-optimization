"""Spend a bounded final mapping budget on distinct, complementary candidates."""
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import shutil
from .abc import ABCError
from .evaluator import positive_metrics
from .objective import Objective


@dataclass(frozen=True)
class Candidate:
    key: str
    state: Path
    graph_sha256: str
    prediction: dict
    plan: tuple = ()


def pool_from_trace(trace, initial, directory, *, prefix=(), decision=0):
    """Reconstruct file/sequence lineage without storing ephemeral absolute paths."""
    entries={row['id']:row for row in trace['candidates']}
    result=[]
    for row in entries.values():
        if row['error']:
            continue
        plan=[];cursor=row
        while cursor['parent'] is not None:
            plan.append(cursor['command']);cursor=entries[cursor['parent']]
        plan.reverse()
        state=Path(initial) if row['id']==0 else Path(directory)/row['state_file']
        result.append(Candidate(f"{decision}:{row['id']}",state,row['graph_sha256'],row['prediction'],(*prefix,*plan)))
    return result


def shortlist(candidates, primary, limit, *, objective=None):
    """Primary choice, area/delay extremes, then remaining predicted cost."""
    objective = objective or Objective()
    if limit<1:
        raise ValueError('Finalist limit must be positive')
    positive_metrics(primary.prediction)
    unique={primary.graph_sha256:primary}
    for candidate in candidates:
        positive_metrics(candidate.prediction)
        unique.setdefault(candidate.graph_sha256,candidate)
    ordered=list(unique.values())
    tie=lambda c:(len(c.plan),c.graph_sha256)
    selected=[primary]
    def add(candidate):
        if len(selected)<limit and all(c.graph_sha256!=candidate.graph_sha256 for c in selected):
            selected.append(candidate)
    add(min(ordered,key=lambda c:(c.prediction['area'],*tie(c))))
    add(min(ordered,key=lambda c:(c.prediction['delay'],*tie(c))))
    for candidate in sorted(ordered,key=lambda c:(*objective.key(c.prediction),*tie(c))):
        add(candidate)
    return selected


def finalize(backend, initial, baseline, baseline_sha256, candidates, primary, *,
             limit=1, verification_budget=None, baseline_verified=False, objective=None):
    """Preserve a verified incumbent; never replace it with an unproven graph.

    The primary is verified first, before trying better mapped alternatives.
    Additional failures therefore do not discard an already certified result.
    The verification budget includes proof time already spent on the baseline.
    """
    objective = objective or Objective()
    if not baseline_verified:
        raise ValueError('The baseline must already be proved equivalent')
    if verification_budget is not None and (not math.isfinite(verification_budget) or verification_budget<=0):
        raise ValueError('Verification budget must be finite and positive')
    positive_metrics(baseline)
    selected=shortlist(candidates,primary,limit,objective=objective)
    incumbent_state,incumbent_metrics,incumbent_key=Path(initial),baseline,'baseline'
    incumbent_sha=baseline_sha256
    chosen_plan=()
    records=[];mapped=[];artifact_paths={}
    for candidate in selected:
        row={'key':candidate.key,'graph_sha256':candidate.graph_sha256,
             'prediction':candidate.prediction,'plan':list(candidate.plan),'status':'pending'}
        records.append(row)
        artifact_paths[candidate.key]={'state':candidate.state,'netlist':None}
        if candidate.graph_sha256==baseline_sha256:
            row.update(status='verified_baseline',mapped={k:baseline[k] for k in ('area','delay')})
            continue
        try:
            metrics=backend.map(candidate.state)
            positive_metrics(metrics)
        except (ABCError,subprocess.TimeoutExpired) as error:
            row.update(status='mapping_failed',error=f'{type(error).__name__}: {error}')
            continue
        row['mapped']={k:metrics[k] for k in ('area','delay')}
        artifact_paths[candidate.key]['netlist']=metrics['netlist']
        mapped.append((candidate,metrics,row))
    # Mapping every finalist cannot take time away from certification: proof
    # accounting uses only backend verification_seconds, not elapsed wall time.
    primary_jobs=[job for job in mapped if job[0].graph_sha256==primary.graph_sha256]
    others=sorted([job for job in mapped if job not in primary_jobs],
                  key=lambda job:objective.key(job[1]))
    for candidate,metrics,row in [*primary_jobs,*others]:
        if not objective.better(metrics,incumbent_metrics):
            row['status']='not_improving'
            continue
        def prove(reference,target):
            if verification_budget is None:
                result=backend.verify(reference,target)
            else:
                remaining=verification_budget-backend.verification_seconds
                if remaining<=0:
                    raise subprocess.TimeoutExpired('verification budget',verification_budget)
                result=backend.verify(reference,target,timeout=remaining)
            if result is not True:
                raise RuntimeError('Equivalence was not proved')
        try:
            prove(initial,candidate.state)
            prove(initial,metrics['netlist'])
        except (ABCError,subprocess.TimeoutExpired) as error:
            row.update(status='unproven',error=f'{type(error).__name__}: {error}')
            continue
        row['status']='verified'
        incumbent_state,incumbent_metrics,incumbent_key=candidate.state,metrics,candidate.key
        incumbent_sha=candidate.graph_sha256
        chosen_plan=candidate.plan
    return {'state':incumbent_state,'metrics':incumbent_metrics,'key':incumbent_key,
        'plan':list(chosen_plan),'records':records,'finalist_limit':limit,
        'artifact_paths':artifact_paths,
        'verification_budget_seconds':verification_budget,
        'verification_seconds':backend.verification_seconds,
        'verification_budget_exhausted':verification_budget is not None and backend.verification_seconds>=verification_budget,
        'graph_sha256':incumbent_sha,
        'primary_key':primary.key,'primary_retained':incumbent_sha==primary.graph_sha256,
        'objective':objective.payload(),'objective_result':objective.describe(incumbent_metrics),
        'equivalence_proven':True}


def retain_artifacts(result, reference, library, destination):
    """Retain unproven finalists for review without confusing them with output."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(reference,destination/'reference.aig')
    shutil.copyfile(library,destination/('library.lib' if Path(library).suffix.lower()=='.lib' else 'library.genlib'))
    for index,row in enumerate(result['records']):
        paths=result['artifact_paths'][row['key']]
        aig=destination/f'candidate-{index}.aig'
        shutil.copyfile(paths['state'],aig)
        row['artifact_aig']=f'{destination.name}/{aig.name}'
        if paths['netlist'] is not None:
            netlist=destination/f'candidate-{index}.mapped.blif'
            shutil.copyfile(paths['netlist'],netlist)
            row['artifact_netlist']=f'{destination.name}/{netlist.name}'


def summary(result):
    return {key:value for key,value in result.items() if key not in ('state','metrics','artifact_paths')}
