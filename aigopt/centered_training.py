"""Circuit-balanced relative-cost fitting and gap-weighted pair ranking."""
from collections import defaultdict
import math
import random
import statistics
from .descriptors import NAMES as BASE_NAMES, VERSION as BASE_VERSION
from .load_descriptors import NAMES as LOAD_NAMES, VERSION as LOAD_VERSION
from .ranker import RankEvaluator, prior
from .evaluator import positive_metrics


def pair_loss_gradient(difference, target):
    """Smooth pair loss: ignore <0.5% gaps and downweight uncertain small gaps."""
    if abs(target)<.005:return 0.,0.
    sign=1. if target>0 else -1.
    weight=min(1.,abs(target)/.1)
    z=(sign*difference-min(.02,abs(target)))/.05
    loss=.0025*weight*(max(0.,-z)+math.log1p(math.exp(-abs(z))))
    derivative=-.05*weight*sign/(1+math.exp(max(-35.,min(35.,z))))
    return loss,derivative


def apply_batch(model, examples, rate, *, anchor=.01):
    """Exact gradient of a circuit-centered two-metric + ADP loss.

    Every circuit contributes one averaged update regardless of label count.
    The weak mean anchor calibrates units without dominating state differences.
    """
    forward=[model.forward(x) for x,target in examples]
    errors=[[p[k]-target[k] for k in range(2)] for (x,target),(h,p) in zip(examples,forward)]
    means=[statistics.mean(e[k] for e in errors) for k in range(2)]
    W,V=model.payload['W'],model.payload['V'];n=len(examples)
    gw=[[0.]*len(w) for w in W];gv=[[0.]*len(v) for v in V];loss=0.
    for (x,target),(h,p),error in zip(examples,forward,errors):
        centered=[error[k]-means[k] for k in range(2)]
        loss+=(.1*sum(v*v for v in centered)+.25*sum(centered)**2)/n
        grad=[(.2*centered[k]+.5*sum(centered)+anchor*means[k])/n for k in range(2)]
        for k in range(2):
            for j,value in enumerate(h+[1.]):gv[k][j]+=grad[k]*value
        for j in range(len(W)):
            dh=(1-h[j]**2)*sum(grad[k]*V[k][j] for k in range(2))
            for i,value in enumerate(x):gw[j][i]+=dh*value
    loss+=.5*anchor*sum(v*v for v in means)
    for matrix,gradient in ((W,gw),(V,gv)):
        for row,grow in zip(matrix,gradient):
            for i,g in enumerate(grow):row[i]-=rate*g
    return loss


def update_order_pair(model, xa, xb, base_difference, target_difference, rate, weight):
    ha,pa=model.forward(xa);hb,pb=model.forward(xb)
    loss,derivative=pair_loss_gradient(base_difference+sum(pa)-sum(pb),target_difference)
    derivative*=weight
    W,V=model.payload['W'],model.payload['V']
    dh_a=[(1-h*h)*sum(derivative*V[k][j] for k in range(2)) for j,h in enumerate(ha)]
    dh_b=[(1-h*h)*sum(-derivative*V[k][j] for k in range(2)) for j,h in enumerate(hb)]
    for k in range(2):
        for j,(a,b) in enumerate(zip(ha+[1.],hb+[1.])):V[k][j]-=rate*derivative*(a-b)
    for j in range(len(W)):
        for i in range(len(xa)):W[j][i]-=rate*(dh_a[j]*xa[i]+dh_b[j]*xb[i])
    return weight*loss


def train(rows, library_sha256, *, features='load40', hidden=4, epochs=300,
          pair_weight=.1, weight_decay=.002, seed=271828, reference_relative=False):
    if features not in ('summary4','topology28','load40') or not rows or min(hidden,epochs)<1:
        raise ValueError('Invalid centered training configuration')
    names=LOAD_NAMES if features=='load40' else BASE_NAMES
    version=LOAD_VERSION if features=='load40' else BASE_VERSION
    size=len(names);rng=random.Random(seed)
    absolute_vectors=[r['rank_features'][:size] for r in rows]
    references={r['circuit']:r['rank_features'][:size] for r in rows if not r['sequence']} if reference_relative else {}
    if reference_relative and set(references)!={r['circuit'] for r in rows}:
        raise ValueError('Every circuit needs its original reference graph')
    vectors=[[v-ref for v,ref in zip(vector,references[row['circuit']])] for row,vector in zip(rows,absolute_vectors)] if reference_relative else absolute_vectors
    if any(len(v)!=size or any(not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError('Invalid training descriptors')
    for row in rows:positive_metrics(row)
    active=[0,1,4,9] if features=='summary4' else list(range(size))
    means=[0.]*size if reference_relative else [statistics.mean(v[i] for v in vectors) for i in range(size)]
    scales=[max(.03,statistics.pstdev(v[i] for v in vectors)) for i in range(size)]
    model=RankEvaluator({'kind':'rank_mlp_v1','descriptor_version':version,'descriptor_names':names,
        'library_sha256':library_sha256,'hidden':hidden,'blend':1.,'active_indices':active,
        'mean':means,'scale':scales,'ranges':[[min(v[i] for v in vectors),max(v[i] for v in vectors)] for i in range(size)],
        'W':[[rng.uniform(-.15,.15) for _ in range(size+1)] for _ in range(hidden)],
        'V':[[rng.uniform(-.01,.01) for _ in range(hidden+1)] for _ in range(2)],
        'objective':'equal-circuit centered residual metrics + centered log-ADP + gap-weighted logistic order loss',
        'feature_set':features,'seed':seed,'epochs':epochs,'pair_weight':pair_weight,'weight_decay_per_epoch':weight_decay})
    groups=defaultdict(list);examples=[]
    for row,vector,absolute in zip(rows,vectors,absolute_vectors):
        base=prior(absolute);target=[math.log(row[k])-b for k,b in zip(('area','delay'),base)]
        groups[row['circuit']].append(len(examples))
        examples.append((model.inputs(vector),target,sum(base),math.log(row['area']*row['delay'])))
    pairs={}
    for circuit,indices in groups.items():
        pairs[circuit]=[(a,b) for i,a in enumerate(indices) for b in indices[i+1:] if abs(examples[a][3]-examples[b][3])>=.005]
        rng.shuffle(pairs[circuit]);pairs[circuit]=pairs[circuit][:80]
    def ranking_loss():
        values=[]
        for circuit in groups:
            local=[]
            for a,b in pairs[circuit]:
                xa,_,ba,ta=examples[a];xb,_,bb,tb=examples[b]
                difference=ba-bb+sum(model.forward(xa)[1])-sum(model.forward(xb)[1])
                local.append(pair_loss_gradient(difference,ta-tb)[0])
            if local:values.append(statistics.mean(local))
        return statistics.mean(values) if values else 0.
    before=ranking_loss();order=list(groups)
    for epoch in range(epochs):
        rate=.03/(1+epoch/200);rng.shuffle(order)
        for circuit in order:
            apply_batch(model,[(examples[i][0],examples[i][1]) for i in groups[circuit]],rate,anchor=0. if reference_relative else .01)
            if pair_weight:
                rng.shuffle(pairs[circuit])
                for a,b in pairs[circuit]:
                    xa,_,ba,ta=examples[a];xb,_,bb,tb=examples[b]
                    update_order_pair(model,xa,xb,ba-bb,ta-tb,rate,pair_weight)
        for matrix in (model.payload['W'],model.payload['V']):
            for row in matrix:
                for i in range(len(row)-1):row[i]*=1-weight_decay
    model.payload['training_report']={'train_examples':len(rows),'train_circuits':sorted(groups),
        'validation_circuits':[],'training_pairs':sum(len(v) for v in pairs.values()),
        'initial_order_loss':before,'final_order_loss':ranking_loss()}
    if reference_relative:
        from .relative_evaluator import RelativeEvaluator
        return RelativeEvaluator({'kind':'reference_rank_mlp_v1','network':model.payload,
            'library_sha256':library_sha256,'blend':1.,'feature_set':features,
            'prediction_semantics':'structural-unit cost proxies with learned cost changes relative to fixed original graph',
            'training_report':model.payload['training_report']})
    return model
