"""Neural cost-change prediction conditioned on one fixed original graph."""
import json
import math
from pathlib import Path
from .descriptors import describe as describe_base
from .load_descriptors import VERSION as LOAD_VERSION,describe as describe_load
from .ranker import RankEvaluator
from .evaluator import positive_metrics


class RelativeEvaluator:
    def __init__(self,payload,reference=None,reference_digest=None):
        if payload.get('kind')!='reference_rank_mlp_v1':raise ValueError('Unknown relative evaluator schema')
        self.payload=payload;self.network=RankEvaluator(payload['network']);self.names=self.network.names
        if not 0<=payload.get('blend',1.)<=1:raise ValueError('Invalid relative residual blend')
        self.describe=describe_load if self.network.payload['descriptor_version']==LOAD_VERSION else describe_base
        if reference is not None and (len(reference)!=len(self.names) or any(not math.isfinite(v) for v in reference)):
            raise ValueError('Invalid reference descriptors')
        self.reference=reference;self.reference_digest=reference_digest

    def for_reference(self,graph):
        # A bound evaluator stays anchored across replanning decisions. Binding
        # an unbound checkpoint returns a new object, so separate circuits do
        # not share mutable context.
        return self if self.reference is not None else RelativeEvaluator(self.payload,self.describe(graph),graph.digest())

    def for_reference_features(self,features):
        return self if self.reference is not None else RelativeEvaluator(self.payload,list(features))

    def encode(self,graph):
        if self.reference is None:raise ValueError('Bind a reference graph before relative inference')
        return [v-r for v,r in zip(self.describe(graph),self.reference)]

    def predict(self,delta):
        if self.reference is None:raise ValueError('Bind a reference graph before relative inference')
        x=self.network.inputs(delta)
        count=max(1,round(math.expm1(delta[0]+self.reference[0])))
        depth=max(1,round(math.expm1(delta[1]+self.reference[1])))
        blend=self.payload.get('blend',1.)
        if blend==0:return {'area':float(count),'delay':float(depth)}
        _,values=self.network.forward(x)
        _,zero=self.network.forward(self.network.inputs([0.]*len(self.names)))
        return positive_metrics({key:base*math.exp(max(-20,min(20,blend*(value-anchor))))
            for key,base,value,anchor in zip(('area','delay'),(count,depth),values,zero)})

    def evaluate(self,state,graph):return self.predict(self.encode(graph))

    def domain_graph(self,graph):
        delta=self.encode(graph)
        return {'known':True,'outside_features':[name for name,value,bounds in zip(self.names,delta,self.network.payload['ranges']) if not bounds[0]<=value<=bounds[1]]}

    def context(self):
        return {'kind':'fixed_original_graph','graph_sha256':self.reference_digest,'descriptors':self.reference}

    def save(self,path):Path(path).write_text(json.dumps(self.payload,indent=2,allow_nan=False)+'\n')
