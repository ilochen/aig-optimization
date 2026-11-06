"""Blend the calibrated evaluator with a learned reference-relative correction."""
import json
import math
from pathlib import Path
from .evaluator import Evaluator,positive_metrics
from .relative_evaluator import RelativeEvaluator


class HybridEvaluator:
    def __init__(self,payload,relative=None,reference_features=None):
        if payload.get('kind')!='hybrid_reference_mlp_v1' or not 0<=payload.get('mix',0)<=1:
            raise ValueError('Invalid hybrid evaluator')
        self.payload=payload;self.previous=Evaluator(payload['previous'])
        self.relative=relative or RelativeEvaluator(payload['relative'])
        if any(m.payload['library_sha256']!=payload['library_sha256'] for m in (self.previous,self.relative)):
            raise ValueError('Hybrid library mismatch')
        self.reference_features=reference_features

    def for_reference(self,graph):
        return self if self.reference_features is not None else HybridEvaluator(self.payload,self.relative.for_reference(graph),graph.features())

    def encode(self,graph):return [*graph.features(),*self.relative.encode(graph)]

    def predict(self,features):
        if self.reference_features is None:raise ValueError('Bind a reference graph before hybrid inference')
        if len(features)!=4+len(self.relative.names):raise ValueError('Hybrid feature schema mismatch')
        previous=self.previous.predict(features[:4]);reference=self.previous.predict(self.reference_features)
        relative=self.relative.predict(features[4:]);mix=self.payload['mix']
        normalized={k:previous[k]/reference[k]*max(1,self.reference_features[i]) for i,k in enumerate(('area','delay'))}
        if mix==0 or normalized==relative:return normalized
        return positive_metrics({k:math.exp((1-mix)*math.log(normalized[k])+mix*math.log(relative[k])) for k in ('area','delay')})

    def evaluate(self,state,graph):return self.predict(self.encode(graph))

    def domain_graph(self,graph):
        previous=self.previous.domain(graph.features());relative=self.relative.domain_graph(graph)
        return {'known':previous['known'] and relative['known'],
            'outside_features':['base:'+name for name in previous['outside_features']]+['change:'+name for name in relative['outside_features']]}

    def context(self):return {**self.relative.context(),'kind':'hybrid_fixed_original_graph','base_features':self.reference_features}

    def save(self,path):Path(path).write_text(json.dumps(self.payload,indent=2,allow_nan=False)+'\n')
