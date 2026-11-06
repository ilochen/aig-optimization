import copy
import json
from pathlib import Path
import tempfile
import unittest
from aigopt.evaluator import Evaluator
from aigopt.hybrid_evaluator import HybridEvaluator
from aigopt.graph import Graph
from aigopt.load_descriptors import VERSION,NAMES

ROOT=Path(__file__).resolve().parents[1]


def payload():
    previous=Evaluator.load(ROOT/'benchmarks/checkpoints/baseline.json').payload
    network={'kind':'rank_mlp_v1','descriptor_version':VERSION,'descriptor_names':NAMES,
        'library_sha256':previous['library_sha256'],'hidden':1,'mean':[0.]*40,'scale':[1.]*40,
        'ranges':[[-100,100]]*40,'W':[[.01]*41],'V':[[.02,.01],[-.01,.02]]}
    relative={'kind':'reference_rank_mlp_v1','network':network,'library_sha256':previous['library_sha256'],'blend':.5}
    return {'kind':'hybrid_reference_mlp_v1','previous':previous,'relative':relative,
        'library_sha256':previous['library_sha256'],'mix':.75}


class HybridTests(unittest.TestCase):
    def test_fixed_reference_binding_roundtrip_and_exact_root_prediction(self):
        model=HybridEvaluator(payload());graph=Graph((2,4),(8,),((6,2,4),(8,6,2)))
        bound=model.for_reference(graph)
        self.assertEqual(bound.evaluate(None,graph),{'area':2.,'delay':2.})
        self.assertIsNone(model.reference_features)
        self.assertIs(bound.for_reference(Graph((2,4),(6,),((6,2,4),))),bound)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';model.save(path);loaded=Evaluator.load(path)
            self.assertIsInstance(loaded,HybridEvaluator)
            self.assertEqual(loaded.for_reference(graph).evaluate(None,graph),bound.evaluate(None,graph))
        self.assertEqual(bound.context()['graph_sha256'],graph.digest())

    def test_disabled_correction_preserves_original_relative_costs(self):
        p=payload();p['mix']=0;model=HybridEvaluator(p)
        root=Graph((2,4),(8,),((6,2,4),(8,6,2)));candidate=Graph((2,4),(6,),((6,2,4),))
        bound=model.for_reference(root);old=Evaluator(p['previous'])
        before=old.predict(root.features());after=old.predict(candidate.features())
        self.assertEqual(bound.evaluate(None,candidate),{'area':after['area']/before['area']*2,'delay':after['delay']/before['delay']*2})
        with self.assertRaisesRegex(ValueError,'Bind a reference'):
            model.predict([0.]*44)

    def test_library_and_feature_mismatches_rejected(self):
        p=payload();p['library_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'library mismatch'):HybridEvaluator(p)
        graph=Graph((2,4),(6,),((6,2,4),));bound=HybridEvaluator(payload()).for_reference(graph)
        with self.assertRaisesRegex(ValueError,'feature schema'):bound.predict([1]*4)
