import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from aigopt.centered_training import train
from aigopt.evaluator import Evaluator
from aigopt.graph import Graph
from aigopt.load_descriptors import describe
from aigopt.relative_evaluator import RelativeEvaluator


class ReferenceEvaluatorTests(unittest.TestCase):
    def model(self):
        rows=[]
        for circuit,size in enumerate((8,16,32)):
            for fraction in (0.,.1,.2):
                n=int(size*(1-fraction));vector=[math.log1p(n),math.log1p(4)]+[0.]*38
                rows.append({'circuit':str(circuit),'sequence':[] if not fraction else ['x'],
                    'rank_features':vector,'area':n*(1+fraction),'delay':4*(1-fraction)})
        return train(rows,'test',epochs=10,features='load40',reference_relative=True)

    def test_reference_prediction_is_exact_and_binding_does_not_mutate_checkpoint(self):
        model=self.model();a=Graph((2,4),(6,),((6,2,4),));b=Graph((2,4),(8,),((6,2,4),(8,6,2)))
        with self.assertRaisesRegex(ValueError,'Bind a reference'):
            model.predict([0.]*40)
        first=model.for_reference(a);second=model.for_reference(b)
        self.assertIsNone(model.reference)
        self.assertEqual(first.predict(first.encode(a)),{'area':1.,'delay':1.})
        self.assertEqual(second.predict(second.encode(b)),{'area':2.,'delay':2.})
        self.assertIs(first.for_reference(b),first)
        self.assertNotEqual(first.reference,second.reference)
        self.assertEqual(first.context()['graph_sha256'],a.digest())

    def test_roundtrip_dispatch_and_disabled_residual_preserve_structural_cost(self):
        model=self.model();graph=Graph((2,4),(8,),((6,2,4),(8,6,2)))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.json';model.save(path);loaded=Evaluator.load(path)
            self.assertIsInstance(loaded,RelativeEvaluator)
            self.assertEqual(model.for_reference(graph).evaluate(None,graph),loaded.for_reference(graph).evaluate(None,graph))
        p=copy.deepcopy(model.payload);p['blend']=0
        initial=Graph((2,4),(6,),((6,2,4),));bound=RelativeEvaluator(p).for_reference(initial)
        self.assertEqual(bound.evaluate(None,graph),{'area':2.,'delay':2.})

    def test_missing_reference_labels_are_rejected(self):
        rows=[{'circuit':'a','sequence':['x'],'rank_features':[math.log1p(3),math.log1p(2)]+[0.]*38,'area':3,'delay':2}]
        with self.assertRaisesRegex(ValueError,'original reference'):
            train(rows,'test',epochs=1,reference_relative=True)
