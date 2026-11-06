import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from aigopt.descriptors import NAMES, VERSION, describe
from aigopt.graph import Graph
from aigopt.ranker import RankEvaluator
from aigopt.evaluator import Evaluator


def payload(hidden=2):
    return {"kind":"rank_mlp_v1","descriptor_version":VERSION,"descriptor_names":NAMES,
        "library_sha256":"example","hidden":hidden,"blend":1.,"mean":[0.]*28,"scale":[1.]*28,
        "ranges":[[-100,100]]*28,"W":[[.01*(j+1)]*29 for j in range(hidden)],
        "V":[[.03]*(hidden+1),[-.01]*(hidden+1)]}


class DescriptorTests(unittest.TestCase):
    def test_resolves_four_summary_feature_aliasing(self):
        a=Graph((2,4,6),(12,),((8,2,4),(10,6,8),(12,9,11)))
        b=Graph((2,4,6),(12,),((8,3,4),(10,6,8),(12,8,11)))
        self.assertEqual(a.features(),b.features())
        self.assertNotEqual(describe(a),describe(b))
        self.assertEqual(len(describe(a)),len(NAMES))

    def test_empty_graph_and_known_chain(self):
        self.assertTrue(all(math.isfinite(x) for x in describe(Graph((),(1,),()))))
        chain=Graph((2,4,6),(10,),((8,2,4),(10,8,6)))
        vector=dict(zip(NAMES,describe(chain)))
        self.assertEqual(vector['and3_opportunity_fraction'],.5)
        self.assertEqual(vector['critical_gate_fraction'],1.)
        self.assertEqual(vector['positive_only_fraction'],1.)


class RankerTests(unittest.TestCase):
    def test_disabled_residual_is_exact_structural_cost(self):
        p=payload();p['blend']=0.
        model=RankEvaluator(p)
        for n,d in ((3,4),(6,2),(5000,37)):
            features=[math.log1p(n),math.log1p(d)]+[0.]*26
            self.assertEqual(model.predict(features),{'area':float(n),'delay':float(d)})

    def test_pair_gradient_matches_numerical_derivative(self):
        initial=payload()
        model=RankEvaluator(copy.deepcopy(initial))
        xa=[.2]*28+[1.]
        xb=[-.1]*28+[1.]
        difference=sum(model.forward(xa)[1])-sum(model.forward(xb)[1])
        target=difference-.03
        rate,weight=1e-5,.7
        updated=RankEvaluator(copy.deepcopy(initial))
        updated.update_pair(xa,xb,0.,target,rate,weight)
        def loss(p):
            m=RankEvaluator(p)
            error=sum(m.forward(xa)[1])-sum(m.forward(xb)[1])-target
            return .5*weight*error*error
        for key,i,j in [('W',0,0),('V',0,0),('V',0,2)]:
            plus,minus=copy.deepcopy(initial),copy.deepcopy(initial)
            epsilon=1e-6
            plus[key][i][j]+=epsilon;minus[key][i][j]-=epsilon
            numerical=(loss(plus)-loss(minus))/(2*epsilon)
            actual=(initial[key][i][j]-updated.payload[key][i][j])/rate
            self.assertAlmostEqual(actual,numerical,places=7)

    def test_pair_training_learns_a_difference_missing_from_structural_prior(self):
        rows=[]
        for n in (4,8,12,16):
            for phase in (0.,1.):
                features=[math.log1p(n),math.log1p(3),0.,0.]+[0.]*24
                features[20]=phase
                rows.append({'circuit':str(n),'rank_features':features,'area':n*(1+.5*phase),'delay':3.})
        model=RankEvaluator.train(rows,'test',hidden=4,epochs=100)
        report=model.payload['training_report']
        self.assertLess(report['final_pair_mse'],report['initial_pair_mse']*.1)
        first,second=rows[0],rows[1]
        self.assertLess(model.predict(first['rank_features'])['area']*model.predict(first['rank_features'])['delay'],
                        model.predict(second['rank_features'])['area']*model.predict(second['rank_features'])['delay'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ranker.json'
            model.save(path)
            loaded=Evaluator.load(path)
            self.assertIsInstance(loaded,RankEvaluator)
            self.assertEqual(model.predict(first['rank_features']),loaded.predict(first['rank_features']))

    def test_wrong_schema_and_nonfinite_weights_are_rejected(self):
        for key,value in [('descriptor_version','wrong'),('scale',[0.]*28),('blend',2.)]:
            with self.assertRaises(ValueError):RankEvaluator(dict(payload(),**{key:value}))
        p=payload();p['V'][0][0]=float('nan')
        with self.assertRaises(ValueError):RankEvaluator(p)


if __name__=='__main__':unittest.main()
