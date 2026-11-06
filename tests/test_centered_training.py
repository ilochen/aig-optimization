import copy
import math
import unittest
from aigopt.centered_training import apply_batch,pair_loss_gradient,update_order_pair,train
from aigopt.load_descriptors import describe,NAMES
from aigopt.ranker import RankEvaluator
from aigopt.graph import Graph
from aigopt.descriptors import NAMES as BASE_NAMES,VERSION


def payload():
    return {'kind':'rank_mlp_v1','descriptor_version':VERSION,'descriptor_names':BASE_NAMES,
        'library_sha256':'test','hidden':2,'blend':1.,'mean':[0.]*28,'scale':[1.]*28,
        'ranges':[[-100,100]]*28,'W':[[.01*(j+1)]*29 for j in range(2)],
        'V':[[.03,.02,.01],[-.01,.02,-.01]]}


class RelativeTrainingTests(unittest.TestCase):
    def test_centered_batch_gradient_matches_finite_differences(self):
        initial=payload();examples=[([.2]*28+[1.],[.1,.3]),([-.1]*28+[1.],[-.2,.4])]
        rate=1e-5;updated=RankEvaluator(copy.deepcopy(initial));apply_batch(updated,examples,rate)
        def loss(p):return apply_batch(RankEvaluator(p),examples,0.)
        for key,i,j in [('W',0,0),('W',1,28),('V',0,1),('V',1,2)]:
            plus,minus=copy.deepcopy(initial),copy.deepcopy(initial);epsilon=1e-6
            plus[key][i][j]+=epsilon;minus[key][i][j]-=epsilon
            numerical=(loss(plus)-loss(minus))/(2*epsilon)
            actual=(initial[key][i][j]-updated.payload[key][i][j])/rate
            self.assertAlmostEqual(actual,numerical,places=7)

    def test_pair_gradient_and_gap_threshold(self):
        for difference,target in ((.03,.08),(-.04,.07),(.05,-.12)):
            loss,gradient=pair_loss_gradient(difference,target);epsilon=1e-6
            numerical=(pair_loss_gradient(difference+epsilon,target)[0]-pair_loss_gradient(difference-epsilon,target)[0])/(2*epsilon)
            self.assertAlmostEqual(gradient,numerical,places=7)
        self.assertEqual(pair_loss_gradient(.1,.001),(0.,0.))
        initial=payload();xa=[.2]*28+[1.];xb=[-.1]*28+[1.];rate=1e-5
        updated=RankEvaluator(copy.deepcopy(initial));update_order_pair(updated,xa,xb,.01,.08,rate,.1)
        def loss(p):
            m=RankEvaluator(p);difference=.01+sum(m.forward(xa)[1])-sum(m.forward(xb)[1])
            return .1*pair_loss_gradient(difference,.08)[0]
        for key,i,j in [('W',0,0),('V',0,1),('V',1,2)]:
            plus,minus=copy.deepcopy(initial),copy.deepcopy(initial);epsilon=1e-6
            plus[key][i][j]+=epsilon;minus[key][i][j]-=epsilon
            numerical=(loss(plus)-loss(minus))/(2*epsilon)
            actual=(initial[key][i][j]-updated.payload[key][i][j])/rate
            self.assertAlmostEqual(actual,numerical,places=7)

    def test_load_descriptors_are_finite_and_sensitive_to_connectivity(self):
        empty=describe(Graph((),(1,),()))
        self.assertEqual(len(empty),len(NAMES));self.assertTrue(all(math.isfinite(v) for v in empty))
        a=Graph((2,4,6),(12,),((8,2,4),(10,6,8),(12,9,11)))
        b=Graph(a.inputs,(10,12),a.gates)
        self.assertGreater(describe(b)[28],describe(a)[28])

    def test_centering_removes_circuit_offsets_and_masked_inputs_cannot_change_prediction(self):
        base=payload();examples=[([.1]*28+[1.],[.2,.1]),([-.1]*28+[1.],[-.2,.3])]
        shifted=[(x,[y[0]+100,y[1]-50]) for x,y in examples]
        first,second=RankEvaluator(copy.deepcopy(base)),RankEvaluator(copy.deepcopy(base))
        apply_batch(first,examples,.01,anchor=0);apply_batch(second,shifted,.01,anchor=0)
        for key in ('W','V'):
            for a,b in zip(first.payload[key],second.payload[key]):
                for u,v in zip(a,b):self.assertAlmostEqual(u,v,places=12)
        base['active_indices']=[0,1,4,9];model=RankEvaluator(base)
        one=[.2]*28;two=one[:];two[20]=100
        self.assertEqual(model.predict(one),model.predict(two))

    def test_relative_training_learns_with_strong_between_circuit_offsets(self):
        rows=[]
        for circuit,offset in enumerate((.2,1.,5.,20.)):
            for phase in (0.,.5,1.):
                vector=[math.log1p(10),math.log1p(3)]+[0.]*38;vector[28]=phase
                rows.append({'circuit':str(circuit),'rank_features':vector,'area':10*offset*(1+phase*.3),'delay':3})
        model=train(rows,'test',epochs=100,features='load40')
        report=model.payload['training_report']
        self.assertLess(report['final_order_loss'],report['initial_order_loss'])
        self.assertLess(model.predict(rows[0]['rank_features'])['area']*model.predict(rows[0]['rank_features'])['delay'],
                        model.predict(rows[2]['rank_features'])['area']*model.predict(rows[2]['rank_features'])['delay'])
