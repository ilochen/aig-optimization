import math
from pathlib import Path
import tempfile
import unittest
from aigopt.evaluator import reward
from aigopt.finalists import Candidate, finalize
from aigopt.objective import Objective
from aigopt.search import search
from aigopt.abc import Action
from aigopt.graph import Graph


class ObjectiveTests(unittest.TestCase):
    def test_default_preserves_adp_rewards_and_area_delay_have_distinct_choices(self):
        baseline = {'area': 10, 'delay': 10}
        small = {'area': 5, 'delay': 15}
        fast = {'area': 12, 'delay': 5}
        self.assertEqual(Objective().reward(small, baseline), reward(small, baseline))
        self.assertTrue(Objective('area').better(small, fast))
        self.assertTrue(Objective('delay').better(fast, small))
        self.assertTrue(Objective().better(fast, small))

    def test_feasibility_outranks_lower_adp_and_infeasible_fallback_is_explicit(self):
        obj = Objective('area', max_delay=5)
        cheap = {'area': 1, 'delay': 6}
        feasible = {'area': 50, 'delay': 5}
        self.assertTrue(obj.better(feasible, cheap))
        self.assertGreater(obj.reward(feasible, cheap), obj.reward(cheap, cheap))
        self.assertFalse(obj.describe(cheap)['feasible'])
        self.assertTrue(obj.better({'area': 2, 'delay': 5.5}, cheap))
        self.assertTrue(Objective(max_area=10, max_delay=5).feasible({'area': 10, 'delay': 5}))
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                Objective(max_delay=value)

    def test_search_uses_predicted_feasibility(self):
        class Backend:
            def apply(self, source, action, destination):
                # Distinct valid graph encodings; the evaluator supplies costs.
                destination.write_text('aag 3 2 0 1 1\n2\n4\n6\n6 '+('5' if action.name=='fast' else '4')+' 2\n')
                return destination
        class Predictor:
            def predict(self, features):
                return {'area': 50, 'delay': 5} if features[2] else {'area': 1, 'delay': 6}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root/'initial.aig';initial.write_text('aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n')
            node, trace = search(Backend(), initial, Predictor(), [Action('cheap','cheap'), Action('fast','fast')],
                root, iterations=2, max_depth=1, choice='best', randomize=False,
                objective=Objective('area', max_delay=5))
            self.assertEqual(trace['plan'], ['fast'])
            self.assertEqual(Graph.read(node.state).features()[2], 1)

    def test_measured_finalization_overrides_optimistic_feasibility_and_handles_impossible_limit(self):
        class Backend:
            verification_seconds = 0
            def map(self, state):
                return dict({'area': 1, 'delay': 6} if state.stem=='cheap' else {'area': 50, 'delay': 5}, netlist=state)
            def verify(self, *args, **kwargs):return True
        cheap = Candidate('cheap',Path('cheap.aig'),'cheap',{'area':1,'delay':4})
        fast = Candidate('fast',Path('fast.aig'),'fast',{'area':50,'delay':3})
        baseline = {'area':10,'delay':7,'netlist':Path('baseline.blif')}
        result = finalize(Backend(),Path('initial.aig'),baseline,'base',[cheap,fast],cheap,
            limit=2,baseline_verified=True,objective=Objective('area',max_delay=5))
        self.assertEqual(result['key'],'fast')
        self.assertTrue(result['objective_result']['feasible'])
        # This is a useful violation reduction, not a claim to meet the limit.
        impossible = finalize(Backend(),Path('initial.aig'),baseline,'base',[cheap,fast],cheap,
            limit=2,baseline_verified=True,objective=Objective('area',max_delay=1))
        self.assertFalse(impossible['objective_result']['feasible'])
        self.assertEqual(impossible['metrics']['delay'],5)
