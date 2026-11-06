from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from aigopt.abc import ABC
from aigopt.finalists import Candidate, finalize, shortlist, pool_from_trace, retain_artifacts


class Backend:
    def __init__(self, failures=()):
        self.mapping_calls=0;self.verification_seconds=.1
        self.calls=[];self.failures=failures
    def map(self,state):
        self.mapping_calls+=1
        area,delay={'p':(9,10),'a':(8,10),'d':(9,9)}[state.stem]
        return {'area':area,'delay':delay,'netlist':state.with_suffix('.mapped.blif')}
    def verify(self,reference,target,*,timeout=None):
        self.calls.append(target.name)
        if target.stem in self.failures:
            self.verification_seconds+=timeout if timeout is not None else 1
            raise subprocess.TimeoutExpired('proof',timeout)
        self.verification_seconds+=.1
        return True


def fixture(root):
    candidates=[Candidate('p',root/'p.aig','p',{'area':1,'delay':10},('p',)),
        Candidate('a',root/'a.aig','a',{'area':.5,'delay':40},('a',)),
        Candidate('d',root/'d.aig','d',{'area':4,'delay':3},('d',))]
    baseline={'area':10,'delay':10,'netlist':root/'baseline.blif'}
    return candidates,baseline


class FinalistTests(unittest.TestCase):
    def test_proof_timeout_override_does_not_change_transform_timeout(self):
        library=Path(__file__).resolve().parents[1]/'examples/teaching.genlib'
        backend=ABC('/unused/abc',library,timeout=5)
        response=SimpleNamespace(returncode=0,stdout='Networks are equivalent',stderr='')
        with patch('aigopt.abc.subprocess.run',return_value=response) as run:
            backend.verify('/unused/a.aig','/unused/b.aig',timeout=30)
            backend.run(['version'],'/unused')
        self.assertEqual(run.call_args_list[0].kwargs['timeout'],30)
        self.assertEqual(run.call_args_list[1].kwargs['timeout'],5)

    def test_shortlist_is_distinct_and_preserves_primary(self):
        pool,_=fixture(Path('/unused'))
        duplicate=Candidate('duplicate',Path('/unused/other.aig'),'a',{'area':.5,'delay':40})
        result=shortlist([*pool,duplicate],pool[0],3)
        self.assertEqual([c.key for c in result],['p','a','d'])
        self.assertEqual(len(shortlist(pool,pool[0],1)),1)

    def test_extra_mapping_recovers_a_better_measured_candidate(self):
        root=Path('/unused');pool,baseline=fixture(root)
        one=finalize(Backend(),root/'initial.aig',baseline,'base',pool,pool[0],limit=1,baseline_verified=True)
        backend=Backend()
        three=finalize(backend,root/'initial.aig',baseline,'base',pool,pool[0],limit=3,baseline_verified=True)
        self.assertEqual(one['metrics']['area']*one['metrics']['delay'],90)
        self.assertEqual(three['metrics']['area']*three['metrics']['delay'],80)
        self.assertEqual(backend.mapping_calls,3)
        self.assertEqual(backend.calls[:2],['p.aig','p.mapped.blif'])
        self.assertEqual(three['key'],'a')

    def test_late_timeout_cannot_discard_verified_primary(self):
        root=Path('/unused');pool,baseline=fixture(root)
        backend=Backend(failures=('a',))
        result=finalize(backend,root/'initial.aig',baseline,'base',pool,pool[0],limit=3,
            verification_budget=1.,baseline_verified=True)
        self.assertEqual(result['key'],'p')
        self.assertTrue(result['equivalence_proven'])
        self.assertTrue(result['verification_budget_exhausted'])
        self.assertAlmostEqual(backend.verification_seconds,1.)
        self.assertEqual(result['records'][1]['status'],'unproven')

    def test_all_unproven_candidates_return_verified_baseline(self):
        root=Path('/unused');pool,baseline=fixture(root)
        backend=Backend(failures=('p',))
        result=finalize(backend,root/'initial.aig',baseline,'base',pool,pool[0],limit=3,
            verification_budget=1.,baseline_verified=True)
        self.assertEqual(result['key'],'baseline')
        self.assertEqual(result['metrics'],baseline)
        self.assertTrue(all(row['status']=='unproven' for row in result['records']))

    def test_baseline_reuse_does_not_spend_mapping_budget(self):
        root=Path('/unused');pool,baseline=fixture(root)
        primary=Candidate('root',root/'initial.aig','base',{'area':10,'delay':10})
        backend=Backend()
        result=finalize(backend,root/'initial.aig',baseline,'base',[primary],primary,baseline_verified=True)
        self.assertEqual(backend.mapping_calls,0)
        self.assertTrue(result['primary_retained'])
        with self.assertRaises(ValueError):
            finalize(backend,root/'initial.aig',baseline,'base',pool,pool[0])

    def test_trace_lineage_crosses_decision_prefix(self):
        trace={'candidates':[{'id':0,'parent':None,'command':None,'state_file':'old.aig',
            'error':None,'graph_sha256':'base','prediction':{'area':2,'delay':2}},
            {'id':1,'parent':0,'command':'balance','state_file':'node-1.aig',
            'error':None,'graph_sha256':'child','prediction':{'area':1,'delay':2}}]}
        pool=pool_from_trace(trace,'/unused/old.aig','/unused/decision',prefix=('rewrite',),decision=1)
        self.assertEqual(pool[1].plan,('rewrite','balance'))
        self.assertEqual(pool[1].state,Path('/unused/decision/node-1.aig'))

    def test_explicit_nonproof_cannot_be_exported_as_verified(self):
        root=Path('/unused');pool,baseline=fixture(root)
        backend=Backend();backend.verify=lambda *args,**kwargs:False
        with self.assertRaisesRegex(RuntimeError,'not proved'):
            finalize(backend,root/'initial.aig',baseline,'base',pool,pool[0],baseline_verified=True)

    def test_unproven_artifacts_are_retained_with_their_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool,baseline=fixture(root)
            reference=root/'initial.aig';reference.write_text('reference')
            library=root/'library.genlib';library.write_text('library')
            for candidate in pool:
                candidate.state.write_text(candidate.key)
                candidate.state.with_suffix('.mapped.blif').write_text(candidate.key+' mapped')
            result=finalize(Backend(failures=('p',)),reference,baseline,'base',pool,pool[0],
                limit=3,verification_budget=1,baseline_verified=True)
            retain_artifacts(result,reference,library,root/'finalists')
            self.assertEqual(result['state'],reference)
            self.assertTrue(all(r['status']=='unproven' for r in result['records']))
            self.assertTrue(all((root/r['artifact_aig']).is_file() for r in result['records']))
            self.assertTrue(all((root/r['artifact_netlist']).is_file() for r in result['records']))


if __name__=='__main__':unittest.main()
