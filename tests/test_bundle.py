import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from aigopt.abc import fingerprint
from aigopt.bundle import check_bundle, verify_bundle

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('ABC_BINARY'), 'Set ABC_BINARY for portable bundle integration')
class BundleIntegrationTests(unittest.TestCase):
    def test_moved_bundle_verifies_and_corruption_and_false_claims_fail(self):
        with tempfile.TemporaryDirectory(prefix='portable result ') as directory:
            root=Path(directory);output=root/'original'
            result=subprocess.run([sys.executable,'-m','aigopt','optimize','--abc',os.environ['ABC_BINARY'],
                '--library',str(ROOT/'examples/teaching.genlib'),'--model',str(ROOT/'examples/evaluator.json'),
                '--input',str(ROOT/'examples/chain-4.blif'),'--output',str(output),'--objective','area',
                '--max-delay','4','--finalists','3','--verification-budget','30','--choice','best',
                '--iterations','24','--widening','1','--steps','2','--compact-trace','--retain-finalists'],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            moved=root/'moved';output.rename(moved)
            before={p.relative_to(moved).as_posix():fingerprint(p) for p in moved.rglob('*') if p.is_file()}
            report=verify_bundle(moved,os.environ['ABC_BINARY'])
            self.assertTrue(report['metrics_reproduced'])
            self.assertTrue(report['objective_result']['feasible'])
            self.assertEqual(before,{p.relative_to(moved).as_posix():fingerprint(p) for p in moved.rglob('*') if p.is_file()})
            manifest=json.loads((moved/'manifest.json').read_text())
            saved_manifest=json.dumps(manifest)
            # Simple corruption is rejected before invoking ABC.
            netlist=moved/'final.mapped.blif';original=netlist.read_bytes()
            netlist.write_bytes(original+b'\n# altered\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                check_bundle(moved)
            # Even a recomputed inventory cannot bless a false functional result.
            netlist.write_text('.model wrong\n.inputs a b c d0 d1 d2 d3\n.outputs F\n.names a F\n1 1\n.end\n')
            manifest['files']['final.mapped.blif']=fingerprint(netlist)
            (moved/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'Equivalence'):
                verify_bundle(moved,os.environ['ABC_BINARY'])
            netlist.write_bytes(original)
            manifest=json.loads(saved_manifest)
            # A measured cost claim must also survive independent remapping.
            trace_path=moved/'trace.json';trace=json.loads(trace_path.read_text())
            trace['final_mapped']['area']/=2
            trace_path.write_text(json.dumps(trace))
            manifest['files']['trace.json']=fingerprint(trace_path)
            (moved/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'area mismatch'):
                verify_bundle(moved,os.environ['ABC_BINARY'])
            # A path escape is invalid even when it points to a real file.
            manifest['files']['../escape']=manifest['files']['final.aig']
            (moved/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Unsafe artifact path'):
                check_bundle(moved)

    def test_unmet_limits_are_not_reported_as_feasible(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'impossible'
            result=subprocess.run([sys.executable,'-m','aigopt','optimize','--abc',os.environ['ABC_BINARY'],
                '--library',str(ROOT/'examples/teaching.genlib'),'--model',str(ROOT/'examples/evaluator.json'),
                '--input',str(ROOT/'examples/chain-4.blif'),'--output',str(output),'--max-delay','.01',
                '--iterations','4','--compact-trace'],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            trace=json.loads((output/'trace.json').read_text())
            self.assertEqual(trace['result_status'],'constraints_unmet')
            self.assertFalse(verify_bundle(output,os.environ['ABC_BINARY'])['objective_result']['feasible'])
