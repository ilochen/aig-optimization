"""Real Liberty mapping, nonlinear timing and portable revalidation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from aigopt.abc import ABC, quote
from aigopt.bundle import verify_bundle

ROOT=Path(__file__).resolve().parents[1]
LIBRARY=ROOT/'libraries/nangate45/NangateOpenCellLibrary_typical.lib'


class ProfileTests(unittest.TestCase):
    def test_liberty_checkpoints_require_the_same_measurement_profile(self):
        backend=ABC('/unused/abc',LIBRARY)
        for payload in ({},{'mapping_profile':{**backend.mapping_profile,'output_load_ff':1}}):
            with self.assertRaisesRegex(ValueError,'profile mismatch'):
                backend.validate_profile(payload)
        backend.validate_profile({'mapping_profile':backend.mapping_profile})
        legacy=ABC('/unused/abc',ROOT/'examples/teaching.genlib')
        legacy.validate_profile({})


@unittest.skipUnless(os.environ.get('ABC_BINARY'),'Set ABC_BINARY for real Liberty integration')
class LibertyIntegrationTests(unittest.TestCase):
    def test_full_library_timing_uses_load_and_exported_netlist_reproduces(self):
        with tempfile.TemporaryDirectory(prefix='liberty with spaces ') as directory:
            work=Path(directory);library=work/'real cells.lib';shutil.copyfile(LIBRARY,library)
            backend=ABC(os.environ['ABC_BINARY'],library,timeout=30)
            initial=backend.prepare(ROOT/'examples/chain-4.blif',work/'initial.aig')
            mapped=backend.map(initial)
            self.assertEqual(backend.measure(mapped['netlist']),{k:mapped[k] for k in ('area','delay')})
            self.assertTrue(backend.verify(initial,mapped['netlist']))
            delays=[]
            for load in (1,100):
                constraints=work/f'load-{load}.constr'
                constraints.write_text(f'set_driving_cell BUF_X1\nset_load {load}\n')
                text=backend.run([*backend.library_commands(),f'read {quote(mapped["netlist"])}',
                    f'read_constr {quote(constraints)}','topo','stime'],work)
                delays.append(backend._mapped_metrics(text)[1])
            self.assertGreater(delays[1],delays[0])
            self.assertGreater(mapped['delay'],0)
            self.assertEqual(backend.mapping_profile['delay_unit'],'ps')

    def test_collect_train_optimize_and_independent_verify_keep_liberty_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            def run(*args,success=True):
                result=subprocess.run([sys.executable,'-m','aigopt',*map(str,args)],cwd=ROOT,capture_output=True,text=True)
                self.assertEqual(result.returncode==0,success,result.stdout+result.stderr)
                return result
            common=['--abc',os.environ['ABC_BINARY'],'--library',LIBRARY]
            run('collect',*common,'--samples','3','--output',work/'labels.json',
                ROOT/'examples/chain-2.blif',ROOT/'examples/chain-3.blif')
            run('train','--dataset',work/'labels.json','--output',work/'model.json','--epochs','100')
            model=json.loads((work/'model.json').read_text())
            self.assertEqual(model['mapping_profile']['schema'],'abc-liberty-map-stime-v1')
            run('optimize',*common,'--model',work/'model.json','--input',ROOT/'examples/chain-4.blif',
                '--choice','best','--iterations','8','--finalists','3','--verification-budget','30',
                '--compact-trace','--output',work/'result')
            self.assertTrue((work/'result/library.lib').is_file())
            self.assertFalse((work/'result/library.genlib').exists())
            self.assertTrue(verify_bundle(work/'result',os.environ['ABC_BINARY'])['metrics_reproduced'])
            model['mapping_profile']['output_load_ff']=1
            (work/'changed.json').write_text(json.dumps(model))
            failed=run('optimize',*common,'--model',work/'changed.json','--input',ROOT/'examples/chain-4.blif',
                '--output',work/'rejected',success=False)
            self.assertIn('profile mismatch',failed.stderr)
            self.assertFalse((work/'rejected').exists())
