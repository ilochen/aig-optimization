"""Set ABC_BINARY to run actual synthesis, mapping and SAT equivalence tests."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from aigopt.abc import ABC, Action, actions
from aigopt.graph import Graph

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("ABC_BINARY"), "Set ABC_BINARY for real ABC integration")
class ABCIntegrationTests(unittest.TestCase):
    def test_all_actions_preserve_truth_table_and_map_equivalently(self):
        backend = ABC(os.environ["ABC_BINARY"], ROOT / "examples/teaching.genlib")
        with tempfile.TemporaryDirectory(prefix="aig test space ") as directory:
            work = Path(directory)
            original = backend.prepare(ROOT / "examples/chain-4.blif", work / "input.aig")
            graph = Graph.read(original)
            for i, action in enumerate(actions(10)):
                child = backend.apply(original, action, work / f"child-{i}.aig")
                self.assertTrue(backend.verify(original, child))
                candidate = Graph.read(child)
                for bits in itertools.product((False, True), repeat=7):
                    self.assertEqual(graph.evaluate(bits), candidate.evaluate(bits))
                mapped = backend.map(child)
                self.assertTrue(backend.verify(original, mapped["netlist"]))
                self.assertGreater(mapped["area"], 0)
                self.assertGreater(mapped["delay"], 0)

    def test_mapping_library_with_spaces_ignores_adjacent_supergate_files(self):
        with tempfile.TemporaryDirectory(prefix="library with spaces ") as directory:
            work = Path(directory)
            library = work / 'cells with spaces.genlib'
            library.write_bytes((ROOT/'examples/teaching.genlib').read_bytes())
            extra = library.with_suffix('.super');extra.write_text('invalid stale supergate data')
            before = {p.name:p.read_bytes() for p in work.iterdir()}
            backend = ABC(os.environ['ABC_BINARY'], library)
            state = backend.prepare(ROOT/'examples/chain-4.blif',work/'input.aig')
            mapped = backend.map(state)
            self.assertTrue(backend.verify(state,mapped['netlist']))
            self.assertEqual(mapped['area'],7.3)
            self.assertEqual(mapped['delay'],5.5)
            for name, content in before.items():
                self.assertEqual((work/name).read_bytes(),content)

    def test_command_errors_and_failed_equivalence_cannot_succeed(self):
        backend = ABC(os.environ["ABC_BINARY"], ROOT / "examples/teaching.genlib")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            state = backend.prepare(ROOT / "examples/chain-4.blif", work / "input.aig")
            with self.assertRaises(RuntimeError):
                backend.apply(state, Action("bad", "nonexistent_command"), work / "bad.aig")
            wrong = work / "wrong.blif"
            wrong.write_text(".model wrong\n.inputs a b c d0 d1 d2 d3\n.outputs F\n.names a F\n1 1\n.end\n")
            with self.assertRaises(RuntimeError):
                backend.verify(state, wrong)

    def test_repeated_mapping_keeps_previous_netlists(self):
        backend = ABC(os.environ["ABC_BINARY"], ROOT / "examples/teaching.genlib")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            state = backend.prepare(ROOT / "examples/chain-4.blif", work / "input.aig")
            first, second = backend.map(state), backend.map(state)
            self.assertNotEqual(first["netlist"], second["netlist"])
            self.assertEqual(first["netlist"].read_bytes(), second["netlist"].read_bytes())

    def test_sequential_input_fails_without_export(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            source = work / "sequential.blif"
            source.write_text(".model sequential\n.inputs a\n.outputs q\n.latch a q 0\n.end\n")
            result = subprocess.run([sys.executable, "-m", "aigopt", "optimize", "--abc",
                os.environ["ABC_BINARY"], "--library", str(ROOT / "examples/teaching.genlib"),
                "--model", str(ROOT / "examples/evaluator.json"), "--input", str(source),
                "--output", str(work / "result")], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Sequential AIGs", result.stderr)
            self.assertFalse((work / "result").exists())

    def test_cli_collect_train_search_and_library_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            def run(*args, success=True):
                result = subprocess.run([sys.executable, "-m", "aigopt", *map(str, args)],
                                        cwd=ROOT, capture_output=True, text=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            common = ["--abc", os.environ["ABC_BINARY"], "--library", ROOT / "examples/teaching.genlib"]
            run("collect", *common, "--samples", 3, "--output", work / "labels.json",
                *sorted((ROOT / "examples").glob("chain-[234].blif")))
            run("train", "--dataset", work / "labels.json", "--epochs", 100,
                "--output", work / "model.json")
            run("optimize", *common, "--model", work / "model.json",
                "--input", ROOT / "examples/chain-4.blif", "--steps", 2,
                "--iterations", 12, "--output", work / "result")
            report = json.loads((work / "result/trace.json").read_text())
            self.assertTrue(report["equivalence_proven"])
            self.assertGreaterEqual(report["mapped_reward"], 0)
            self.assertEqual(report["mapping_calls"], 2)
            self.assertEqual(report["search_mapping_calls"], 0)
            self.assertEqual(len(report["decisions"]), 2)
            changed = work / "changed.genlib"
            changed.write_text((ROOT / "examples/teaching.genlib").read_text() + "\n# changed\n")
            result = run("optimize", "--abc", os.environ["ABC_BINARY"], "--library", changed,
                         "--model", work / "model.json", "--input", ROOT / "examples/chain-4.blif",
                         "--output", work / "bad-result", success=False)
            self.assertIn("different cell library", result.stderr)
            self.assertFalse((work / "bad-result").exists())


if __name__ == "__main__":
    unittest.main()
