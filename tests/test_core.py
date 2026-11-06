import itertools
import json
import math
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from aigopt.abc import Action, actions, quote
from aigopt.evaluator import Evaluator, reward
from aigopt.graph import Graph
from aigopt.search import search

ROOT = Path(__file__).resolve().parents[1]


class GraphTests(unittest.TestCase):
    def test_binary_matches_ascii_and_complemented_constant_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.aig"
            # (a AND !b) OR a: output negation and constants exercise literal polarity.
            path.write_bytes(b"aig 4 2 0 2 2\n9\n1\n" + bytes([1, 3, 2, 1]))
            binary = Graph.read(path)
            path.write_text("aag 4 2 0 2 2\n2\n4\n9\n1\n6 5 2\n8 6 5\n")
            self.assertEqual(binary, Graph.read(path))
            for a, b in itertools.product((False, True), repeat=2):
                self.assertEqual(binary.evaluate([a, b]), (not ((not b) and a), True))
            self.assertEqual(binary.features(), [2, 2, 2, 2])

    def test_rejects_truncated_sequential_and_forward_references(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.aig"
            for data in (b"aig 2 1 0 1 1\n4\n", b"aag 2 1 1 1 0\n", b"aag 3 1 0 1 1\n2\n6\n6 4 2\n",
                         b"aag -1 0 0 0 0\n", b"aag 1 1 0 1 1\n2\n4\n4 2 2\n"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    Graph.read(path)

    def test_empty_and_constant_graph_has_finite_features(self):
        self.assertEqual(Graph((), (1,), ()).features(), [0, 0, 0, 0])

    def test_graph_identity_ignores_comments_but_keeps_polarity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.aag"
            content = "aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n"
            path.write_text(content + "c\nfirst comment\n")
            first = Graph.read(path).digest()
            path.write_text(content + "i0 changed_name\nc\nnew comment\n")
            self.assertEqual(first, Graph.read(path).digest())
            path.write_text(content.replace("6 4 2", "6 5 2"))
            self.assertNotEqual(first, Graph.read(path).digest())


class EvaluatorTests(unittest.TestCase):
    def test_scaled_costs_do_not_saturate_to_one_absolute_size(self):
        model = Evaluator.load(ROOT / "examples/evaluator.json")
        payload = dict(model.payload, W=[[0.] * 5 for _ in range(8)],
                       V=[[0.] * 9 for _ in range(2)], output_transform="scaled_exp",
                       training_feature_ranges=[[1, 10], [1, 10], [0, 10], [1, 5]])
        scaled = Evaluator(payload)
        self.assertEqual(scaled.predict([5000, 200, 1000, 40]), {"area": 5000., "delay": 200.})
        self.assertEqual(len(scaled.domain([5000, 200, 1000, 40])["outside_features"]), 4)
        for key, value in (("scale", [1, 1, 1, 0]), ("mean", [0, 0, 0, math.nan]),
                           ("output_transform", "bad")):
            with self.assertRaises(ValueError):
                Evaluator(dict(payload, **{key: value}))

    def test_learns_positive_costs_and_round_trips(self):
        rows = [{"circuit": f"c{w}", "features": [w, w // 2, w % 3, 2],
                 "area": w * 1.3, "delay": w * .7} for w in range(2, 18)]
        model = Evaluator.train(rows, "library", epochs=500)
        report = model.payload["training_report"]
        self.assertLess(report["final_log_mse"], report["initial_log_mse"] * .05)
        self.assertGreater(report["validation_examples"], 0)
        self.assertTrue(set(report["validation_circuits"]).isdisjoint(
            r["circuit"] for r in rows if r["circuit"] not in report["validation_circuits"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = Evaluator.load(path)
            for row in rows:
                self.assertEqual(model.predict(row["features"]), loaded.predict(row["features"]))
                self.assertTrue(all(math.isfinite(v) and v > 0 for v in loaded.predict(row["features"]).values()))

    def test_reward_is_relative_to_fixed_area_delay_baseline(self):
        self.assertAlmostEqual(reward({"area": 8, "delay": 5}, {"area": 10, "delay": 10}), .6)
        self.assertLess(reward({"area": 8, "delay": 20}, {"area": 10, "delay": 10}), 0)
        with self.assertRaises(ValueError):
            reward({"area": 0, "delay": 1}, {"area": 1, "delay": 1})


class SearchTests(unittest.TestCase):
    def test_timeouts_are_rejected_and_never_selected_for_export(self):
        class Backend:
            def __init__(self, fail_all):
                self.fail_all = fail_all
            def apply(self, source, action, destination):
                if self.fail_all or action.name == "slow":
                    raise subprocess.TimeoutExpired(action.command, 1)
                destination.write_bytes(Path(source).read_bytes())
                return destination
        class Predictor:
            def predict(self, _):
                return {"area": 1., "delay": 1.}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            initial = work / "input.aig"
            initial.write_text("aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n")
            for fail_all in (False, True):
                trial = work / str(fail_all)
                trial.mkdir()
                chosen, trace = search(Backend(fail_all), initial, Predictor(),
                    [Action("slow", "slow"), Action("fast", "fast")], trial,
                    iterations=12, randomize=False)
                self.assertIsNone(chosen.error)
                self.assertTrue(chosen.state.is_file())
                self.assertTrue(all(f["timeout"] for f in trace["failed_actions"]))
                self.assertEqual(trace["frames"][-1]["nodes"][0]["N"], 12)
                if fail_all:
                    self.assertEqual(chosen.id, 0)
                    self.assertEqual(trace["visited_max_depth"], 0)

    def test_best_candidate_preserves_a_useful_sequence_after_uphill_move(self):
        class Backend:
            def apply(self, source, action, destination):
                count = Graph.read(source).features()[0]
                count = 3 if action.name == "uphill" else (1 if count == 3 else count)
                lines = [f"aag {count + 1} 1 0 1 {count}", "2", str(2 * (count + 1))]
                lines += [f"{2 * (i + 2)} {2 * (i + 1)} 2" for i in range(count)]
                destination.write_text("\n".join(lines) + "\n")
                return destination
        class Predictor:
            def predict(self, features):
                return {"area": features[0], "delay": 1}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            initial = work / "input.aig"
            initial.write_text("aag 3 1 0 1 2\n2\n6\n4 2 2\n6 4 2\n")
            chosen, trace = search(Backend(), initial, Predictor(),
                [Action("uphill", "uphill"), Action("finish", "finish")], work,
                iterations=12, max_depth=2, randomize=False, choice="best")
            self.assertEqual(chosen.features[0], 1)
            self.assertEqual(trace["plan"], ["uphill", "finish"])
            self.assertGreater(chosen.parent.features[0], 2)

    def test_duplicate_graphs_reuse_predictions_and_compact_run_is_identical(self):
        class Backend:
            def apply(self, source, action, destination):
                destination.write_bytes(Path(source).read_bytes())
                return destination
        class Predictor:
            calls = 0
            def predict(self, _):
                self.calls += 1
                return {"area": 1., "delay": 1.}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            initial = work / "input.aig"
            initial.write_text("aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n")
            outcomes = []
            for compact in (False, True):
                trial = work / str(compact)
                trial.mkdir()
                predictor = Predictor()
                node, trace = search(Backend(), initial, predictor, actions(7), trial,
                    iterations=20, seed=71, choice="best", record_frames=not compact)
                self.assertEqual(predictor.calls, 1)
                self.assertEqual(trace["evaluation_cache_hits"], trace["transformations"])
                self.assertEqual(node.id, 0)  # Ties retain the unchanged input.
                outcomes.append((node.id, trace["plan"]))
                if compact:
                    self.assertEqual(trace["frames"], [])
            self.assertEqual(outcomes[0], outcomes[1])
            widened = work / "widened"
            widened.mkdir()
            _, trace = search(Backend(), initial, Predictor(), actions(7), widened,
                iterations=24, max_depth=3, seed=71, choice="best", widening=1., record_frames=False)
            self.assertEqual(trace["visited_max_depth"], 3)
            self.assertLess(trace["root_actions_tried"], len(actions(7)))

    def test_optimistic_surrogate_cannot_export_a_worse_mapped_candidate(self):
        from aigopt.__main__ import optimize
        class Backend:
            library_sha256 = "library"
            mapping_profile = {}
            def validate_profile(self, payload):pass
            mapping_calls = 0
            def prepare(self, source, destination):
                destination.write_text("aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n")
                return destination
            def verify(self, *args):
                return True
            def map(self, state):
                self.mapping_calls += 1
                netlist = state.with_suffix(".blif")
                netlist.write_text("baseline" if self.mapping_calls == 1 else "worse candidate")
                return {"area": 10 if self.mapping_calls == 1 else 20, "delay": 10, "netlist": netlist}
        class Predictor:
            payload = {"library_sha256": "library"}
            def predict(self, features):
                return {"area": 1, "delay": 1}
        def optimistic_search(backend, state, *args, **kwargs):
            candidate = state.parent / "candidate.aig"
            candidate.write_bytes(state.read_bytes())
            return SimpleNamespace(state=candidate), {"predicted_reward": .99}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            model = work / "model.json"
            model.write_text("{}")
            source = work / "source.blif"
            source.write_text("input")
            args = SimpleNamespace(abc="unused", library="unused", model=model,
                output=work / "result", input=source, steps=1, seed=0, iterations=16,
                depth=3, exploration=.8)
            with patch("aigopt.__main__.ABC", return_value=Backend()), patch(
                    "aigopt.__main__.Evaluator.load", return_value=Predictor()), patch(
                    "aigopt.__main__.search", side_effect=optimistic_search), patch("aigopt.__main__.write_bundle"), patch("builtins.print"):
                optimize(args)
            report = json.loads((work / "result/trace.json").read_text())
            self.assertFalse(report["candidate_accepted"])
            self.assertEqual(report["mapped_reward"], 0)
            self.assertEqual((work / "result/final.mapped.blif").read_text(), "baseline")

    def test_uct_backup_depth_and_frozen_actions(self):
        class Backend:
            def apply(self, source, action, destination):
                destination.write_bytes(Path(source).read_bytes())
                return destination
        class Predictor:
            def predict(self, _):
                return {"area": 2, "delay": 3}
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            initial = work / "input.aig"
            initial.write_text("aag 3 2 0 1 1\n2\n4\n6\n6 4 2\n")
            chosen, trace = search(Backend(), initial, Predictor(), actions(42), work,
                                   iterations=40, max_depth=2,
                                   baseline={"area": 4, "delay": 3})
        frames = trace["frames"]
        self.assertEqual(frames[-1]["nodes"][0]["N"], 40)
        self.assertLessEqual(max(n["depth"] for n in frames[-1]["nodes"]), 2)
        for i in range(0, 160, 4):
            selection, expansion, evaluation, backup = frames[i:i+4]
            for node, after in zip(evaluation["nodes"], backup["nodes"]):
                increment = node["id"] in backup["path"]
                self.assertEqual(after["N"], node["N"] + increment)
                self.assertAlmostEqual(after["W"], node["W"] + .5 * increment)
            for parent_id, child_id in zip(selection["path"], selection["path"][1:]):
                parent = selection["nodes"][parent_id]
                children = [n for n in selection["nodes"] if n["parent"] == parent_id]
                score = lambda n: n["Q"] + .8 * math.sqrt(math.log(parent["N"]) / n["N"])
                self.assertAlmostEqual(score(selection["nodes"][child_id]), max(map(score, children)))
        roots = [n for n in frames[-1]["nodes"] if n["parent"] == 0]
        self.assertEqual(chosen.mean, max(n["Q"] for n in roots))
        self.assertEqual(actions(42), actions(42))
        self.assertTrue(actions(42)[-1].command.startswith("&get;"))

    def test_paths_cannot_inject_abc_commands(self):
        with self.assertRaises(ValueError):
            quote("bad;quit.aig")


if __name__ == "__main__":
    unittest.main()
