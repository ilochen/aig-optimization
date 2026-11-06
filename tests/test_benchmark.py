import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from aigopt.abc import fingerprint
from aigopt.benchmark import assert_disjoint, summarize
from aigopt.probes import ranking_metrics


class BenchmarkTests(unittest.TestCase):
    def test_overlap_check_includes_training_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "circuit"
            path.write_text("reserved test design")
            digest = fingerprint(path)
            for report in ({"train_circuits": [digest], "validation_circuits": []},
                           {"train_circuits": [], "validation_circuits": [digest]}):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    assert_disjoint(SimpleNamespace(payload={"training_report": report}), [path])
            with self.assertRaisesRegex(ValueError, "hash-matching dataset"):
                assert_disjoint(SimpleNamespace(payload={"training_report": {}}), [path])

    def test_summary_keeps_raw_regressions_and_circuit_weighting(self):
        def row(circuit, raw):
            return {"circuit": circuit, "method": "example", "raw_reward": raw,
                    "accepted_reward": max(0, raw), "fallback": raw < 0,
                    "search_seconds": 1, "total_seconds": 2, "transformations": 24,
                    "eda_commands": 30, "mapping_calls": 2}
        result = summarize([row("a", .2), row("a", .2), row("b", -.1),
                            {"circuit": "c", "method": "example", "seed": 1, "error": "failed", "total_seconds": 3}])
        summary = result["methods"]["example"]
        self.assertAlmostEqual(summary["mean_raw_reward"], .05)
        self.assertAlmostEqual(summary["mean_accepted_reward"], .2 / 3)
        self.assertEqual(summary["failure_rate"], .25)
        self.assertEqual(len(result["failures"]), 1)

    def test_ranking_ties_do_not_look_like_correct_predictions(self):
        tied = [{"area": a, "delay": 1, "prediction": {"area": 5, "delay": 1}} for a in (1, 2, 3)]
        metrics = ranking_metrics(tied)
        self.assertEqual(metrics["pair_accuracy"], .5)
        self.assertEqual(metrics["predicted_tied_pairs"], 3)
        for row in tied:
            row["prediction"]["area"] = row["area"]
        self.assertEqual(ranking_metrics(tied)["pair_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
