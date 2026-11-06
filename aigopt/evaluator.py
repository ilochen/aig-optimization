"""A dependency-free 4 → 8 → 2 feature MLP with portable JSON weights."""
import json
import math
from pathlib import Path
import random
from .graph import FEATURE_NAMES, FEATURE_VERSION


def positive_metrics(metrics):
    if any(not math.isfinite(metrics[k]) or metrics[k] <= 0 for k in ("area", "delay")):
        raise ValueError("Area and delay must be finite and positive")
    return metrics


def reward(metrics, baseline):
    positive_metrics(metrics)
    positive_metrics(baseline)
    return 1 - metrics["area"] * metrics["delay"] / (baseline["area"] * baseline["delay"])


class Evaluator:
    def __init__(self, payload):
        if payload["feature_version"] != FEATURE_VERSION or payload["feature_names"] != FEATURE_NAMES:
            raise ValueError("Checkpoint feature schema mismatch")
        if len(payload["W"]) != 8 or any(len(w) != 5 for w in payload["W"]) or len(payload["V"]) != 2 or any(len(v) != 9 for v in payload["V"]):
            raise ValueError("Expected a 4 → 8 → 2 checkpoint")
        if len(payload["mean"]) != 4 or len(payload["scale"]) != 4 or any(
                not math.isfinite(s) or s <= 0 for s in payload["scale"]):
            raise ValueError("Checkpoint has invalid input normalization")
        if any(not math.isfinite(v) for row in payload["W"] + payload["V"] for v in row) or any(
                not math.isfinite(v) for v in payload["mean"]):
            raise ValueError("Checkpoint weights and means must be finite")
        if payload.get("output_transform", "exp") not in ("exp", "scaled_exp"):
            raise ValueError("Unknown checkpoint output transform")
        self.payload = payload

    def predict(self, features):
        p = self.payload
        x = self._inputs(features, p["mean"], p["scale"])
        _, y = self._forward(x, p["W"], p["V"])
        scales = self._cost_scales(features) if p.get("output_transform") == "scaled_exp" else (1, 1)
        return positive_metrics(dict(zip(("area", "delay"),
            (s * math.exp(max(-20, min(20, v))) for v, s in zip(y, scales)))))

    @staticmethod
    def _cost_scales(features):
        # Learn cost per AND and per logic level; preserve size scaling outside
        # the training range instead of forcing all large graphs to one value.
        return max(1, features[0]), max(1, features[1])

    def domain(self, features):
        self._inputs(features, self.payload["mean"], self.payload["scale"])
        ranges = self.payload.get("training_feature_ranges")
        outside = [] if not ranges else [name for name, f, (low, high) in zip(
            FEATURE_NAMES, features, ranges) if f < low or f > high]
        return {"known": ranges is not None, "outside_features": outside}

    @staticmethod
    def _inputs(features, mean, scale):
        if len(features) != 4 or any(not math.isfinite(f) or f < 0 for f in features):
            raise ValueError("Expected four finite nonnegative graph features")
        return [(math.log1p(f) - m) / s for f, m, s in zip(features, mean, scale)] + [1]

    @staticmethod
    def _forward(x, W, V):
        hidden = [math.tanh(sum(a * b for a, b in zip(w, x))) for w in W]
        y = [sum(a * b for a, b in zip(v, hidden + [1])) for v in V]
        return hidden, y

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get('kind') == 'hybrid_reference_mlp_v1':
            from .hybrid_evaluator import HybridEvaluator
            return HybridEvaluator(payload)
        if payload.get('kind') == 'reference_rank_mlp_v1':
            from .relative_evaluator import RelativeEvaluator
            return RelativeEvaluator(payload)
        if payload.get("kind") == "rank_mlp_v1":
            from .ranker import RankEvaluator
            return RankEvaluator(payload)
        return cls(payload)

    @classmethod
    def train(cls, rows, library_sha256, epochs=1500, seed=314159, scaled=True):
        if epochs < 1:
            raise ValueError("Epochs must be positive")
        if len(rows) < 2:
            raise ValueError("At least two mapping examples are required")
        for r in rows:
            cls._inputs(r["features"], [0] * 4, [1] * 4)
            positive_metrics(r)
        # Split by original circuit, never by transformed state of that circuit.
        groups = sorted({row["circuit"] for row in rows})
        rng = random.Random(seed)
        rng.shuffle(groups)
        heldout = set(groups[:max(1, len(groups) // 5)]) if len(groups) >= 3 else set()
        training = [r for r in rows if r["circuit"] not in heldout]
        validation = [r for r in rows if r["circuit"] in heldout]
        logs = [[math.log1p(f) for f in r["features"]] for r in training]
        mean = [sum(x[i] for x in logs) / len(logs) for i in range(4)]
        scale = [max(.1, math.sqrt(sum((x[i] - mean[i]) ** 2 for x in logs) / len(logs))) for i in range(4)]
        W = [[rng.uniform(-.3, .3) for _ in range(5)] for _ in range(8)]
        V = [[rng.uniform(-.3, .3) for _ in range(9)] for _ in range(2)]
        examples = [(cls._inputs(r["features"], mean, scale),
                     [math.log(r[k] / s) for k, s in zip(("area", "delay"),
                      cls._cost_scales(r["features"]) if scaled else (1, 1))]) for r in training]

        def loss():
            return sum(sum((a - b) ** 2 for a, b in zip(cls._forward(x, W, V)[1], y))
                       for x, y in examples) / (2 * len(examples))

        before = loss()
        for epoch in range(epochs):
            rng.shuffle(examples)
            rate = .02 / (1 + epoch / 500)
            for x, target in examples:
                h, y = cls._forward(x, W, V)
                error = [max(-5, min(5, a - b)) for a, b in zip(y, target)]
                dh = [(1 - h[j] ** 2) * sum(error[k] * V[k][j] for k in range(2)) for j in range(8)]
                for k in range(2):
                    for j, activation in enumerate(h + [1]):
                        V[k][j] -= rate * error[k] * activation
                for j in range(8):
                    for i in range(5):
                        W[j][i] -= rate * dh[j] * x[i]
        model = cls({"feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
                     "library_sha256": library_sha256, "architecture": [4, 8, 2],
                     "input_transform": "standardized log1p", "output_transform": "scaled_exp" if scaled else "exp",
                     "training_feature_ranges": [[min(r["features"][i] for r in training),
                         max(r["features"][i] for r in training)] for i in range(4)],
                     "mean": mean, "scale": scale, "W": W, "V": V,
                     "seed": seed, "epochs": epochs})
        report = {"train_examples": len(training), "validation_examples": len(validation),
                  "train_circuits": sorted(set(groups) - heldout),
                  "validation_circuits": sorted(heldout), "initial_log_mse": before,
                  "final_log_mse": loss()}
        if validation:
            report["validation_mean_relative_error"] = {
                k: sum(abs(model.predict(r["features"])[k] / r[k] - 1) for r in validation) / len(validation)
                for k in ("area", "delay")}
        model.payload["training_report"] = report
        return model
