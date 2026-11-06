"""Residual neural cost model trained on within-circuit log-ADP differences."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from .descriptors import VERSION, NAMES, describe
from .evaluator import positive_metrics


def prior(features):
    return [math.log(max(1., math.expm1(features[i]))) for i in (0, 1)]


class RankEvaluator:
    def __init__(self, payload):
        from . import load_descriptors
        schemas={VERSION:(NAMES,describe),load_descriptors.VERSION:(load_descriptors.NAMES,load_descriptors.describe)}
        schema=schemas.get(payload.get('descriptor_version'))
        if schema is None or payload.get('descriptor_names')!=schema[0]:
            raise ValueError("Ranking checkpoint descriptor mismatch")
        self.names,self.describe=schema
        size, hidden = len(self.names), payload["hidden"]
        active=payload.get('active_indices',list(range(size)))
        if not active or len(set(active))!=len(active) or any(type(i) is not int or not 0<=i<size for i in active):
            raise ValueError('Invalid active descriptor indices')
        if hidden < 1 or len(payload["W"]) != hidden or any(len(w) != size + 1 for w in payload["W"]) or len(payload["V"]) != 2 or any(len(v) != hidden + 1 for v in payload["V"]):
            raise ValueError("Ranking checkpoint shape mismatch")
        if len(payload["mean"]) != size or len(payload["scale"]) != size or any(s <= 0 or not math.isfinite(s) for s in payload["scale"]):
            raise ValueError("Invalid ranking checkpoint normalization")
        if not 0 <= payload.get("blend", 1.) <= 1:
            raise ValueError("Invalid ranking residual blend")
        if any(not math.isfinite(v) for row in payload["W"] + payload["V"] for v in row) or any(not math.isfinite(v) for v in payload["mean"]):
            raise ValueError("Ranking checkpoint must be finite")
        self.payload = payload

    def inputs(self, features):
        if len(features) != len(self.names) or any(not math.isfinite(v) for v in features):
            raise ValueError(f"Expected {len(self.names)} finite mapping descriptors")
        active=set(self.payload.get('active_indices',range(len(self.names))))
        return [max(-5., min(5., (v - m) / s)) if i in active else 0. for i,(v,m,s) in enumerate(zip(features,self.payload['mean'],self.payload['scale']))]+[1.]

    def forward(self, x):
        W, V = self.payload["W"], self.payload["V"]
        h = [math.tanh(sum(a*b for a, b in zip(w, x))) for w in W]
        return h, [sum(a*b for a, b in zip(v, h+[1.])) for v in V]

    def predict(self, features):
        x = self.inputs(features)
        blend = self.payload.get("blend", 1.)
        if blend == 0:
            # A disabled learned residual is the exact structural control,
            # including its tie behavior; exp(log(n)) roundoff must not act
            # as an accidental ranking signal.
            return {k: float(max(1, round(math.expm1(features[i])))) for k, i in (("area", 0), ("delay", 1))}
        _, values = self.forward(x)
        base = prior(features)
        return positive_metrics(dict(zip(("area", "delay"), (
            math.exp(max(-20, min(30, b + blend * v))) for b, v in zip(base, values)))))

    def evaluate(self, state, graph):
        return self.predict(self.describe(graph))

    def encode(self, graph):
        return self.describe(graph)

    def domain_graph(self, graph):
        values = self.describe(graph)
        outside = [name for name, value, bounds in zip(self.names, values, self.payload["ranges"]) if not bounds[0] <= value <= bounds[1]]
        return {"known": True, "outside_features": outside}

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload, indent=2, allow_nan=False) + "\n")

    def update(self, x, hidden, error, rate):
        W, V = self.payload["W"], self.payload["V"]
        dh = [(1 - h*h) * sum(error[k] * V[k][j] for k in range(2)) for j, h in enumerate(hidden)]
        for k in range(2):
            for j, activation in enumerate(hidden+[1.]):
                V[k][j] -= rate * error[k] * activation
        for j in range(len(W)):
            for i, value in enumerate(x):
                W[j][i] -= rate * dh[j] * value

    def update_pair(self, xa, xb, base_difference, target_difference, rate, weight):
        ha, pa = self.forward(xa)
        hb, pb = self.forward(xb)
        error = sum(pa) - sum(pb) + base_difference - target_difference
        derivative = weight * max(-.1, min(.1, error))
        original_v = [row[:] for row in self.payload["V"]]
        self.update(xa, ha, [derivative]*2, rate)
        updated_v = self.payload["V"]
        self.payload["V"] = [row[:] for row in original_v]
        self.update(xb, hb, [-derivative]*2, rate)
        second_v = self.payload["V"]
        self.payload["V"] = [[u+s-o for u,s,o in zip(ur,sr,orr)] for ur,sr,orr in zip(updated_v,second_v,original_v)]
        return error

    @classmethod
    def train(cls, rows, library_sha256, hidden=16, epochs=600, pair_weight=1., seed=271828, weight_decay=0.):
        if epochs < 1 or hidden < 1 or pair_weight < 0 or len(rows) < 2 or not 0 <= weight_decay < 1:
            raise ValueError("Invalid ranking training configuration")
        rng = random.Random(seed)
        size = len(NAMES)
        for row in rows:
            positive_metrics(row)
            if len(row["rank_features"]) != size or any(not math.isfinite(x) for x in row["rank_features"]):
                raise ValueError("Invalid training descriptors")
        mean = [statistics.mean(r["rank_features"][i] for r in rows) for i in range(size)]
        scale = [max(.03, statistics.pstdev(r["rank_features"][i] for r in rows)) for i in range(size)]
        model = cls({"kind": "rank_mlp_v1", "descriptor_version": VERSION, "descriptor_names": NAMES,
            "library_sha256": library_sha256, "hidden": hidden, "blend": 1., "mean": mean, "scale": scale,
            "ranges": [[min(r["rank_features"][i] for r in rows), max(r["rank_features"][i] for r in rows)] for i in range(size)],
            "W": [[rng.uniform(-.15, .15) for _ in range(size+1)] for _ in range(hidden)],
            "V": [[rng.uniform(-.01, .01) for _ in range(hidden+1)] for _ in range(2)],
            "seed": seed, "epochs": epochs, "pair_weight": pair_weight,
            "weight_decay_per_epoch": weight_decay,
            "objective": "within-circuit log-ADP difference Huber + 0.1 residual metric MSE"})
        examples, groups = [], defaultdict(list)
        for row in rows:
            features = row["rank_features"]
            base = prior(features)
            y = [math.log(row[k]) - b for k, b in zip(("area", "delay"), base)]
            x = model.inputs(features)
            groups[row["circuit"]].append(len(examples))
            examples.append((x, y, sum(base), math.log(row["area"]*row["delay"])))
        # Equal pair budget per circuit avoids weighting large candidate pools
        # quadratically. Include structural mis-rankings, not just large wins.
        pairs = []
        for indices in groups.values():
            candidates = [(a, b) for n, a in enumerate(indices) for b in indices[n+1:]
                          if abs(examples[a][3]-examples[b][3]) >= .005]
            rng.shuffle(candidates)
            pairs.extend(candidates[:80])
        def pair_loss():
            if not pairs:
                return 0.
            predictions = [sum(model.forward(x)[1])+base for x, _, base, _ in examples]
            return statistics.mean(((predictions[a]-predictions[b]) - (examples[a][3]-examples[b][3]))**2 for a,b in pairs)
        before = pair_loss()
        order = list(range(len(examples)))
        for epoch in range(epochs):
            rate = .015 / (1 + epoch/250)
            rng.shuffle(order)
            for i in order:
                x, target, _, _ = examples[i]
                h, prediction = model.forward(x)
                model.update(x, h, [.1 * max(-1, min(1, a-b)) for a,b in zip(prediction,target)], rate)
            if pair_weight:
                rng.shuffle(pairs)
                for a,b in pairs:
                    xa, _, ba, ta = examples[a]
                    xb, _, bb, tb = examples[b]
                    model.update_pair(xa, xb, ba-bb, ta-tb, rate, pair_weight)
            if weight_decay:
                for matrix in (model.payload["W"], model.payload["V"]):
                    for row in matrix:
                        for i in range(len(row)-1):
                            row[i] *= 1-weight_decay
        model.payload["training_report"] = {"train_examples": len(rows), "training_pairs": len(pairs),
            "train_circuits": sorted(groups), "validation_circuits": [],
            "initial_pair_mse": before, "final_pair_mse": pair_loss()}
        return model
