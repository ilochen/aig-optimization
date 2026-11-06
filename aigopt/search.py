"""Budgeted UCT with one expansion/evaluation and path-only backup per trial."""
from dataclasses import dataclass, field
import math
import random
import subprocess
from pathlib import Path
from .abc import ABCError
from .objective import Objective
from .graph import Graph


@dataclass
class Node:
    state: object
    parent: object = None
    action: object = None
    depth: int = 0
    visits: int = 0
    total: float = 0.
    children: list = field(default_factory=list)
    id: int = 0
    features: list = field(default_factory=list)
    prediction: dict = field(default_factory=dict)
    graph_sha256: str = ""
    remaining: list = field(default_factory=list)
    error: str = None
    evaluator_features: list = field(default_factory=list)

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.


def search(backend, initial, evaluator, action_space, workdir, iterations=16,
           max_depth=3, exploration=.8, baseline=None, *, seed=0, randomize=True,
           choice="root", record_frames=True, widening=None, objective=None):
    objective = objective or Objective()
    reward = objective.reward
    if iterations < 1 or max_depth < 1 or not action_space or exploration < 0 or not math.isfinite(exploration):
        raise ValueError("Search requires positive budgets, actions, and nonnegative exploration")
    if choice not in ("root", "best"):
        raise ValueError("Choose a root action or the best evaluated candidate")
    if widening is not None and (not math.isfinite(widening) or widening <= 0):
        raise ValueError("Progressive widening coefficient must be finite and positive")
    if hasattr(evaluator,'for_reference'):
        evaluator=evaluator.for_reference(Graph.read(initial))
    rng = random.Random(seed)
    cache, evaluations, cache_hits = {}, 0, 0

    def initialize(node):
        nonlocal evaluations, cache_hits
        graph = Graph.read(node.state)
        node.features, node.graph_sha256 = graph.features(), graph.digest()
        node.remaining = list(action_space)
        if randomize:
            rng.shuffle(node.remaining)
        if node.graph_sha256 in cache:
            cache_hits += 1
            node.prediction, node.evaluator_features = cache[node.graph_sha256]
        else:
            node.evaluator_features = evaluator.encode(graph) if hasattr(evaluator, "encode") else node.features
            node.prediction = evaluator.predict(node.evaluator_features) if hasattr(evaluator, "encode") else evaluator.evaluate(node.state, graph) if hasattr(evaluator, "evaluate") else evaluator.predict(node.features)
            cache[node.graph_sha256] = (node.prediction, node.evaluator_features)
            evaluations += 1

    root = Node(initial)
    initialize(root)
    nodes, frames, failures = [root], [], []
    baseline = baseline or root.prediction

    def snapshot(phase, path, value=None):
        if not record_frames:
            return
        frames.append({"phase": phase, "path": [n.id for n in path], "reward": value,
                       "nodes": [{"id": n.id, "parent": n.parent.id if n.parent else None,
                                  "depth": n.depth, "action": n.action.name if n.action else None,
                                  "command": n.action.command if n.action else None,
                                  "N": n.visits, "W": n.total, "Q": n.mean,
                                  "features": n.features, "prediction": n.prediction,
                                  "graph_sha256": n.graph_sha256, "error": n.error,
                                  "evaluator_features": n.evaluator_features} for n in nodes]})

    for _ in range(iterations):
        node, path = root, [root]
        def expansion_limit(n):
            return len(action_space) if widening is None else min(
                len(action_space), max(1, math.ceil(widening * math.sqrt(n.visits + 1))))
        while node.depth < max_depth:
            eligible = [child for child in node.children if child.error is None]
            if node.remaining and (len(node.children) < expansion_limit(node) or not eligible):
                break
            if not eligible:
                break
            parent = node
            node = max(eligible, key=lambda child: child.mean + exploration * math.sqrt(math.log(parent.visits) / child.visits))
            path.append(node)
        snapshot("Select", path)
        if node.depth < max_depth and node.remaining:
            action = node.remaining.pop(0)
            child = Node(None, node, action, node.depth + 1, id=len(nodes))
            try:
                child.state = backend.apply(node.state, action, workdir / f"node-{len(nodes)}.aig")
                initialize(child)
            except (ABCError, subprocess.TimeoutExpired) as error:
                child.error = f"{type(error).__name__}: {str(error)[-1500:]}"
                child.remaining = []
                failures.append({"node": child.id, "command": action.command,
                                 "timeout": isinstance(error, subprocess.TimeoutExpired), "error": child.error})
            node.children.append(child)
            nodes.append(child)
            node = child
            path.append(node)
        snapshot("Expand", path)
        value = -1. if node.error else reward(node.prediction, baseline)
        snapshot("Evaluate", path, value)
        for ancestor in path:
            ancestor.visits += 1
            ancestor.total += value
        snapshot("Backpropagate", path, value)
    valid_root_children = [child for child in root.children if child.error is None]
    chosen = max(valid_root_children, key=lambda child: (child.mean, child.visits)) if choice == "root" and valid_root_children else max(
        [n for n in nodes if n.error is None], key=lambda child: (reward(child.prediction, baseline), -child.depth, -child.id))
    path = []
    cursor = chosen
    while cursor is not None:
        path.append(cursor)
        cursor = cursor.parent
    path.reverse()
    snapshot("Decision", path, reward(chosen.prediction, baseline))
    return chosen, {"iterations": iterations, "max_depth": max_depth,
                    "exploration": exploration, "baseline_prediction": baseline,
                    "objective": objective.payload(),
                    "evaluator_context": evaluator.context() if hasattr(evaluator,"context") else None,
                    "chosen_node": chosen.id, "frames": frames, "choice": choice,
                    "seed": seed, "randomized_expansion": randomize,
                    "widening": widening, "root_actions_tried": len(root.children),
                    "failed_actions": failures, "failure_penalty": -1.,
                    "transformations": len(nodes) - 1, "evaluations": evaluations,
                    "evaluation_cache_hits": cache_hits,
                    "successful_candidates": sum(n.error is None for n in nodes) - 1,
                    "visited_max_depth": max(n.depth for n in nodes if n.error is None),
                    "chosen_prediction": chosen.prediction,
                    "candidates": [{"id": n.id, "parent": n.parent.id if n.parent else None,
                        "command": n.action.command if n.action else None, "depth": n.depth,
                        "state_file": Path(n.state).name if n.state else None,
                        "graph_sha256": n.graph_sha256, "prediction": n.prediction,
                        "features": n.features, "evaluator_features": n.evaluator_features,
                        "error": n.error} for n in nodes],
                    "plan": [n.action.command for n in path if n.action]}
