"""Size-stable topology descriptors for mapped-cost ranking (no EDA calls)."""
from collections import Counter, defaultdict
import math

VERSION = "aig-map-descriptors-v1"
NAMES = ["log_ands", "log_depth", "log_inputs", "log_outputs", "inverted_edge_fraction",
    "fanin_inv0", "fanin_inv1", "fanin_inv2", "shared_gate_fraction", "log_max_gate_fanout",
    "log_mean_gate_fanout", "log_max_input_fanout", "pi_edge_fraction", "dual_phase_gate_fraction",
    "inverted_output_fraction", "output_gate_ratio", "mean_level_ratio", "std_level_ratio",
    "critical_gate_fraction", "mean_output_level_ratio", "and3_opportunity_fraction",
    "reconvergent_sibling_fraction", "mean_level_imbalance", "late_gate_fraction",
    "single_fanout_fraction", "positive_only_fraction", "negative_only_fraction", "unused_gate_fraction"]


def describe(graph):
    base = graph.features()
    count, depth = max(1, base[0]), max(1, base[1])
    gates = {lhs // 2: (a, b) for lhs, a, b in graph.gates}
    levels = {0: 0, **{r // 2: 0 for r in graph.inputs}}
    fan, phases = Counter(), defaultdict(set)
    inv_hist, pi_edges, imbalance = Counter(), 0, 0.
    for lhs, a, b in graph.gates:
        key = lhs // 2
        levels[key] = 1 + max(levels[a // 2], levels[b // 2])
        inv_hist[(a & 1) + (b & 1)] += 1
        imbalance += abs(levels[a // 2] - levels[b // 2]) / depth
        for r in (a, b):
            fan[r // 2] += 1
            phases[r // 2].add(r & 1)
            pi_edges += r // 2 not in gates
    for r in graph.outputs:
        fan[r // 2] += 1
        phases[r // 2].add(r & 1)
    # Backward distance to any output identifies gates on longest output paths.
    remaining = {r // 2: 0 for r in graph.outputs}
    for lhs, a, b in reversed(graph.gates):
        key = lhs // 2
        if key in remaining:
            for r in (a, b):
                remaining[r // 2] = max(remaining.get(r // 2, -1), remaining[key] + 1)
    out_depth = max((levels[r // 2] for r in graph.outputs), default=0)
    critical = sum(levels[k] + remaining.get(k, -10**9) == out_depth for k in gates)
    merge, reconvergent = 0, 0
    for a, b in gates.values():
        merge += any(not (r & 1) and r // 2 in gates and fan[r // 2] == 1 for r in (a, b))
        if a // 2 in gates and b // 2 in gates:
            reconvergent += bool({r // 2 for r in gates[a // 2]} & {r // 2 for r in gates[b // 2]})
    gate_fans = [fan[k] for k in gates]
    mean_level = sum(levels[k] for k in gates) / count
    std_level = math.sqrt(sum((levels[k] - mean_level)**2 for k in gates) / count)
    outputs = max(1, len(graph.outputs))
    result = [math.log1p(base[0]), math.log1p(base[1]), math.log1p(len(graph.inputs)),
        math.log1p(len(graph.outputs)), base[2] / (2 * count),
        *(inv_hist[i] / count for i in range(3)), sum(f > 1 for f in gate_fans) / count,
        math.log1p(max(gate_fans, default=0)), math.log1p(sum(gate_fans) / count),
        math.log1p(max((fan[r // 2] for r in graph.inputs), default=0)), pi_edges / (2 * count),
        sum(len(phases[k]) == 2 for k in gates) / count,
        sum(r & 1 for r in graph.outputs) / outputs, len(graph.outputs) / count,
        mean_level / depth, std_level / depth, critical / count,
        sum(levels[r // 2] for r in graph.outputs) / (outputs * depth), merge / count,
        reconvergent / count, imbalance / count,
        sum(levels[k] >= .75 * depth for k in gates) / count,
        sum(f == 1 for f in gate_fans) / count,
        sum(phases[k] == {0} for k in gates) / count,
        sum(phases[k] == {1} for k in gates) / count,
        sum(f == 0 for f in gate_fans) / count]
    return result
