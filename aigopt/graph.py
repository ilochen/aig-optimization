"""Read combinational AIGER, including binary delta-encoded AND gates."""
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import json

FEATURE_NAMES = ["and_count", "logic_depth", "inverted_edges", "max_fanout"]
FEATURE_VERSION = "aiger-combinational-v1"


@dataclass(frozen=True)
class Graph:
    inputs: tuple
    outputs: tuple
    gates: tuple

    @classmethod
    def read(cls, path):
        stream = BytesIO(Path(path).read_bytes())
        header = stream.readline().decode("ascii").split()
        if len(header) != 6 or header[0] not in ("aig", "aag"):
            raise ValueError("Expected basic combinational AIGER (aig or aag)")
        maximum, inputs, latches, outputs, ands = map(int, header[1:])
        if min(maximum, inputs, latches, outputs, ands) < 0:
            raise ValueError("AIGER counts cannot be negative")
        if latches:
            raise ValueError("Sequential AIGs require a sequential evaluator")
        binary = header[0] == "aig"
        if binary and maximum != inputs + ands:
            raise ValueError("Noncanonical binary AIGER header")
        pis = tuple(2 * (i + 1) for i in range(inputs)) if binary else tuple(
            int(stream.readline()) for _ in range(inputs))
        pos = tuple(int(stream.readline()) for _ in range(outputs))

        def delta():
            value, shift = 0, 0
            while True:
                char = stream.read(1)
                if not char or shift > 63:
                    raise ValueError("Truncated or invalid AIGER delta")
                byte = char[0]
                value |= (byte & 127) << shift
                if byte < 128:
                    return value
                shift += 7

        gates = []
        for i in range(ands):
            if binary:
                lhs = 2 * (inputs + i + 1)
                rhs0 = lhs - delta()
                rhs1 = rhs0 - delta()
                gate = (lhs, rhs0, rhs1)
            else:
                gate = tuple(map(int, stream.readline().split()))
            if len(gate) != 3:
                raise ValueError("Invalid AND record")
            gates.append(gate)
        graph = cls(pis, pos, tuple(gates))
        graph.features()  # Validate topology before exposing a state.
        if any(r // 2 > maximum for r in (*pis, *pos)) or any(
                r // 2 > maximum for gate in gates for r in gate):
            raise ValueError("AIGER literal exceeds declared maximum")
        return graph

    def digest(self):
        """Exact graph identity, ignoring comments/symbols; not a Boolean hash."""
        content = json.dumps([self.inputs, self.outputs, self.gates], separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def features(self):
        levels = {0: 0}
        for literal in self.inputs:
            if literal <= 0 or literal % 2 or literal // 2 in levels:
                raise ValueError("Invalid primary input")
            levels[literal // 2] = 0
        fanout, inverted = Counter(), 0
        for lhs, a, b in self.gates:
            if lhs <= 0 or lhs % 2 or lhs // 2 in levels:
                raise ValueError("Invalid AND output")
            if any(r < 0 or r // 2 not in levels for r in (a, b)):
                raise ValueError("AIG gates must be topologically ordered")
            levels[lhs // 2] = 1 + max(levels[a // 2], levels[b // 2])
            fanout.update((a // 2, b // 2))
            inverted += (a & 1) + (b & 1)
        if any(r < 0 or r // 2 not in levels for r in self.outputs):
            raise ValueError("Invalid output literal")
        return [len(self.gates), max(levels.values()), inverted,
                max(fanout.values(), default=0)]

    def evaluate(self, bits):
        if len(bits) != len(self.inputs):
            raise ValueError("Wrong input count")
        values = {0: False, **{r // 2: bool(b) for r, b in zip(self.inputs, bits)}}
        def value(r):
            return values[r // 2] ^ bool(r & 1)
        for lhs, a, b in self.gates:
            values[lhs // 2] = value(a) and value(b)
        return tuple(value(r) for r in self.outputs)
