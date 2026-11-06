"""One cost policy shared by predicted search and measured final selection."""
from dataclasses import asdict, dataclass
import math
from .evaluator import positive_metrics


@dataclass(frozen=True)
class Objective:
    metric: str = 'adp'
    max_area: float = None
    max_delay: float = None

    def __post_init__(self):
        if self.metric not in ('adp', 'area', 'delay'):
            raise ValueError('Objective must be adp, area, or delay')
        for limit in (self.max_area, self.max_delay):
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError('Area and delay limits must be finite and positive')

    @property
    def constrained(self):
        return self.max_area is not None or self.max_delay is not None

    def cost(self, metrics):
        positive_metrics(metrics)
        return metrics['area'] * metrics['delay'] if self.metric == 'adp' else metrics[self.metric]

    def violation(self, metrics):
        positive_metrics(metrics)
        return sum(max(0., metrics[name] / limit - 1.) for name, limit in
                   (('area', self.max_area), ('delay', self.max_delay)) if limit is not None)

    def feasible(self, metrics):
        return self.violation(metrics) == 0

    def key(self, metrics):
        # Feasibility first. Among infeasible states, reduce normalized total
        # violation; then minimize cost. A verified fallback may be infeasible.
        violation = self.violation(metrics)
        return (violation > 0, violation, self.cost(metrics))

    def better(self, candidate, incumbent):
        return self.key(candidate) < self.key(incumbent)

    def reward(self, metrics, baseline):
        cost, base = self.cost(metrics), self.cost(baseline)
        if not self.constrained:
            return 1. - cost / base
        violation = self.violation(metrics)
        if violation:
            return -violation / (1. + violation)
        # Bounded positive rewards put every predicted feasible state above
        # every infeasible one. Baseline is fixed across decisions.
        return 1. - cost / (cost + base)

    def describe(self, metrics):
        return {'feasible': self.feasible(metrics), 'normalized_violation': self.violation(metrics),
                'cost': self.cost(metrics)}

    def payload(self):
        return asdict(self)
