"""Cheap load/path topology summaries; no mapping or measured test costs."""
from collections import Counter
import math
from .descriptors import NAMES as BASE_NAMES, describe as base_describe

VERSION='aig-load-descriptors-v2'
NAMES=[*BASE_NAMES,'log_load_weighted_depth','weighted_depth_ratio','critical_fanout_mean',
    'log_critical_fanout_max','gate_fanout_std','high_fanout_fraction','log_input_load_sum',
    'max_local_cut_fanout','log_phase_weighted_area','positive_chain_cover_fraction',
    'critical_inversion_fraction','log_output_load_ratio']


def describe(graph):
    result=base_describe(graph);count=max(1,len(graph.gates));depth=max(1,graph.features()[1])
    gates={lhs//2:(a,b) for lhs,a,b in graph.gates};fan=Counter()
    for _,a,b in graph.gates:
        fan[a//2]+=1;fan[b//2]+=1
    for literal in graph.outputs:fan[literal//2]+=1
    levels={0:0,**{r//2:0 for r in graph.inputs}};weighted={k:0. for k in levels}
    for lhs,a,b in graph.gates:
        k=lhs//2;levels[k]=1+max(levels[a//2],levels[b//2])
        # Unit gate delays augmented by pin-loading and inversion penalties.
        # These are topology proxies, not Liberty timing or fitted cell data.
        weighted[k]=max(weighted[a//2],weighted[b//2])+1+.15*math.log1p(fan[k])+.1*((a&1)+(b&1))
    distance={r//2:0 for r in graph.outputs}
    for lhs,a,b in reversed(graph.gates):
        k=lhs//2
        if k in distance:
            for r in (a,b):distance[r//2]=max(distance.get(r//2,-1),distance[k]+1)
    output_depth=max((levels[r//2] for r in graph.outputs),default=0)
    critical=[k for k in gates if levels[k]+distance.get(k,-10**9)==output_depth]
    fans=[fan[k] for k in gates];average=sum(fans)/count
    spread=math.sqrt(sum((n-average)**2 for n in fans)/count)
    weighted_depth=max((weighted[r//2] for r in graph.outputs),default=0)
    merge=sum(any(not(r&1) and r//2 in gates and fan[r//2]==1 for r in ab) for ab in gates.values())
    critical_set=set(critical)
    inv=sum((a&1)+(b&1) for k,(a,b) in gates.items() if k in critical_set)
    result.extend([math.log1p(weighted_depth),weighted_depth/depth,
        sum(fan[k] for k in critical)/max(1,len(critical)),math.log1p(max((fan[k] for k in critical),default=0)),
        spread,sum(n>=4 for n in fans)/count,math.log1p(sum(fan[r//2] for r in graph.inputs)),
        max((fan[a//2]+fan[b//2] for a,b in gates.values()),default=0),
        math.log1p(sum(1+.1*((a&1)+(b&1))+.1*math.log1p(fan[k]) for k,(a,b) in gates.items())),
        merge/count,inv/(2*max(1,len(critical))),math.log1p(10*len(graph.outputs)/count)])
    return result
