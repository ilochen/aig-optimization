# Architecture

The optimizer operates on combinational AND-inverter graphs. Each search state is an immutable AIG file produced by Berkeley ABC; graph literals preserve complemented edges and shared logic.

## Offline learning

ABC transformations generate candidate structures. Technology mapping produces area and delay labels, and equivalence checks certify both the transformed AIG and mapped netlist. Exact graph hashes remove duplicate states. Feature preprocessing and cost scales are serialized with library/profile hashes.

The feature evaluator is a 4→8→2 MLP using AND count, depth, complemented gate-input count and maximum fanout. Scaled residual targets restore size factors at inference. A reference-relative branch learns candidate cost changes from the original graph, using centered metric/ADP fitting and gap-weighted pair-ranking supervision. The selected correction uses four summary-information inputs and four hidden neurons. A hybrid combines that branch with the calibrated MLP.

Reference-relative predictions are positive cost proxies in structural units. Their ordering and normalized rewards are useful within a reference circuit; they are not absolute physical area or delay. Separate circuit bindings do not change a shared checkpoint, and a bound reference remains fixed across replanning decisions.

## Search

UCT uses `Q + c sqrt(log(N_parent) / N)` with default exploration coefficient 0.8. Each trial selects a path, expands one action, evaluates the graph without mapping, and backs up the reward along the visited path. Seeded action order and frozen per-decision macros make runs inspectable. Exact graph identities cache inference without merging separate tree-edge statistics.

The eight actions include rewrite/balance/refactor/resubstitution primitives and a seeded ABC9/LUT macro. The macro starts with `&get`, explicitly transferring the current candidate to ABC9. Progressive widening can encourage deeper exploration. No immediate-improvement pruning prevents useful sequences from crossing a worse intermediate state.

Default ADP reward is `1 − predicted_ADP / initial_predicted_ADP`. Other objectives and constraints share one policy between search predictions and measured final selection. Constraint feasibility takes priority; otherwise relative violation and selected cost determine preference.

## Final selection and export

A bounded shortlist contains the primary proposal and predicted area/delay extremes, deduplicated by exact graph hash. Candidates are mapped after search. The primary is certified before attempting better measured alternatives, so late proof timeouts cannot discard a verified incumbent. Measured regressions retain the baseline under the chosen objective.

Export is staged only after certification. A portable bundle contains source/reference logic, final AIG/netlist, library, checkpoint, trace and hashes. The verifier does not execute saved commands: it checks integrity, proves source equivalence to the reference and final artifacts, repeats baseline/final mapping, measures the submitted netlist and checks the reported objective.

## Supported scope

The flow supports flat, ABC-readable combinational designs. Sequential AIGER and zero-cost constant-only designs are unsupported. Nangate45 timing uses cell tables and explicit driver/load assumptions without placement, extracted wires, buffering or sizing. Successful logic verification does not imply that an arbitrary user timing limit is feasible.
