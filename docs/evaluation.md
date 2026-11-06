# Mapped-cost evaluation

This evaluation measures the implementation distributed in this repository. It is separate from the archived course report's pre-mapping AND-count table.

## Protocol

Nine ISCAS’85 source designs are held out from training and model selection. c1355 is excluded because it is functionally equivalent to c499; c17 is outside the standard ten-circuit set. The archive URL/hash and source file hashes are in [the suite](../benchmarks/suite.json). Normalized original graphs were checked against training graph hashes. These checks do not establish arbitrary subcircuit or semantic independence.

Training uses 532 verified graph states from 34 IWLS, EPFL and teaching sources. Three circuit-grouped control folds are used for model selection. No ISCAS outcome contributes to fitting, preprocessing or selection. Labels, circuit identities, baseline weights and model-selection records are included.

All methods use Nangate45 typical process, 1.1 V, 25 °C. ABC `read_lib` and `map` produce the mapped design; `topo; stime` measures full-library timing with BUF_X1 input drivers and 10 fF output loads. Wire-load models and physical parasitics are disabled. Area is in square micrometres and delay in picoseconds for this library.

Each method receives three seeds (11, 23, 37), 24 candidate-action trials, depth three, widening one, best-candidate selection and three finalists. At most four mappings are allowed, including the baseline. Transformation/mapping commands have a ten-second timeout; baseline and candidate proofs share thirty seconds. Actions and maximum mapper calls are matched; wall time is reported rather than matched. Offline training cost is excluded from run time. Failed runs count as zero accepted improvement.

## Results

| Guidance | Mean accepted ADP reduction | Mean mapper calls | Median run time |
| --- | ---: | ---: | ---: |
| Structural | 25.48% | 4.00 | 1.41 s |
| Baseline neural | 25.33% | 4.00 | 1.40 s |
| Regression neural with expanded data | 24.62% | 3.89 | 1.51 s |
| Reference-relative neural | **26.28%** | 4.00 | 1.40 s |

All 108 scheduled runs produced verified outputs. Means average seeds within circuit, then weight circuits equally. The relative model beats the baseline neural model by 0.95 percentage points, with circuit-bootstrap 95% interval [+0.39, +1.58]; the 27 matched circuit/seed pairs contain nine wins, zero losses and eighteen ties. Its 0.79-point difference over structural guidance has interval [−0.24, +1.83], which does not establish a reliable structural-baseline advantage.

An equal-family sensitivity check gives relative-minus-structural difference +0.02 points with interval [−0.97, +1.29], and relative-minus-baseline-neural difference +1.42 points with interval [+0.50, +2.35]. The four descriptive clusters are small and unequal; ALU/control dominates the corpus. These intervals are exploratory rather than broad independence guarantees.

## Ranking and model selection

A model-independent probe pool applies the original graph plus eight root actions per circuit. Exact duplicates are removed; all 75 distinct mapped graphs passed equivalence. True relative cost ties below 1e-4 are excluded, and predicted ties below 1e-6 receive half credit.

| Guidance | Mean pair accuracy | Mean probe top-choice regret |
| --- | ---: | ---: |
| Structural | 76.97% | 1.95% |
| Baseline neural | 78.34% | **1.66%** |
| Regression neural with expanded data | 69.84% | 6.14% |
| Reference-relative neural | **79.57%** | 1.95% |

Relative-minus-baseline-neural pair accuracy is +1.23 points, interval [−0.49, +2.87]; relative-minus-structural is +2.60 points, interval [+0.62, +5.03]. Improved overall pair ordering did not improve the single best root choice in this pool. Pair accuracy, top-choice regret and multistep search outcomes are distinct measures.

The ranking objective addresses the distinction between circuit-wide calibration and within-circuit choice. In the baseline training subset, about 85.3% of area-residual variance and 95.5% of delay-residual variance lies between circuits. Centered training and reference-relative inputs emphasize changes among implementations of one circuit. This identifies an objective mismatch without proving it is the only source of errors.

Both 28-feature topology and 40-feature load/path variants were evaluated. More features did not consistently win selection. The selected hybrid uses the calibrated 4→8→2 MLP and a four-summary-information, four-hidden-unit relative branch, with residual blend 0.5 and hybrid mix 0.75.

The baseline neural model retained the best validation top-choice regret. The ranking-focused candidate was selected for pair accuracy subject to at most 0.5 percentage point additional mean choice regret. Validation regret/accuracy were 2.92%/86.71% for the baseline and 3.15%/87.21% for the hybrid. The expanded-data regression control retains the original internal holdout, whereas the relative branch refits all training examples after grouped selection; it does not isolate every fitting-policy difference.

## Evidence and limits

- [Recorded search runs](../benchmarks/evaluation/iscas85/runs.jsonl)
- [Independent probe records](../benchmarks/evaluation/iscas85/probes.json)
- [Aggregate analysis](../benchmarks/evaluation/iscas85/analysis.json)
- [Cross-validation results](../benchmarks/selection/cross-validation.json)
- [Ranking selection policy](../benchmarks/selection/ranking-policy.json)
- [Core-code, library and artifact hashes](../benchmarks/evidence.json)

The selected evaluator emits relative ranking proxies. Absolute area/delay MAPE fields produced by the generic probe helper are not physically meaningful for it and are not used here. Numerical costs, candidate identities and proof statuses are preserved in the records; private absolute scratch/tool paths are removed from diagnostic strings if present.

One training seed, one library corner, nine related designs and pre-layout timing limit generalization. No foundry signoff, equal-runtime speedup or universal neural advantage is claimed. Circuit sources are fetched from their providers rather than redistributed. Any further tuning on this set requires a different held-out set for new generalization claims.
