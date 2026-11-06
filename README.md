# Technology-Aware AIG Optimization

NTU Deep Reinforcement Learning course project.

Monte Carlo tree search over Berkeley ABC transformations, guided by learned estimates of technology-mapped area and delay. The pipeline separates expensive label generation from search-time inference and certifies the exported circuit with equivalence checks and measured mapped costs.

[Interactive demonstration](https://ilochen.github.io/projects/aig-optimization/) · [Architecture](docs/architecture.md) · [Benchmark results](docs/evaluation.md)

```text
Circuits → transformations → technology mapping → verified cost labels → train evaluator
AIG → UCT selection/expansion → ABC action → graph features → cost prediction → backup
Candidate shortlist → actual mapping → equivalence proofs → verified result or baseline
```

## Quick start

Python 3.10+ and a compiled Berkeley ABC executable are required. The Python runtime has no third-party dependencies.

```sh
git clone https://github.com/berkeley-abc/abc.git abc
git -C abc checkout 8ba3d9b91c3d529223e90bcab0e9094bc015eac7
make -C abc -j4 ABC_USE_NO_READLINE=1

python3 -m aigopt optimize --abc ./abc/abc \
  --library examples/teaching.genlib --model examples/evaluator.json \
  --input examples/chain-4.blif --choice best --widening 1 --iterations 24 \
  --finalists 3 --verification-budget 30 --compact-trace --output build/example

python3 -m aigopt verify --abc ./abc/abc --result build/example
```

Optional installation provides an `aigopt` command:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install .
aigopt --help
```

Each result includes the final AIG and mapped netlist, the original input and normalized reference, library, checkpoint, trace, and hashed artifact manifest. Verification repeats logic checks and measured costs in a private workspace. Use a new output directory for each optimization.

## Standard-cell mapping

The repository includes the unchanged Nangate45 typical-corner Liberty library with upstream license and source hashes. The Liberty backend uses `map; topo; stime`, BUF_X1 input drivers, 10 fF output loads, and no wire parasitics. These are pre-layout cell-timing estimates, not signoff timing. See [the library profile](libraries/nangate45/README.md).

```sh
python3 -m aigopt optimize --abc ./abc/abc \
  --library libraries/nangate45/NangateOpenCellLibrary_typical.lib \
  --model benchmarks/checkpoints/relative.json --input examples/chain-4.blif \
  --choice best --widening 1 --iterations 24 --finalists 3 \
  --verification-budget 30 --compact-trace --output build/nangate-example
```

Checkpoints bind to a library hash and mapping/timing profile. The included relative evaluator combines a calibrated feature MLP with a small reference-relative neural correction. Its predictions are ranking proxies; measured costs determine acceptance.

## Training

Generate verified labels for a target library and train a feature evaluator:

```sh
python3 -m aigopt collect --abc ./abc/abc --library examples/teaching.genlib \
  --samples 17 --output build/labels.json examples/chain-*.blif
python3 -m aigopt train --dataset build/labels.json \
  --epochs 1500 --output build/evaluator.json
```

Collection deduplicates exact AIG graphs and proves the transformed and mapped circuits equivalent before recording labels. The feature MLP uses a circuit-grouped training/validation split. Incomplete collections are rejected by training.

The supplied ranking dataset and baseline checkpoint also support circuit-grouped model selection and reference-relative training:

```sh
python3 tools/cross_validate.py --dataset benchmarks/data/ranking.json --output build/cv
python3 tools/train_model.py --dataset benchmarks/data/ranking.json \
  --cv build/cv --output build/models
```

The relative loss emphasizes differences among candidates from the same circuit. Gap-weighted pair supervision ignores differences below 0.5%; reference binding stays fixed across replanning decisions. Model selection and its tradeoffs are documented in [the evaluation](docs/evaluation.md).

## Search controls

- `--choice best` retains the best full sequence found; `--choice root` follows the highest-mean root action.
- `--widening 1` trades root breadth for deeper exploration at small budgets.
- `--steps` repeats planning while retaining the original reward/reference baseline.
- `--objective adp`, `area`, or `delay` controls predicted and measured selection.
- `--max-area` and `--max-delay` prioritize measured feasibility. An unmet limit is explicitly reported as `constraints_unmet`.
- `--finalists 3` maps distinct complementary candidates after search, allowing at most four mapper calls including the baseline.
- `--verification-budget` sets total proof time independently of per-command `--timeout`.
- `--retain-finalists` saves diagnostic candidates with individual proof statuses.

Mapping never runs inside surrogate-guided search. Failed actions are recorded and excluded. A verified incumbent remains available if another candidate times out during proof. A result can preserve logic while still missing a requested area or timing limit.

## Evaluation

On nine held-out ISCAS’85 designs, three seeds, and matched three-finalist budgets:

| Guidance | Mean accepted area × delay reduction |
| --- | ---: |
| Structural | 25.48% |
| Baseline neural | 25.33% |
| Regression neural with expanded data | 24.62% |
| Reference-relative neural | **26.28%** |

All 108 runs passed equivalence checks. The relative model improves the baseline neural result by 0.95 percentage points, with circuit-bootstrap interval [+0.39, +1.58]. Its 0.79-point lead over structural guidance has interval [−0.24, +1.83], so a reliable advantage over the structural baseline is not established. The corpus, one training seed, and one library corner limit generalization.

Reproduce the comparison with the pinned circuit archive:

```sh
python3 tools/fetch_inputs.py  # requires libarchive bsdtar
python3 tools/run_benchmark.py --abc ./abc/abc --selection build/models \
  --dataset benchmarks/data/ranking.json --output build/evaluation
```

[Full methodology and ranking results](docs/evaluation.md) · [Recorded runs](benchmarks/evaluation/iscas85/runs.jsonl) · [Artifact hashes](benchmarks/evidence.json)

## Validation

```sh
ABC_BINARY="$PWD/abc/abc" python3 -m unittest discover -s tests -v
node --test tests/browser-model.test.js
```

The suites cover graph parsing, numerical training gradients, fixed-reference inference, UCT behavior, real ABC transformations/mapping, equivalence failures, library-profile mismatches, portable results, and the browser demonstration. The GitHub workflow builds the pinned ABC revision and runs the Python and browser suites.

## Course materials and credits

The original NTU Deep Reinforcement Learning course environment is credited to [lin-1214/DRL_Final](https://github.com/lin-1214/DRL_Final). Supplied environment variants, helper code, and their README are preserved under `legacy/`. Its instructions describe those archived files. The packaged entry point is `python3 -m aigopt`.

The course report's pre-mapping AND-count results and this repository's mapped area/delay evaluations use different metrics and should be cited separately. The browser demonstration uses its own illustrative transformations, cell-cover labels and weights; it does not invoke ABC or share the standard-cell checkpoint.

Berkeley ABC, Nangate45, and benchmark sources retain their respective attribution and licensing terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
