# Nangate45 typical-corner library

This directory contains an unchanged copy of the Nangate Open Cell Library typical Liberty file distributed with [OpenROAD Flow Scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/51b237556adb2ce42e32ad75d44e739d6072de5f/flow/platforms/nangate45). `source.json` pins the upstream revision, URLs, byte counts and SHA-256 hashes. The upstream platform `LICENSE` and original Liberty copyright header are preserved. No blanket project license supersedes them.

Corner: typical process, 1.1 V, 25 °C. This is an open research standard-cell library, not a production-foundry signoff result.

The `.lib` backend reads the complete Liberty data using ABC `read_lib`. It maps using `map`, then runs `topo; stime` for nonlinear cell-table timing with the fixed `abc-liberty-map-stime-v1` profile:

- Primary-input driver: BUF_X1.
- Primary-output load: 10 fF on every output.
- Wire-load model: disabled; no extracted or placement-based wire parasitics.
- Cell area: square micrometres for this library.
- Delay: picoseconds as reported by ABC `stime`.
- No gate sizing or buffering pass is added.

Library loading derives a gate model for mapping. Final acceptance and training labels use the full-library timing result, which depends on slew and pin loading. They do not use the illustrative GENLIB's delay model. These assumptions are deliberately simple and shared by every method; they are not a replacement for a physical-design or signoff timing flow.

The measurement profile is recorded alongside the library hash in new datasets, checkpoints, traces and portable results. The optimizer rejects a mismatched profile. GENLIB checkpoints remain supported for their original measurement flow. Liberty results store `library.lib` in the portable bundle, and `aigopt verify` independently repeats the same timing measurement.

The default Liberty profile currently requires a BUF_X1 cell and is validated against this Nangate45 file. Supporting a different library may require an explicitly designed driver/load profile and new labels; renaming another file to `.lib` does not make its assumptions equivalent.
