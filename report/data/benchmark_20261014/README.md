# Validation benchmark data (2026-10-14)

These JSON files are fresh local measurements for the merged-project report. They were generated on macOS arm64 with the project Python 3.11.17 environment. Each heavy method ran in an isolated subprocess with `PYTHONHASHSEED=0`.

- `uca_3h_rerun_20261014.json`: one fresh UCA-P1 direct, colour-first, and incidence+S3 comparison for relativistic 3H (`p+n+n`, source spins `(0,0,1)`). All methods emitted 3,775,680 packed entries; candidate dictionaries were compared row by row against UCA.
- `recursive_3h_rerun_20261014.json`: one fresh UCA versus sparse antisymmetric recursion comparison for the same 3H fixture. The complete expanded dictionaries were compared exactly.
- `seven_examples_rerun_20261014.json`: one isolated run for the seven ordinary-colour examples used by the sibling project.
- `eighth_example_pn_omega_pi_plus_20261014.json`: raw Wick enumeration for the eighth historical fixture (`pn_omega_pi+`) plus a bounded fused-materialization attempt. The latter exceeded a 100,000,000-entry safety cap and is explicitly marked deferred.
- `pnn_special_input_20261014.json`: explicit normal-special 3He input with source external spins `(0,0,1)` used for the merged in-memory fused run.
- `ppnn_count_rerun_20261014.json`: exact ppnn/4He count-only validation (`518,400` Wick maps; `Nlist=1,407,974,400`; `Ncontr=87,998,400`). It does not claim entry-by-entry equality for the full dictionary.
- `eight_examples_plus_ppnn_validation_20261014.json`: compact report-facing summary joining these measurements and limitations.

The UCA and recursive scripts exclude SQLite output and comparison from timed regions; process peak RSS is measured after dictionary generation. The seven-example runner includes normalization and fused output dictionary construction. The ppnn count runner does not materialize the full independent dictionary.
- `ppnn_fixed_288_progress_20261014.json`: saved progress record for the selected 288-network normal-wick block evaluation. Its `max_rss_bytes=13152583680` is the process peak recorded during fixed-layout evaluation; it does not include the separate plan-generation phase.

The combined summary also records a merged-project `normal-special` p+n+n cross-time smoke: 2,880 Wick maps, 3,775,680 dictionary entries, 4.9023679 s wall time, and 1,110.953125 MiB peak RSS. This in-memory benchmark verifies the compact cross-time kernel. The `write_unified` file route preserves the colour-orbit JSON v3 sparse schema; the in-memory kernel is used only for this performance and dictionary comparison.
