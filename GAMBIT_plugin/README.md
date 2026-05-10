# GAMBIT plugin: paraprof

This folder contains the ScannerBit Python plugin that exposes
[paraprof](https://github.com/anderkve/paraprof) as a scanner inside
[GAMBIT](https://github.com/GambitBSM/gambit).

## Files

- `gambit_paraprof.py` — the plugin module. Drop it into
  `ScannerBit/src/scanners/python/plugins/` in your GAMBIT source tree.
- `paraprof_example.yaml` — a minimal scanner YAML snippet showing the
  exposed options.

## Requirements

- GAMBIT compiled with MPI (`cmake -DWITH_MPI=1 …`).
- At least 2 MPI processes (1 master + ≥1 worker). The paraprof master rank
  performs no target evaluations; throwing fewer than 2 ranks at it will
  raise an explicit error during plugin construction.
- `paraprof` and `mpi4py` installed in the Python environment GAMBIT picks up
  (typically the same one used by other Python scanners). To install paraprof:
  `pip install git+https://github.com/anderkve/paraprof.git`.

## Plugin name

The plugin registers itself as `paraprof`, so set:

```yaml
Scanner:
  use_scanner: paraprof
  scanners:
    paraprof:
      plugin: paraprof
      ...
```

See `paraprof_example.yaml` for a complete worked example.

## How it integrates with ScannerBit

- The plugin runs paraprof's master/worker MPI scheme inside ScannerBit's
  single-`run()`-per-rank model. Rank 0 builds the `ProfileProjector` and
  enters paraprof's master loop; ranks 1+ enter `paraprof.worker_main` with
  the GAMBIT loglike supplied directly (the bound method cannot be pickled,
  so the standard paraprof "broadcast the target function" path is bypassed).
- Every worker evaluation goes through `self.loglike_hypercube(x)` followed
  by `self.print(1.0, "Posterior")`. That keeps the standard ScannerBit
  printer flow intact: each evaluation gets its own point id and shows up in
  GAMBIT's HDF5/ASCII output exactly as for the other Python scanners
  (`grid`, `binminpy`, etc.).
- Projection `dims` may be parameter **names** (matching the YAML
  `Parameters` block) or integer indices; names are resolved by paraprof.
- `samples_output_file` is forwarded to paraprof and only written by rank 0.
  This is purely a paraprof-side diagnostic: GAMBIT's printers already
  capture every evaluation, so it defaults to off.
- `save_plots: true` enables paraprof's built-in 1D/2D profile plots
  alongside the GAMBIT outputs, written to the working directory.
