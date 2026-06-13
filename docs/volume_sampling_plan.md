# Volume sampling: design and architecture

The volume-sampling stage collects a sample set that **balances space-filling
and lnL-filling** inside the region of interest
`{logL > global_max − roi_threshold}` after the projections complete. It is
built on an **affine-invariant ensemble of umbrella walkers** (this document
is the design record; the user-facing description lives in the README).

## Goal

Profile-likelihood scans concentrate samples on low-dimensional profile
surfaces (measure zero in the ROI volume). Global-fit users also want
representative points spread through the good-fit volume *and* across fit
quality (lnL). The stage delivers both:

- **space-filling** — points spread through the ROI volume, including thin
  valleys and disconnected islands;
- **lnL-filling** — points spread across the band in lnL, not piled at the
  edge (which is where the bulk of the volume lives in high dimensions).

The stage's `roi_threshold` defaults to the projection's but can be set larger
to reach into the shell outside the good-fit region (the band is one-sided,
running up to the global max).

## Algorithm

A single pooled ensemble of `n_walkers` walkers. Each walker `k` carries a
fixed **home level** `ℓ_k`, assigned to span the band uniformly in ΔlnL:

    ℓ_k = global_max − (k + 0.5) / n_walkers · roi_threshold

and an individual **umbrella target**

    π_k(θ) ∝ exp(−(logL(θ) − ℓ_k)² / (2σ²)),   σ = sigma_frac · roi_threshold.

Walkers are updated with the **Goodman–Weare affine-invariant stretch move**:
to update walker `k`, pick a partner `j` from the frozen complementary half
(red/black split) and propose

    θ_k' = θ_j + Z·(θ_k − θ_j),   Z ~ g(z) ∝ 1/√z on [1/a, a],  a = 2,

accept with `min(1, Z^(d−1)·π_k(θ_k')/π_k(θ_k))`. Affine invariance handles
stretched/curved geometries with no step size, gradient, or covariance tuning.

**Pooling.** Partners are drawn from the whole ensemble by default (no shell
structure). Each walker keeps targeting its own umbrella regardless of the
partner's level — the stretch acceptance is a valid Metropolis–Hastings update
for any target with any frozen partner, so detailed balance holds. Pooling was
measured to match strict same-level shells on lnL-uniformity and coverage at
~10% lower acceptance, while removing the `K ≳ 2d` per-shell sizing constraint.
`partner_level_window` optionally restricts partners to walkers within a lnL
window of `ℓ_k` (a fixed property, so correctness-safe); `None` = full pool.

**Seeding / warm start.** Walkers start at points drawn inside the
`ProjectionEnvelope` (the converged projection grids as an upper-bound
prefilter). With warm start on, each walker instead starts from a logged scan
sample whose logL is near its home level `ℓ_k` (using the projection run's
work to place high-level walkers near the peak), falling back to envelope draws
where the scan has no nearby points.

**Parallelism.** Red/black sweeps: each sub-sweep proposes for one half of the
ensemble (independent, since partners are frozen) and evaluates them as one
batch on the worker pool. `n_steps` sweeps, optionally capped by `eval_budget`.

**Logging.** Every evaluation streams to `samples_output_file` tagged
`PHASE_VOLUME`. Band membership is re-derived at output time from stored logL
against the final global max (so a mid-stage global-max improvement is handled).

## Configuration

```python
volume_sampling = {
    'roi_threshold': None,         # band depth ΔlnL; None = projection's roi_threshold
    'n_walkers': 1000,             # ensemble size (each carries a home level)
    'n_steps': 30,                 # stretch sweeps per walker
    'eval_budget': None,           # optional cap; None = n_walkers*(n_steps+1)
    'sigma_frac': 0.05,            # umbrella width sigma / roi_threshold
    'partner_level_window': None,  # lnL window for partners; None = full pool
    'warm_start': True,            # seed near each level from scan samples
    'output_file': 'volume_samples.csv',   # in-band samples [params..., logL]
    'summary_file': None,          # JSON; None = derived from output_file
}
```

Run automatically by `run_all_projections` when `volume_sampling` is set; also
exposed standalone as `run_volume_sampling(comm, sampler, projection_results)`.
The stage skips itself if a projection grids the full parameter space
(direct-eval mode — the grid already covers the volume).

## Module layout

- `volume.py` — `ProjectionEnvelope` (seeding + degenerate-skip), config
  validation (`normalize_volume_config`), the ensemble core as MPI-free
  helpers (`assign_levels`, `umbrella_logpi`, `draw_stretch`, `propose_stretch`,
  `warm_start_positions`, `write_volume_output`), and `volume_band`.
- `master.py` — `run_volume_sampling`: builds the envelope, seeds walkers,
  runs the red/black batch loop on the worker pool, writes outputs.
- `phases.py` — single `PHASE_VOLUME` (replaces `PHASE_VOLUME_PROBE/SEARCH`).
- `visualization.py` — `plot_volume_samples`: scatter the in-band samples on a
  2D projection, coloured by logL.

## Outputs

- **Sample file** (`output_file`, csv/h5): one row `[params..., logL]` per
  in-band evaluation — the populated ROI set.
- **JSON summary** (`summary_file`): config, band edge, walker/step/eval
  counts, in-band fraction, mean acceptance, global-max drift, and the logL
  histogram of the in-band samples (to check the achieved lnL spread).
- `sampler.volume_stage_result` — the same in memory, plus the sample array.

## Removed (relative to the earlier funnel design)

The three-tier harvest/probe/anchored-search funnel, the interior walk +
tangential randomization + adaptive depth quota, per-anchor coverage radii and
the `covered/projected/hole/unbudgeted` classification, the uniform-probe
subset and the band **volume estimate**, and the provenance tag column. These
were replaced wholesale after prototyping showed the ensemble gives better
lnL-balance and stiff-geometry coverage with far fewer moving parts.
