# Implementation plan: curvature-scaled quasi-Newton polish

Status: implemented in `LBFGSBJob` (see the class docstring and the
changelog). Deviations from the plan below: a positive curvature estimate
outside `[1e-6, 1] * width` is clipped to that range rather than replaced by
the box width (only a non-positive or non-finite estimate falls back); a
history-free step is capped so that no coordinate moves by more than a
quarter of its box, which bounds the number of Armijo halvings from a far
start when the fallback scale is in use; and a converged start on a stiff
direction still gets the probe, because the first gradient is taken with the
box-scaled finite-difference step and its forward-difference bias
(eps * h_ii / 2) reads as a gradient there. The benchmark results are recorded
in the paper draft's `benchmark_extra.md`.

## Problem

`LBFGSBJob` runs its two-loop recursion in the raw parameter units of the
target. With an empty history the first search direction is minus the
gradient with an identity initial Hessian, so on a likelihood whose profiled
parameters have very different scales (a toy neutrino oscillation fit: the
sin^2(theta13) box is 0.025 wide with a prior width of 0.0007, the Delta m^2_31
box is 6.4 wide) the first step overshoots the stiff directions by orders of
magnitude, the Armijo backtracking halves alpha 15-20 times before a step is
accepted, and the single scalar scaling L-BFGS derives from the first pair then
leaves the wide directions under-stepped. Measured with a synchronous driver
from three stalled cells: 434-529 evaluations and 50-60 iterations per polish;
in bounds-normalised coordinates 220-335 and 24-39. scipy's L-BFGS-B needs
336-488 from the same points, MIGRAD (which builds a diagonal Hessian
estimate before its first step) needed 2.5-4x fewer evaluations per cell than
scipy on that target.

## Design

Per job, a positive scale vector `scale` over the optimised dims (length
`n_opt_dims`), fixed for the job's lifetime, sets a diagonal metric:

- two-loop recursion in u = x / scale: `q = grad * scale`, pairs enter as
  `s / scale` and `y * scale`, the direction is `-(z * scale)`; the stored
  `s_hist` / `y_hist` stay in raw units so neighbour seeding and
  `optimizer_state` are unchanged. With no pairs the first step is
  `-(grad * scale**2)`, a diagonal Newton step when `scale_i = 1 / sqrt(h_ii)`.
- finite-difference step `eps_i = FD_REL * scale_i` (FD_REL between 1e-7 and
  1e-6, floored at 1e-10) instead of `1e-8 * |x_i|`, which goes to 1e-10 for a
  parameter sitting near zero.

The scale comes from a one-round curvature probe:

1. After the first gradient round at x0 (status `NEEDS_GRADIENT` ->
   `_finalize_gradient_and_line_search`, `iteration == 0`), issue one task per
   optimised dim at `x0 + h_i e_i` with `h_i = 0.01 * width_i` (probe in the
   other direction when the step leaves the box), sub-type
   `LBFGS_CURVATURE`, new status `NEEDS_CURVATURE`.
2. On completion: `h_ii = 2 (f(x0 + h_i e_i) - f0 - g_i h_i) / h_i^2` with the
   forward-difference `g_i` already in hand. `scale_i = 1 / sqrt(h_ii)` when
   `h_ii > 0` and the result lies in `[1e-6, 1] * width_i`; otherwise the
   fallback `scale_i = width_i` (box width). A flat or concave direction at a
   far start therefore keeps today's behaviour in that direction.
3. Then continue exactly as now: two-loop, line search.

Skip the probe (keep the box fallback) when the job will not iterate anyway:
a converged start where the predicted first-step decrease
`c1 * |g . (-(g * width^2))|` is already below `ftol`, so the known-start /
flat-step path still costs `n_Q + 1` evaluations. Apply the probe to every
job type, including `INITIAL_OPTIMIZATION`: the global stage was the dominant
cost on the toy (356k of 470k evaluations) and the quadratic priors give the
same curvature everywhere, which is where the stiffness sits.

Cost: `n_Q` evaluations per polish that iterates (8 on the toy against ~400 per
polish; 2 on Himmelblau against ~15-25 per cell), issued as one parallel round.

## Tests

Synchronous driver (`tests/test_lbfgsb_exits.py` style):

- separable quadratic with widths 1e-3 and 1 in a [-1, 1]^2 box: from a start
  off the optimum the polish converges within 3 iterations with the probe,
  first accepted alpha = 1, `scale` within 10% of the widths.
- a flat direction (f independent of x1): `scale_1 == width_1` (fallback).
- a converged start: no `LBFGS_CURVATURE` tasks, total cost `n_Q + 1`.
- the sphere and Himmelblau results of the existing exit tests unchanged to
  tolerance; the neighbour-seeding tests unchanged (pairs still in raw units).
- user-gradient jobs: probe tasks are plain evaluations, `compute_gradient`
  false.

## Benchmarks and acceptance

1. Driver on the three stalled toy cells: expect <= the unit-box numbers
   (220-335 evaluations).
2. Toy, all four projections, one seed: polish and global-stage evaluations
   before/after at equal quality; then three seeds.
3. Full test-function sweep (`benchmarks/run_all.sh`, paraprof phases) and
   `compare_runs.py` against `results/archive_v2`: quality unchanged on every
   case, scan-stage cost within +-5% on the equal-width boxes, Rosenbrock
   and the 12-D nuisance case not worse.
4. Keep only if (3) holds; otherwise restrict the probe to grid-anchored jobs
   or drop it. Record the tables in the paper draft's `benchmark_extra.md`.

## Not in scope

Normalising by the spread of the profiled parameters across the ROI cells
(data-driven, but circular during the global stage and conflating cell-to-cell
variation with local curvature); user-supplied per-parameter scales (could be
added later as an optional argument, defaulting to the box width).
