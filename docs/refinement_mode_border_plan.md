# Refinement at a border between two optima: finding and proposal

Status: proposal, not implemented. Written November 2026, against the code
after the removal of dead and inert code (changelog, Unreleased). Step 2 of
the proposal was revised when patching got its test for a jump between
neighbouring cells; the measurements were not repeated.

## Summary

Refinement clustering is meant to find the correct optimum of the profiled
parameters where the refined grid crosses a border between two optima. With
the default settings it does not act: the clustering runs, but its result is
only used with `refinement_direct_eval=True`.

The proposal is to use several start candidates in optimisation mode as well,
and to take them from a local test on the neighbouring coarse cells instead of
from the clustering. In the tests below this puts every cell next to a border
in the correct optimum, at 0.4 to 6.6 % more refinement evaluations, and the
clustering code and the scikit-learn dependency are no longer needed.

## What the code does now

1. `setup_refinement_run` clusters the coarse cells by their profiled
   parameters and projection coordinates (`cluster_coarse_grid_by_modes`,
   DBSCAN). A cell with a neighbour in another cluster is a border cell.
2. For a fine grid point with a border cell among its six nearest coarse
   cells, `get_cluster_based_initialization` returns several candidates for
   the profiled parameters. The first is the plain interpolation. The others
   are one weighted mean per nearby cluster.
3. `create_refinement_lbfgsb_jobs` evaluates all candidates and keeps the best
   only if `refinement_direct_eval` is set. Otherwise it starts the L-BFGS-B
   run from the first candidate, the interpolated point.

So with `refinement_direct_eval=False` (the default) the candidates from the
clusters are computed and dropped. Refined grids were identical with
`use_clustering` on and off in 30 of 30 runs. Without scikit-learn the
clustering is skipped with a warning.

## Why the interpolated start fails at a border

On the two sides of a border the coarse cells hold different optima of the
profiled parameters. Interpolating between them gives a point between the two
optima, which belongs to neither. L-BFGS-B from there follows the local slope
into one of them, not always the higher one, or stops where the target is
flat between them.

## Proposal

1. Optimisation mode: at a fine grid point with more than one candidate,
   evaluate the candidates and start the L-BFGS-B run from the best. The value
   at the start is then known, so the run needs no initial evaluation.
2. Take the candidates from a local test. Look at the coarse cells that
   enclose the fine point: two or four in a 2D projection. If two of them are
   across a jump in the profiled parameters, the candidates are the
   interpolated point and the parameters of each enclosing cell. A jump is
   what patching takes it to be (`_is_jump` in `sampler.py`): a difference in
   some profiled parameter above the median plus three MAD of its differences
   over all pairs of neighbouring coarse cells. Patching and refinement then
   share one definition. The measurements below were made with a fixed
   threshold instead: the largest deviation of any enclosing cell from their
   mean, per profiled parameter, as a fraction of the bound extent, above
   0.05. This is the test `_is_de_skippable` applies to neighbours.
3. Use the same candidates in direct-evaluation mode.
4. Remove the clustering: `cluster_coarse_grid_by_modes` and its helpers in
   `interpolation.py` (about 250 lines), `use_clustering`, the `clustering`
   keys of `advanced_config`, the cluster state on the sampler and the
   `clustering` extra in `pyproject.toml`.

Step 1 does not depend on the others. It can be done with the clustering as
the source of the candidates.

## Measurements

Scans ran on virtual workers (`tools/virtual_pool.py`): one 20 x 20 `'lbfgsb'`
projection over two of four parameters, `grid_refinement_factor` 2, five
seeds. A seed gives the same coarse grid in every variant. In optimisation
mode the candidate step was emulated: a wrapper around
`_get_refined_initialization_candidates` evaluated the candidates and returned
the best one first, and those evaluations were counted separately. The scripts
are not in the repository.

Two targets were built with two optima in one profiled parameter `q`.
Parameters are `(a, b, p, q)` in the unit hypercube, `a` and `b` are the
projection parameters:

    smooth = -((a - 0.5) / 0.15)^2 / 2 - ((b - 0.5) / 0.15)^2 / 2
             - ((p - 0.5 - 0.3 (a - 0.5)) / 0.05)^2 / 2

    straight border:  smooth + logaddexp(-((q - 0.3) / 0.04)^2 / 2,
                                         4 (a - 0.5) - ((q - 0.7) / 0.04)^2 / 2)

    diagonal border:  q1 = 0.3 + 0.2 (b - 0.5),  q2 = 0.7 - 0.2 (a - 0.5)
                      smooth + logaddexp(-((q - q1) / 0.04)^2 / 2,
                                         6 (a - b) - ((q - q2) / 0.04)^2 / 2)

In the first the border lies on a coarse grid line, `a = 0.5`. In the second
it lies on `a = b` and both optima move with the projection parameters. The
true profile was computed by optimising from both optima. A cell counts as
being in an optimum if its `q` is within 0.05 of it. Cells where the two
optima tie count as correct.

### Cells in the wrong place, optimisation mode

Cells inside the region of interest per run, out of about 1000, with
`patch_refined_grid` off (the default):

| Start of the L-BFGS-B run          | Straight border                              | Diagonal border                            |
|------------------------------------|----------------------------------------------|--------------------------------------------|
| Interpolated point (current code)  | 2 between the optima, up to 12 too low       | 11 in the wrong optimum, up to 0.15 too low |
| Best of the clustering candidates  | 0                                            | 0                                          |
| Best of the local-test candidates  | 0                                            | 0                                          |

With `patch_refined_grid=True` all three rows are 0: patching repairs the
cells afterwards. The coarse grid had no cell in the wrong optimum.

### Standard test functions, optimisation mode

On Himmelblau, Rosenbrock and Rastrigin (4D) no cell inside the region of
interest changed its value, for either source of candidates. On Levy one cell
ended 0.09 lower in five runs, and two cells with patching on (up to 0.97).
The candidate with the highest value at the fine point does not always lead to
the highest optimum.

### Cost, optimisation mode

Extra evaluations per run, as a share of the evaluations of the refinement
stage:

| Target           | Clustering candidates | Local test |
|------------------|-----------------------|------------|
| Himmelblau 4D    | 1.8 %                 | 1.2 %      |
| Rosenbrock 4D    | 4.7 %                 | 6.6 %      |
| Rastrigin 4D     | 5.8 %                 | 1.7 %      |
| Levy 4D          | 1.6 %                 | 0.4 %      |
| Straight border  | 2.6 %                 | 1.5 %      |

For comparison, `patch_refined_grid=True` added 14 % on the straight-border
target. Since patching also tests neighbours across a jump it adds about
16 %.

### Direct-evaluation mode

Here the clustering candidates are in use today. Without candidates and
without patching, 37 (straight) and 43 (diagonal) cells per run ended between
the optima, up to 12 and 19 too low. With either source of candidates there
were none.

Summed shortfall against the true profile per run, over the cells inside the
region of interest:

| Candidates  | Straight, no patching | Diagonal, no patching | Straight, patching | Diagonal, patching |
|-------------|-----------------------|-----------------------|--------------------|--------------------|
| None        | 404                   | 337                   | 0.40               | below 1e-10        |
| Clustering  | 0.36                  | 0.33                  | 0.0015             | 0.088              |
| Local test  | 0.39                  | 0.61                  | 0.32               | 0.31               |

The clustering candidates are weighted means over several cells of one
cluster, so they are slightly better estimates than the parameters of a single
enclosing cell. This shows only in direct-evaluation mode, where no
optimisation follows. The largest shortfall of a single cell was 0.019 with
the local test and 0.021 with the clustering.

On the standard functions without patching, both sources improved the same
number of cells over five runs (48 on Himmelblau, 10 on Rosenbrock, 0 on
Rastrigin, 2 on Levy), and the values differed by at most 1.2e-5.

## Limits of the evidence

- Two constructed targets and four standard ones, five seeds, 2D projections,
  refinement factor 2, coarse grid from `'lbfgsb'`.
- The candidate step in optimisation mode was emulated, not implemented.
- The threshold of 0.05 was not varied, and the jump test of step 2 was not
  measured in its place. In the (delta, m32) projection of `octant_6d` its
  thresholds are 0.001 to 0.007 of the bound extent, so it flags smaller
  jumps than 0.05 does, and more fine points. A steep but smooth region can
  pass either test; that costs evaluations, not correctness.
- Both optima must be present among the enclosing coarse cells. A coarse grid
  that has the border in the wrong place is not corrected by the candidates,
  apart from what they happen to reach. Patching of the coarse grid moves
  such a border before refinement starts, as long as both optima are on the
  grid.
- A projection over more than two parameters has up to `2^k` enclosing cells
  for `k` projection parameters. The cost per flagged point grows with it.

## Implementation notes

- The branch to change is the one in `create_refinement_lbfgsb_jobs` that
  takes `candidates[0]`.
- The job needed evaluates several candidates at a cell and starts an
  L-BFGS-B run from the best. `SuspectRecheckJob` did this until the suspect
  recheck was removed (changelog, Unreleased); the git history has it as a
  model. A refinement cell has no population entry yet, so the run the job
  starts must be of type `REFINEMENT_LBFGSB`, with `start_fitness` set to the
  value found. `LBFGSBJob` then skips its initial evaluation. The master
  handles a job started from `on_finish` already.
- Enclosing coarse cells of a fine index `i` along one projection parameter,
  with refinement factor `rf`: `i // rf`, and `i // rf + 1` if `i % rf != 0`.
- `_is_jump` takes differences through `_param_delta`, so two values on
  either side of a periodic bound do not count as a jump. Its thresholds
  belong to the coarse grid: `_jump_thresholds()`, which the first patching
  wave stores. They have to be passed on with the coarse solution, and
  computed there if the coarse grid was not patched.
- In direct-evaluation mode the list of candidates can feed the existing path
  that evaluates several candidates in one `ActivationJob`.
- The two constructed targets are small enough to serve as tests on virtual
  workers, in the manner of `tests/test_patching.py`.
