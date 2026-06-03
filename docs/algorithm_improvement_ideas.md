# paraprof — algorithm improvement ideas

Analysis of the current paraprof algorithm and a set of candidate improvements,
organised around three goals: **sample efficiency**, **correctness guarantees**,
and **adaptability** (auto-tuning settings so the user doesn't have to). All
ideas respect the constraint of *no large computational dependencies* (no GP
emulators, etc.).

## The structural lens

Almost every idea below flows from two facts about profile likelihoods that the
current algorithm only partially exploits.

**Fact 1 — Profiling error is one-sided.** The grid value at a cell is
`Lₚ(θ) = max_φ f(θ, φ)`. Any φ we evaluate gives `f(θ, φ) ≤ Lₚ(θ)`, so **every
cell value is a lower bound on the truth: we can only ever be too low, never too
high.** This turns heuristics into *certificates* — taking the max over
candidate φ's is always safe, and a value deficit relative to what's achievable
is a *proof* of under-optimization. Patching (`create_patching_wave_jobs`) uses
this implicitly but never names it or turns it into a stopping/quality
guarantee.

**Fact 2 — The profiled argmax φ\*(θ) is piecewise-smooth in θ.** Away from
mode-crossings the implicit function theorem gives `dφ*/dθ = −H_φφ⁻¹ H_φθ`. The
current warm-starting is *zeroth-order*: it copies a neighbour's φ
(`ActivationJob.warm_start_params`; the `NEEDS_NEIGHBOR_TEST` step in
`lbfgsb_job.py`). A first-order predictor is nearly free and far better.

The "view the projection as a (sparse) matrix" question is the natural home for
both facts: the grid *is* a discretization of two fields over θ — the value
field `L(i, j)` and the vector field `φ*(i, j)` — and the facts above are
statements about those fields.

---

## A. Sample efficiency

### A1. Predictor–corrector continuation — *tried, benchmarked neutral, reverted*
Idea: a newly-activated cell starts its inner optimization from a neighbour's φ\*
with no use of the local trend of the field. Since φ\*(θ) is piecewise-smooth,
fit its local Jacobian `J = dφ*/dθ` at the source by least squares over all
in-population neighbours of the source and predict

    φ*_target ≈ φ*_source + J · (θ_target − θ_source)    # zero extra evals

instead of `φ*_target ≈ φ*_source`, expecting O(Δθ)→O(Δθ²) warm-start error on
curved ridges. (A direction-free neighbourhood fit was used rather than a single
colinear "grandparent" secant, so the prediction doesn't depend on the
stochastic, MPI-order-dependent activation direction; reduces to the secant with
one neighbour, a robust plane fit with several. Note the L-BFGS history alone
can't supply this: `optimizer_state['s'/'y']` gives only `H_φφ⁻¹`, never the
`H_φθ` cross term, because the inner optimization never varies θ.)

**Outcome (do not re-attempt without a new angle).** Implemented and A/B-tested
on Rosenbrock-4D (6×2D), Himmelblau-4D (6×2D), Rosenbrock-6D (4×1D, 5 profiled
dims), and Rosenbrock-10D (1×2D, 8 profiled dims). The predictor was **neutral
within run-to-run noise everywhere** (occasionally a hair worse), with ROI
quality unchanged — no measurable target-call reduction. **Reverted.**

Why it didn't pay off: paraprof's existing warm-starting already captures the
benefit. The L-BFGS neighbour-test (`lbfgsb_job.py` `NEEDS_NEIGHBOR_TEST`) not
only copies a neighbour's φ\* but also adopts its `(s, y)` **curvature history**,
so the corrector already starts well; a better *seed point* saves at most a
sliver next to the fixed per-cell costs (the `pop_per_grid_point` initial evals
and the per-iteration gradient fan-out). A1 optimised a part of the pipeline that
wasn't the bottleneck. Any future continuation work should target **corrector
iteration count directly** and only where an analytic `grad_func` makes the
`H_φθ` term cheap.

### A2. Analytic profiling of declared Gaussian/quadratic nuisances
`nuisance_wrapper.py` shows the real use case: a few POIs plus many
tightly-constrained nuisances entering as `Σ log L_constraint(nuisance_i)`. When
a profiled dim enters quadratically (Gaussian constraint), the inner max over it
is a **closed-form linear solve**, not a search. If the user declares "these
profiled dims are Gaussian nuisances with curvature C," they drop out of the
numeric inner problem → lower inner dimension → large, compounding efficiency
gain (DE cost scales badly with inner dimension).

### A3. Symmetry priors
Many physics likelihoods have sign/reflection or permutation symmetries. Given a
user-supplied symmetry group (signed-permutation maps), for free:
- generate orbit images of every found optimum → inject into `initial_maxima`
  and the global pool without evaluating;
- restrict the global multistart / grid to a fundamental domain and mirror;
- sample-efficiency multiplier ≈ orbit size.
Hook on `register_initial_optimum` / `_update_global_pool`.

### A4. Adaptive mesh refinement toward contours + adaptive per-cell budget
- **AMR:** the deliverable is the 68/95% CL contour. Current refinement
  multiplies the whole grid uniformly. Instead refine where the contour passes
  and where the value-Laplacian is large. Extend the existing refinement
  pre-screening predicate (`create_refinement_lbfgsb_jobs`).
- **Per-cell budget:** `pop_per_grid_point` and `lbfgsb_max_iter` are global.
  Allocate by predicted difficulty (predictor residual + neighbour-φ
  disagreement): smooth interior cells get ~1 polish, hard cells near
  mode-crossings get more. Self-tuning.

---

## B. Correctness guarantees (and the matrix view)

### B1. Consistency residual as a one-sided certificate + adaptive stop
Define per cell `r(cell) = max over candidate φ's [ f(θ_cell, φ) ] − L_current`,
candidates = neighbour φ\*'s (and pool/symmetry images). By Fact 1, `r > 0` is a
proof the cell is under-optimized, and applying the max only ever improves
correctness. This gives:
- a **principled self-tuning stop**: iterate patching/recheck until
  `max r < tol` (replaces `max_patching_waves` guesswork);
- a **reported quality metric**: "max residual = X, mean = Y" is an honest,
  computable bound on remaining one-sided error. paraprof currently can't tell
  the user how converged the grid is.

### B2. Sparse-matrix structure → failure & missed-mode detection
Treat the ROI mask as a sparse binary image:
- **Holes = failures.** A sub-threshold cell fully surrounded by ROI cells is,
  by Fact 1, almost certainly an optimization failure (a real dip would be
  broad, not a pinhole). A morphological hole detector is a near-zero
  false-positive failure finder — cleaner than the current MAD-on-φ test.
  Nuance: a φ jump at a *real* mode-crossing kink is **not** an error, so the
  value-deficit criterion (B1) is strictly safer than `_find_suspect_cells`'s
  φ-discontinuity test; lead with value, keep φ as a secondary signal.
- **Connected components = modes visible in this projection.** Compare to a
  known/expected count (C1) to detect a *missed* disconnected ROI island —
  dynamic activation only expands contiguously, so a separated mode is found
  only if the initial stage seeded it.

### B3. Cross-projection consistency (free correctness amplifier)
paraprof runs many projections of the *same* function and keeps a *full-D* global
pool. The 1-D profile of x₀ from the (x₀,x₁) projection must equal the one from
(x₀,x₂) — both equal the true 1-D profile. In practice both are lower bounds, so
**take the larger (Fact 1) and inject the full-D point that achieved it into the
lagging projection's pool**, where proximity warm-start / patching repairs the
corresponding cell. Nothing currently does cross-projection *verification* — only
knowledge transfer.

---

## C. Priors / adaptability

### C1. Known mode count → basin detection
Hook on `basin_detection_should_stop` / `basin_detection_roi_stats`. Upper bound:
stop the rolling multistart the moment `W` distinct ROI optima are registered.
Lower bound / exact count: refuse to stop early until `W ≥ known_min`. Or fold
the count in as a Bayesian prior on the Boender–Rinnooy-Kan estimator.

### C2. Periodic boundaries → toroidal grid
For angular/phase parameters. Localized changes:
- `_get_valid_neighbors`: wrap instead of clip — opposite edges become
  neighbours, so ROI/patching/suspect rings flow across the seam.
- `_ensure_bounds`: wrap periodic dims (mod) rather than clip — clipping a
  periodic param builds a *false wall* that traps the inner optimizer.
- Grid construction: for a periodic dim, don't duplicate the wrap point.
Expose as a per-dimension `periodic` flag.

### C3. Other cheap priors
- **Unimodality/monotonicity** in a projection dim: stop expanding the ROI past a
  turning point (opt-in; harmful if wrong).
- **User "ridge guess" `φ̂(θ)`**: analytic approximation of where the optimum
  lives, used as an extra warm-start seed — like `initial_points` but as a
  function.
- **Low-rank/smooth matrix completion** of the value field — *only* to prioritize
  activation order or pre-screen the contour, **never to replace an
  evaluation**, which keeps Fact 1 intact and stays far from the GP weight class.

---

## Prioritized shortlist

| Idea | Axis | Payoff | Effort | Risk |
|---|---|---|---|---|
| ~~**A1** Predictor–corrector warm starts~~ | efficiency | ~~High~~ tried → neutral, reverted | Med | Low |
| **B1** Consistency residual → adaptive stop + quality bound | correctness + adapt | High | Low–Med | Low |
| **B3** Cross-projection consistency + pool repair | correctness | High | Med | Low |
| **C1** Known mode-count prior | adapt | Med | Low | Low |
| **C2** Periodic boundaries | correctness + efficiency | Med–High* | Med | Low |
| **B2** Matrix holes/components detection | correctness | Med | Low | Low |
| **A2** Analytic Gaussian-nuisance profiling | efficiency | High* | Med | Low |
| **A4** AMR toward contours + adaptive budget | efficiency + adapt | Med–High | Med–High | Med |
| **A3** Symmetry priors | efficiency | High* | Med | Low |

\* large payoff but only for targets/users that have the structure.

**Recommended starting order:** A1 + B1 + C1 (highest value-to-effort, no new
deps), then B3 (most distinctive — leverages paraprof's multi-projection +
global-pool design in a way nothing else does).

Two cross-cutting notes: (1) lead correctness work with the value-deficit
(one-sided) criterion rather than φ-discontinuity, since a φ jump at a real
mode-crossing is not a bug; (2) keep matrix completion strictly as a
prioritization/pre-screen signal, never a substitute for a real evaluation.
