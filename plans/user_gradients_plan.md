# Plan: Support user-supplied gradient functions in paraprof

## Motivation

Paraprof is mainly intended for profiling numerical functions where gradients
are not available, and it therefore estimates gradients via finite differences
(FD) as part of the L-BFGS-B algorithm. In some cases the user does have access
to functions that provide all or some of the target-function gradients. In
those cases paraprof should use them to reduce the number of target-function
calls required to obtain gradients via FD.

This plan covers the design, integration points, edge cases, tests, and docs
for that feature. Differential Evolution is gradient-free and out of scope —
this feature only affects the L-BFGS-B paths.

## Where the current FD logic lives

- `src/paraprof/jobs/lbfgsb_job.py`
  - `_calculate_gradient_tasks`: issues N (forward) or 2N (central) MPI tasks
    per gradient evaluation.
  - `_process_gradient_result`: assembles components into a gradient, then
    triggers the two-loop recursion and a line-search task.
  - Reused by `INITIAL_OPTIMIZATION`, `LBFGSB`, `LBFGSB_LOOP`,
    `POST_ACTIVATION_LBFGSB`, `PATCHING_LBFGSB`, `REFINEMENT_LBFGSB`.
- `src/paraprof/worker.py`: only ever calls `target_func(params)`.
- `src/paraprof/master.py`: broadcasts `sampler.target_func` to workers in
  `run_scan` / `master_main`.
- `src/paraprof/sampler.py`: holds `target_func`, `lbfgsb_gradient_method`,
  `target_calls` / `target_call_errors` counters, `_register_target_call`.

## What to design and implement

### 1. New constructor argument

Add a new constructor argument on `ProfileProjector` for the user gradient,
e.g. `grad_func=None` (default: `None`, preserving today's behaviour).

Think carefully about the cleanest API. Suggested shape:

- `grad_func(params) -> np.ndarray | dict`
  - Full: a length-`n_dims` array of finite floats (sign convention: gradient
    of the function being **maximized**, i.e. `∇target_func`).
  - Partial via array: same shape, with `np.nan` in positions the user does
    not provide.
  - Partial via dict: `{dim_index: value}` for known components only.
- Accept both forms. Validate shape/keys/finiteness at the call site and
  produce a clear error if the user returns garbage.
- Document the sign convention explicitly: user returns `∇(target_func)`,
  paraprof internally negates for the minimization objective.

### 2. Broadcast `grad_func` to workers

Mirror the existing `target_func` plumbing.

- Extend `worker_main` to optionally receive `grad_func` from the master
  broadcast (or accept it as a pre-supplied kwarg, like `target_func`).
- Keep the broadcast contract backwards-compatible: when `grad_func` is
  `None`, do not break existing host integrations (e.g. `GAMBIT_plugin`) that
  supply `target_func` directly.

### 3. Extend the worker task protocol

Let the master request a gradient alongside a target evaluation. Add a
context flag like `compute_gradient=True`. On `True`, after computing
`target_val` the worker also calls `grad_func(params)` and ships back:

- The gradient array (length `n_dims`, `np.nan` for unknown components), or
  a normalized form of whatever the user returned.
- An `error` field if `grad_func` raises (coerce to all-nan gradient and
  record the error; do not crash the run).

### 4. Rework `LBFGSBJob` to use user gradients with FD fallback

Sketch:

- Replace the current "issue N/2N FD tasks at x" step with a single
  value+gradient task at x (worker returns `f(x)` and the partial gradient).
- On result: slice gradient to `opt_dims`; sign-flip for the minimization
  objective.
- For dims still `nan` after slicing, issue forward/central FD tasks for
  ONLY those dims (reuse the existing FD code paths; do not duplicate them).
- If `grad_func` is `None` or returned nothing usable, fall through to the
  current pure-FD path unchanged.

**Important optimization**: the line-search task already evaluates `f(x_new)`.
Make that task ALSO carry `compute_gradient=True` so that after acceptance we
have `f` and `∇f` at `x_new` in a single round trip, and only need FD tasks
for the dims the user didn't cover. Mirror the same trick for the
`NEEDS_INITIAL_F` and `NEEDS_NEIGHBOR_TEST` stages where it's a win.

### 5. Bookkeeping

Add a counter (e.g. `sampler.target_calls_saved_by_user_gradient`) that
tracks how many FD target-calls were avoided versus the pure-FD baseline,
and surface it in the end-of-run summary log next to `target_calls` /
`target_call_errors`. This is the headline metric for whether the feature
is doing its job.

### 6. Edge cases

- User returns a gradient with non-finite entries in a dim they "claim" to
  provide → treat that dim as missing, log a warning the first few times.
- User's gradient covers projection dims (which are FIXED during profiling) —
  those components are simply unused, do not error.
- `opt_dims` is empty (already a degenerate path) — short-circuit before
  requesting a gradient.
- `n_opt_dims == 1` with user providing that one component → no FD tasks
  issued at all, straight to line search.
- `grad_func` raises on a worker → coerce to all-nan, increment a dedicated
  error counter, fall back to FD; do not abort the job.

### 7. Sign / units sanity

Write a short docstring note plus an internal assertion in debug mode that
the user-supplied gradient matches the FD gradient to within a loose tolerance
on a couple of reference evals during the very first L-BFGS-B job of a run.
Make this opt-in via an advanced_config flag like
`advanced_config['lbfgsb']['validate_user_gradient'] = True` so it does not
add cost in production runs.

### 8. `advanced_config` updates

- Keep `gradient_method` (`'forward'` | `'central'`) as the FD fallback mode
  used for any missing components. Do NOT introduce a `'user'` value — user
  gradients are an orthogonal feature controlled by the presence of
  `grad_func`.
- Add the new `validate_user_gradient` toggle described above.

## Testing

New tests in `tests/test_integration.py` (or a new `test_user_gradients.py`):

- Full analytic gradient (e.g. `sphere` or `rosenbrock_nd`, both of which have
  closed-form gradients): assert that with `grad_func` set, `target_calls`
  drops by approximately the expected savings relative to a pure-FD baseline
  run on the same projection seed.
- Partial gradient: provide gradient for dims `[0, 2]` of a 4D function,
  assert FD tasks fire ONLY for the remaining dims.
- `grad_func` returns dict form vs. array form: both work.
- `grad_func` raises: run completes, error counter increments, results are
  still numerically close to FD baseline.
- `grad_func` returns a shape-wrong array: clean `ConfigurationError` (or
  equivalent) at construction / first call.

Existing tests must still pass without modification (`grad_func=None` is a
true no-op).

## Docs

- Update the constructor docstring of `ProfileProjector` with the new argument
  and the sign convention.
- Add a short "User-supplied gradients" subsection to `README.md` under
  "Configuration", documenting the array-with-nan and dict forms, noting that
  paraprof maximizes (so grad is `∇target_func`), and that any missing
  components are filled in by finite differences using the configured
  `gradient_method`.
- Mention in `CHANGELOG.md`.

## Out of scope

- Hessian / second-derivative support.
- Vectorized batch gradient calls.
- Any DE changes (DE does not use gradients).

## Design notes to think through before coding

- API shape: array-with-nan vs. dict vs. both. Both is more user-friendly but
  costs a small amount of normalization code on the worker / master side.
- Whether to fold gradient requests into existing tasks (line-search,
  initial-f, neighbor-test) vs. always issuing a dedicated value+gradient
  task. The former saves a round trip per L-BFGS iteration and is the main
  place the savings come from.
- Where to normalize the user's return value (worker side, so the master sees
  a uniform array-with-nan shape) vs. master side (so the worker code stays
  minimal). Worker-side normalization keeps the wire format uniform.
