"""
Local quadratic surrogate for DE trial pre-screening.

A closed-form local-quadratic fit by least squares on cached
(profiled_params, target_val) samples from a grid cell's neighbourhood,
used by ``DEGridPointJob`` to skip the worker evaluation of trial vectors
the surrogate confidently predicts will be worse than the slot occupant.
Fit and predict are pure numpy; no hyperparameter tuning, no iterative
solvers, no per-fit work that scales worse than O(n*p^2 + p^3) where
p = basis size and n = local sample count.
"""
import numpy as np


def quadratic_basis_size(n_dims):
    """1 (const) + n_dims (linear) + n_dims*(n_dims+1)/2 (quadratic incl. cross)."""
    return 1 + n_dims + n_dims * (n_dims + 1) // 2


def build_quadratic_basis(X):
    """Design matrix columns: 1, x_0..x_{D-1}, x_0^2, x_0*x_1, ..., x_{D-1}^2."""
    n, d = X.shape
    phi = np.empty((n, quadratic_basis_size(d)))
    phi[:, 0] = 1.0
    phi[:, 1:1 + d] = X
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            phi[:, col] = X[:, i] * X[:, j]
            col += 1
    return phi


def fit_local_quadratic(X, y, cond_max=1e10, min_samples_factor=2.0):
    """Fit a local quadratic by least squares. Returns ``None`` if the
    design is too ill-conditioned, there are too few samples, or the
    fit cannot be trusted. Otherwise a dict with 'coeffs', 'center',
    'scale', 'condition', 'rmse', 'r_squared'. ``min_samples_factor=2.0``
    keeps the fit from being a bare interpolant of its training points."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    p = quadratic_basis_size(d)
    if n < int(np.ceil(min_samples_factor * p)):
        return None

    # Centre/rescale columns — a linear change of variables that improves
    # conditioning. The surrogate is used only for ranking, so this is
    # harmless.
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = (X - center) / scale

    phi = build_quadratic_basis(Xs)
    coeffs, _, _, sv = np.linalg.lstsq(phi, y, rcond=None)
    if sv.size == 0 or sv[-1] <= 0.0:
        return None
    cond = float(sv[0] / sv[-1])
    if not np.isfinite(cond) or cond > cond_max:
        return None

    resid = y - phi @ coeffs
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        'coeffs': coeffs,
        'center': center,
        'scale': scale,
        'condition': cond,
        'rmse': float(np.sqrt(ss_res / n)),
        'r_squared': 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0,
    }


def predict_local_quadratic(model, X):
    """Predict target values at the rows of X using a fitted local quadratic."""
    if model is None:
        return None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xs = (X - model['center']) / model['scale']
    return build_quadratic_basis(Xs) @ model['coeffs']
