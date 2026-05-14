"""
Local quadratic surrogate for DE trial pre-screening.

A cheap, closed-form local-quadratic model fit by least squares against the
cached (profiled_params, target_val) samples in a grid cell's neighbourhood,
used to pre-screen DE trial vectors and send only the most promising ones to
the workers for full target-function evaluation. Fit and predict are pure
numpy; no hyperparameter tuning, no iterative solvers.
"""
import numpy as np


def quadratic_basis_size(n_dims):
    """Number of coefficients in a full quadratic basis in n_dims dimensions:
    1 (const) + n_dims (linear) + n_dims*(n_dims+1)/2 (quadratic incl. cross)."""
    return 1 + n_dims + n_dims * (n_dims + 1) // 2


def build_quadratic_basis(X):
    """Build the design matrix for a full quadratic in n_dims dimensions.

    Columns are ordered: 1, x_0, ..., x_{D-1}, x_0^2, x_0*x_1, ..., x_{D-1}^2.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_dims)
    """
    n, d = X.shape
    p = quadratic_basis_size(d)
    phi = np.empty((n, p))
    phi[:, 0] = 1.0
    phi[:, 1:1 + d] = X
    col = 1 + d
    for i in range(d):
        for j in range(i, d):
            phi[:, col] = X[:, i] * X[:, j]
            col += 1
    return phi


def fit_local_quadratic(X, y, cond_max=1e10, min_samples_factor=2.0):
    """Fit a local quadratic by least squares.

    Returns ``None`` if the design matrix is too ill-conditioned to trust
    or there are too few samples. Otherwise returns a dict with keys
    'coeffs', 'center', 'scale' (used for column rescaling that improves
    conditioning), 'condition', and 'rmse'.

    Parameters
    ----------
    X : np.ndarray, shape (n, D)
    y : np.ndarray, shape (n,)
    cond_max : float
        If the condition number of the (rescaled) design matrix exceeds this,
        return None and let the caller fall back to no-prescreening.
    min_samples_factor : float
        Require ``n >= min_samples_factor * basis_size`` samples for the
        fit. The default of 2.0 keeps the surrogate conservative on stiff
        problems: a bare-minimum fit (factor 1.0) can interpolate the data
        exactly while extrapolating wildly for trial candidates.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape

    p = quadratic_basis_size(d)
    if n < int(np.ceil(min_samples_factor * p)):
        return None

    # Center and rescale the input columns to improve conditioning. The
    # surrogate is only used to rank candidates relative to each other, so a
    # linear change of variables is harmless.
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

    # In-sample RMSE and R² of the fit. Callers use rmse as a
    # self-calibrating "is this prediction trustworthy enough to skip the
    # eval?" margin, and r_squared as a gate: when the local quadratic
    # explains very little of the variance, the surface is too non-quadratic
    # to trust for skip decisions and the caller should fall back to a
    # safer best-of-K selection.
    resid = y - phi @ coeffs
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(ss_res / n))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    return {
        'coeffs': coeffs,
        'center': center,
        'scale': scale,
        'condition': cond,
        'rmse': rmse,
        'r_squared': r_squared,
    }


def predict_local_quadratic(model, X):
    """Predict target values at the rows of X using a fitted local quadratic."""
    if model is None:
        return None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xs = (X - model['center']) / model['scale']
    phi = build_quadratic_basis(Xs)
    return phi @ model['coeffs']


def select_top_k_by_prediction(candidates, predicted_values, k):
    """Return indices of the k candidates with the highest predicted target values.

    Stable sort, so ties resolve in the candidate's original order — that
    preserves DE's stochastic candidate stream when the surrogate is flat.
    """
    k = min(int(k), len(candidates))
    if k <= 0:
        return np.empty(0, dtype=int)
    # argpartition is O(n) but unordered; we need a stable ranking only among
    # the top k, so do a full stable argsort. n is tiny (≤ a few dozen).
    order = np.argsort(-np.asarray(predicted_values), kind='stable')
    return order[:k]
