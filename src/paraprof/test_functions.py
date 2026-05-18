"""Benchmark test functions (negated for maximization; global max value 0
when known).

Categories: unimodal (Sphere, Rosenbrock); multimodal with few peaks
(Himmelblau, Beale, Eggholder); multimodal with many regular peaks
(Rastrigin, Ackley, Griewank); multimodal with steep/rugged landscapes
(Michalewicz, Styblinski-Tang, Levy, Schwefel).

Each ``*_nd`` implementation works in any supported dimension; the
``*_2d/4d/6d/10d`` aliases are kept for convenience.

References: Jamil & Yang (2013), Int. J. Math. Modelling Numerical
Optimisation 4(2), 150-194; Momin & Yang (2013), same journal.
"""
import numpy as np


# --- Unimodal ---

def sphere_nd(params):
    """N-dimensional Sphere function (negated for maximization).

    Unimodal, convex, separable. Global optimum f(0, ..., 0) = 0 on [-5, 5]^n.
    """
    return -np.sum(params**2)


def rosenbrock_nd(params):
    """N-dimensional Rosenbrock function (negated for maximization).

    Unimodal but non-convex with a narrow curved valley. Global optimum
    f(1, ..., 1) = 0 on [-6, 6]^n.
    """
    return -0.1 * np.sum(
        100.0 * (params[1:] - params[:-1]**2.0)**2.0
        + (1 - params[:-1])**2.0
    )


# --- Multimodal: few peaks ---

def himmelblau_4d(params):
    """4D Himmelblau function (negated and scaled for maximization).

    Two independent copies of the 2D Himmelblau function. Has four equal-height
    global optima per copy, so 16 peaks total on [-6, 6]^4.
    """
    x1, x2, x3, x4 = params
    term1 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    term2 = (x3**2 + x4 - 11)**2 + (x3 + x4**2 - 7)**2
    scale = 0.05
    return -1 * scale * (term1 + term2)


def beale_2d(params):
    """2D Beale function (negated for maximization).

    Global optimum f(3, 0.5) = 0 on [-4.5, 4.5]^2.
    """
    x, y = params
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y**2)**2
    term3 = (2.625 - x + x*y**3)**2
    return -(term1 + term2 + term3)


# Original-form minimum value of the 2D Eggholder, used to shift the
# generalized N-D extension so the global optimum is at 0.
_EGGHOLDER_2D_MIN = -959.6407


def eggholder_nd(params):
    """N-dimensional Eggholder function (negated and shifted for maximization).

    Defined for even dimensions as the sum of independent 2D Eggholder
    sub-functions over (x_{2k}, x_{2k+1}) pairs. Highly multimodal on
    [-512, 512]^n. Global optimum value is 0 at (512, 404.2319, ...).
    """
    params = np.asarray(params)
    if params.size % 2 != 0:
        raise ValueError(
            f"eggholder_nd requires an even-dimensional input "
            f"(got {params.size})."
        )
    f_original = 0.0
    for k in range(0, params.size, 2):
        x = params[k]
        y = params[k + 1]
        f_original += -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47))))
        f_original += -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    f_min = (params.size // 2) * _EGGHOLDER_2D_MIN
    return -(f_original - f_min)


# --- Multimodal: many regular peaks ---

def rastrigin_nd(params):
    """N-dimensional Rastrigin function (negated for maximization).

    Highly multimodal, separable. Global optimum f(0, ..., 0) = 0 on
    [-5.12, 5.12]^n.
    """
    A = 10
    n = len(params)
    return -(A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params)))


def ackley_nd(params):
    """N-dimensional Ackley function (negated for maximization).

    Nearly flat outer region with a sharp central peak. Global optimum
    f(0, ..., 0) = 0 on [-5, 5]^n.
    """
    n = len(params)
    sum_sq = np.sum(params**2)
    sum_cos = np.sum(np.cos(2 * np.pi * params))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    f_original = term1 + term2 + 20 + np.e
    return -f_original


def griewank_nd(params):
    """N-dimensional Griewank function (negated for maximization).

    Multimodal with many local optima; tends to be easier in higher dimensions.
    Global optimum f(0, ..., 0) = 0 on [-100, 100]^n.
    """
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    return -(sum_sq / 4000 - prod_cos + 1)


# --- Multimodal: steep / rugged landscape ---

# Approximate Michalewicz minima (m=10) by dimension, used as a shift so the
# global optimum is at 0. Dimensions outside this table are returned unshifted.
_MICHALEWICZ_MIN = {2: -1.8013, 4: -3.72, 6: -5.69, 10: -9.66}


def michalewicz_nd(params):
    """N-dimensional Michalewicz function (negated and shifted for maximization).

    Multimodal with steep valleys and ridges, non-separable. Domain [0, π]^n.
    The shift uses tabulated minima for dim ∈ {2, 4, 6, 10}; other dimensions
    return the unshifted negated value.
    """
    m = 10
    n = len(params)
    f_original = -np.sum(
        np.sin(params)
        * np.sin((np.arange(1, n + 1) * params**2) / np.pi)**(2*m)
    )
    f_min = _MICHALEWICZ_MIN.get(n, 0.0)
    return -(f_original - f_min)


def styblinski_tang_nd(params):
    """N-dimensional Styblinski-Tang function (negated and shifted for maximization).

    Separable, multimodal. Global optima at (-2.903534, ..., -2.903534) with
    value 0 on [-5, 5]^n.
    """
    n = len(params)
    f_original = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)
    f_min = -39.16617 * n
    return -(f_original - f_min)


def levy_nd(params):
    """N-dimensional Levy function (negated for maximization).

    Wave-like, non-separable. Global optimum f(1, ..., 1) = 0 on [-10, 10]^n.
    """
    w = 1 + (params - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return -(term1 + term2 + term3)


def schwefel_nd(params):
    """N-dimensional Schwefel function (negated and shifted for maximization).

    Highly multimodal and deceptive: the global optimum lies far from local
    optima. Global optimum f(420.9687, ..., 420.9687) = 0 on [-500, 500]^n.
    """
    n = len(params)
    f_original = 418.9829 * n - np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return -f_original


# --- Per-dimension aliases (the *_nd implementations dispatch on len(params)) ---

sphere_2d = sphere_4d = sphere_6d = sphere_10d = sphere_nd
rosenbrock_2d = rosenbrock_4d = rosenbrock_6d = rosenbrock_10d = rosenbrock_nd
eggholder_2d = eggholder_4d = eggholder_6d = eggholder_nd
rastrigin_2d = rastrigin_4d = rastrigin_6d = rastrigin_10d = rastrigin_nd
ackley_2d = ackley_4d = ackley_6d = ackley_10d = ackley_nd
griewank_2d = griewank_4d = griewank_6d = griewank_10d = griewank_nd
michalewicz_2d = michalewicz_4d = michalewicz_6d = michalewicz_10d = michalewicz_nd
styblinski_tang_2d = styblinski_tang_4d = styblinski_tang_nd
styblinski_tang_6d = styblinski_tang_10d = styblinski_tang_nd
levy_2d = levy_4d = levy_6d = levy_10d = levy_nd
schwefel_2d = schwefel_4d = schwefel_6d = schwefel_10d = schwefel_nd


# --- Factory function ---

def _make_uniform_registry(prefix, func, bounds, peak_value, dims=(2, 4, 6, 10)):
    """Registry entries for a function family with uniform bounds and a single peak."""
    return {
        f"{prefix}_{n}d": (func, bounds, [np.full(n, peak_value)])
        for n in dims
    }


_FUNCTION_REGISTRY = {
    # Sphere
    **_make_uniform_registry("sphere", sphere_nd, [-5, 5], 0.0),

    # Rosenbrock
    **_make_uniform_registry("rosenbrock", rosenbrock_nd, [-6, 6], 1.0),

    # Himmelblau (single 4D variant, four peaks)
    "himmelblau_4d": (himmelblau_4d, [-6, 6], [
        np.array([3.0, 2.0, 3.0, 2.0]),
        np.array([-2.805118, 3.131312, -2.805118, 3.131312]),
        np.array([-3.779310, -3.283186, -3.779310, -3.283186]),
        np.array([3.584428, -1.848126, 3.584428, -1.848126]),
    ]),

    # Beale (single 2D variant)
    "beale_2d": (beale_2d, [-4.5, 4.5], [np.array([3.0, 0.5])]),

    # Eggholder (even dimensions; peak at (512, 404.2319) for each pair)
    "eggholder_2d": (eggholder_nd, [-512, 512],
                     [np.array([512.0, 404.2319])]),
    "eggholder_4d": (eggholder_nd, [-512, 512],
                     [np.array([512.0, 404.2319, 512.0, 404.2319])]),
    "eggholder_6d": (eggholder_nd, [-512, 512],
                     [np.array([512.0, 404.2319] * 3)]),

    # Rastrigin
    **_make_uniform_registry("rastrigin", rastrigin_nd, [-5.12, 5.12], 0.0),

    # Ackley
    **_make_uniform_registry("ackley", ackley_nd, [-5, 5], 0.0),

    # Griewank
    **_make_uniform_registry("griewank", griewank_nd, [-100, 100], 0.0),

    # Michalewicz (peaks not exactly known)
    "michalewicz_2d": (michalewicz_nd, [0, np.pi], []),
    "michalewicz_4d": (michalewicz_nd, [0, np.pi], []),
    "michalewicz_6d": (michalewicz_nd, [0, np.pi], []),
    "michalewicz_10d": (michalewicz_nd, [0, np.pi], []),

    # Styblinski-Tang
    **_make_uniform_registry("styblinski_tang", styblinski_tang_nd, [-5, 5],
                             -2.903534),

    # Levy
    **_make_uniform_registry("levy", levy_nd, [-10, 10], 1.0),

    # Schwefel
    **_make_uniform_registry("schwefel", schwefel_nd, [-500, 500], 420.9687),
}


def list_test_functions():
    """Return a sorted list of registered test-function names."""
    return sorted(_FUNCTION_REGISTRY.keys())


def get_test_function(name):
    """Look up a registered test function by name.

    Returns ``(func, bounds, peaks)``. ``name`` is ``'<function>_Nd'``
    (e.g. ``'sphere_2d'``, ``'eggholder_6d'``). ``peaks`` is empty when
    the peak locations are not precisely known.
    """
    if name not in _FUNCTION_REGISTRY:
        available = list_test_functions()
        raise ValueError(
            f"Unknown test function: '{name}'\n"
            f"Available functions:\n" +
            "\n".join(f"  - {fn}" for fn in available)
        )

    func, bounds_per_dim, peaks = _FUNCTION_REGISTRY[name]

    if peaks:
        n_dims = len(peaks[0])
    else:
        n_dims = int(name.split('_')[-1].replace('d', ''))

    bounds = [bounds_per_dim] * n_dims

    return func, bounds, peaks
