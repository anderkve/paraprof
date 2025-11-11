"""
Test functions for benchmarking the optimization algorithm.

This module provides a comprehensive suite of test functions with diverse
characteristics for testing profile likelihood computations. All functions
are negated for maximization and shifted so that the global optimum has
a value of 0.0 (when the optimum value is known).

Function Categories
-------------------
- Unimodal: Sphere, Rosenbrock
- Multimodal (few peaks): Himmelblau, Beale, Eggholder
- Multimodal (many regular peaks): Rastrigin, Ackley, Griewank
- Multimodal (steep/rugged): Michalewicz, Styblinski-Tang, Levy, Schwefel

Dimensionality
--------------
Functions are provided in multiple dimensionalities:
- 2D: Good for visualization and basic testing
- 4D: Standard testing dimension
- 6D: Medium-dimensional testing
- 10D: High-dimensional testing

References
----------
- Jamil, M. & Yang, X.-S. (2013). A literature survey of benchmark functions
  for global optimization problems. Int. J. Mathematical Modelling and
  Numerical Optimisation, 4(2), 150-194.
- Momin, J. & Yang, X.-S. (2013). A literature survey of benchmark functions
  for global optimization. Journal of Mathematical Modelling and Numerical
  Optimisation, 4(2), 150-194.
"""
import numpy as np


# =============================================================================
# UNIMODAL FUNCTIONS
# =============================================================================

def sphere_2d(params):
    """
    2D Sphere function (negated for maximization).

    Characteristics:
    - Unimodal, convex, separable
    - Simplest benchmark function
    - Tests basic optimization capability

    Global optimum: f(0, 0) = 0.0
    Domain: [-5, 5]^2
    """
    return -np.sum(params**2)


def sphere_4d(params):
    """
    4D Sphere function (negated for maximization).

    Characteristics:
    - Unimodal, convex, separable
    - Simplest benchmark function
    - Tests basic optimization capability

    Global optimum: f(0, 0, 0, 0) = 0.0
    Domain: [-5, 5]^4
    """
    return -np.sum(params**2)


def sphere_6d(params):
    """
    6D Sphere function (negated for maximization).

    Characteristics:
    - Unimodal, convex, separable
    - Simplest benchmark function
    - Tests basic optimization capability

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5, 5]^6
    """
    return -np.sum(params**2)


def sphere_10d(params):
    """
    10D Sphere function (negated for maximization).

    Characteristics:
    - Unimodal, convex, separable
    - Simplest benchmark function
    - Tests basic optimization capability

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5, 5]^10
    """
    return -np.sum(params**2)


def rosenbrock_2d(params):
    """
    2D Rosenbrock function (negated for maximization).

    Characteristics:
    - Unimodal, non-convex, non-separable
    - Narrow valley from local minimum to global minimum
    - Classic test of optimization in valleys

    Global optimum: f(1, 1) = 0.0
    Domain: [-6, 6]^2
    """
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)


def rosenbrock_4d(params):
    """
    4D Rosenbrock function (negated for maximization).

    Characteristics:
    - Unimodal, non-convex, non-separable
    - Narrow valley from local minimum to global minimum
    - Classic test of optimization in valleys

    Global optimum: f(1, 1, 1, 1) = 0.0
    Domain: [-6, 6]^4
    """
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)


def rosenbrock_6d(params):
    """
    6D Rosenbrock function (negated for maximization).

    Characteristics:
    - Unimodal, non-convex, non-separable
    - Narrow valley from local minimum to global minimum
    - Classic test of optimization in valleys

    Global optimum: f(1, 1, 1, 1, 1, 1) = 0.0
    Domain: [-6, 6]^6
    """
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)


def rosenbrock_10d(params):
    """
    10D Rosenbrock function (negated for maximization).

    Characteristics:
    - Unimodal, non-convex, non-separable
    - Narrow valley from local minimum to global minimum
    - Classic test of optimization in valleys

    Global optimum: f(1, 1, ..., 1) = 0.0
    Domain: [-6, 6]^10
    """
    return -0.1 * np.sum(100.0 * (params[1:] - params[:-1]**2.0)**2.0 + (1 - params[:-1])**2.0)


# =============================================================================
# MULTIMODAL FUNCTIONS - FEW PEAKS
# =============================================================================

def himmelblau_4d(params):
    """
    4D Himmelblau function (negated and scaled for maximization).

    Characteristics:
    - Multimodal with 4 known peaks
    - Combination of two 2D Himmelblau functions
    - Tests finding multiple optima

    Global optima: 4 peaks at equal height
    Domain: [-6, 6]^4
    """
    x1, x2, x3, x4 = params
    term1 = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    term2 = (x3**2 + x4 - 11)**2 + (x3 + x4**2 - 7)**2
    scale = 0.05
    return -1 * scale * (term1 + term2)


def beale_2d(params):
    """
    2D Beale function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with steep valleys
    - Unimodal in practice (one main minimum)
    - Tests optimization in narrow valleys

    Global optimum: f(3, 0.5) = 0.0
    Domain: [-4.5, 4.5]^2

    Reference: Beale, E. M. L. (1958). On an iterative method for finding
    a local minimum of a function of more than one variable.
    """
    x, y = params
    term1 = (1.5 - x + x*y)**2
    term2 = (2.25 - x + x*y**2)**2
    term3 = (2.625 - x + x*y**3)**2
    f_original = term1 + term2 + term3
    # Minimum value is 0 at (3, 0.5)
    return -f_original


def eggholder_2d(params):
    """
    2D Eggholder function (negated and shifted for maximization).

    Characteristics:
    - Highly multimodal with many local optima
    - Asymmetric, rugged landscape
    - Very challenging for optimization
    - Tests exploration capability

    Global optimum: f(512, 404.2319) = 0.0
    Original minimum: -959.6407
    Domain: [-512, 512]^2

    Reference: Test functions for optimization (Wikipedia)
    """
    x, y = params
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    f_original = term1 + term2
    f_min = -959.6407  # Known minimum value
    return -(f_original - f_min)


def eggholder_4d(params):
    """
    4D Eggholder function (negated and shifted for maximization).

    Characteristics:
    - Extension using two pairs of variables
    - Highly multimodal with many local optima
    - Asymmetric, rugged landscape

    Global optimum: f(512, 404.2319, 512, 404.2319) = 0.0
    Domain: [-512, 512]^4
    """
    x1, y1, x2, y2 = params
    term1 = -(y1 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (y1 + 47))))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (y1 + 47))))
    term3 = -(y2 + 47) * np.sin(np.sqrt(np.abs(x2/2 + (y2 + 47))))
    term4 = -x2 * np.sin(np.sqrt(np.abs(x2 - (y2 + 47))))
    f_original = term1 + term2 + term3 + term4
    f_min = 2 * (-959.6407)  # Two instances
    return -(f_original - f_min)


def eggholder_6d(params):
    """
    6D Eggholder function (negated and shifted for maximization).

    Characteristics:
    - Extension using three pairs of variables
    - Highly multimodal with many local optima
    - Asymmetric, rugged landscape

    Global optimum: f(512, 404.2319, 512, 404.2319, 512, 404.2319) = 0.0
    Domain: [-512, 512]^6
    """
    x1, y1, x2, y2, x3, y3 = params
    term1 = -(y1 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (y1 + 47))))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (y1 + 47))))
    term3 = -(y2 + 47) * np.sin(np.sqrt(np.abs(x2/2 + (y2 + 47))))
    term4 = -x2 * np.sin(np.sqrt(np.abs(x2 - (y2 + 47))))
    term5 = -(y3 + 47) * np.sin(np.sqrt(np.abs(x3/2 + (y3 + 47))))
    term6 = -x3 * np.sin(np.sqrt(np.abs(x3 - (y3 + 47))))
    f_original = term1 + term2 + term3 + term4 + term5 + term6
    f_min = 3 * (-959.6407)  # Three instances
    return -(f_original - f_min)


# =============================================================================
# MULTIMODAL FUNCTIONS - MANY REGULAR PEAKS
# =============================================================================

def rastrigin_2d(params):
    """
    2D Rastrigin function (negated for maximization).

    Characteristics:
    - Highly multimodal with many regularly distributed local optima
    - Separable
    - Tests ability to escape numerous local optima

    Global optimum: f(0, 0) = 0.0
    Domain: [-5.12, 5.12]^2

    Reference: Rastrigin, L. A. (1974). Systems of extremal control.
    """
    A = 10
    n = len(params)
    return -(A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params)))


def rastrigin_4d(params):
    """
    4D Rastrigin function (negated for maximization).

    Characteristics:
    - Highly multimodal with many regularly distributed local optima
    - Separable
    - Tests ability to escape numerous local optima

    Global optimum: f(0, 0, 0, 0) = 0.0
    Domain: [-5.12, 5.12]^4
    """
    A = 10
    n = len(params)
    return -(A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params)))


def rastrigin_6d(params):
    """
    6D Rastrigin function (negated for maximization).

    Characteristics:
    - Highly multimodal with many regularly distributed local optima
    - Separable
    - Tests ability to escape numerous local optima

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5.12, 5.12]^6
    """
    A = 10
    n = len(params)
    return -(A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params)))


def rastrigin_10d(params):
    """
    10D Rastrigin function (negated for maximization).

    Characteristics:
    - Highly multimodal with many regularly distributed local optima
    - Separable
    - Tests ability to escape numerous local optima

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5.12, 5.12]^10
    """
    A = 10
    n = len(params)
    return -(A * n + np.sum(params**2 - A * np.cos(2 * np.pi * params)))


def ackley_2d(params):
    """
    2D Ackley function (negated for maximization).

    Characteristics:
    - Multimodal with nearly flat outer region
    - Sharp central peak
    - Tests exploration in flat regions

    Global optimum: f(0, 0) = 0.0
    Domain: [-5, 5]^2

    Reference: Ackley, D. H. (1987). A connectionist machine for genetic
    hillclimbing.
    """
    n = len(params)
    sum_sq = np.sum(params**2)
    sum_cos = np.sum(np.cos(2 * np.pi * params))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    f_original = term1 + term2 + 20 + np.e
    return -f_original


def ackley_4d(params):
    """
    4D Ackley function (negated for maximization).

    Characteristics:
    - Multimodal with nearly flat outer region
    - Sharp central peak
    - Tests exploration in flat regions

    Global optimum: f(0, 0, 0, 0) = 0.0
    Domain: [-5, 5]^4
    """
    n = len(params)
    sum_sq = np.sum(params**2)
    sum_cos = np.sum(np.cos(2 * np.pi * params))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    f_original = term1 + term2 + 20 + np.e
    return -f_original


def ackley_6d(params):
    """
    6D Ackley function (negated for maximization).

    Characteristics:
    - Multimodal with nearly flat outer region
    - Sharp central peak
    - Tests exploration in flat regions

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5, 5]^6
    """
    n = len(params)
    sum_sq = np.sum(params**2)
    sum_cos = np.sum(np.cos(2 * np.pi * params))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    f_original = term1 + term2 + 20 + np.e
    return -f_original


def ackley_10d(params):
    """
    10D Ackley function (negated for maximization).

    Characteristics:
    - Multimodal with nearly flat outer region
    - Sharp central peak
    - Tests exploration in flat regions

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-5, 5]^10
    """
    n = len(params)
    sum_sq = np.sum(params**2)
    sum_cos = np.sum(np.cos(2 * np.pi * params))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    f_original = term1 + term2 + 20 + np.e
    return -f_original


def griewank_2d(params):
    """
    2D Griewank function (negated for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Becomes easier in higher dimensions
    - Tests optimizer's ability to find global optimum

    Global optimum: f(0, 0) = 0.0
    Domain: [-100, 100]^2

    Reference: Griewank, A. O. (1981). Generalized descent for global
    optimization.
    """
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    f_original = sum_sq / 4000 - prod_cos + 1
    return -f_original


def griewank_4d(params):
    """
    4D Griewank function (negated for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Becomes easier in higher dimensions
    - Tests optimizer's ability to find global optimum

    Global optimum: f(0, 0, 0, 0) = 0.0
    Domain: [-100, 100]^4
    """
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    f_original = sum_sq / 4000 - prod_cos + 1
    return -f_original


def griewank_6d(params):
    """
    6D Griewank function (negated for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Becomes easier in higher dimensions
    - Tests optimizer's ability to find global optimum

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-100, 100]^6
    """
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    f_original = sum_sq / 4000 - prod_cos + 1
    return -f_original


def griewank_10d(params):
    """
    10D Griewank function (negated for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Becomes easier in higher dimensions
    - Tests optimizer's ability to find global optimum

    Global optimum: f(0, ..., 0) = 0.0
    Domain: [-100, 100]^10
    """
    sum_sq = np.sum(params**2)
    prod_cos = np.prod(np.cos(params / np.sqrt(np.arange(1, len(params) + 1))))
    f_original = sum_sq / 4000 - prod_cos + 1
    return -f_original


# =============================================================================
# MULTIMODAL FUNCTIONS - STEEP/RUGGED LANDSCAPE
# =============================================================================

def michalewicz_2d(params):
    """
    2D Michalewicz function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with steep valleys and ridges
    - Non-separable
    - Tests fine-grained optimization

    Global optimum: f(2.20, 1.57) ≈ 0.0 (approximate location)
    Original minimum: ≈ -1.8013 for m=10
    Domain: [0, π]^2

    Reference: Michalewicz, Z. (1996). Genetic Algorithms + Data Structures
    = Evolution Programs.
    """
    m = 10  # Defines steepness of valleys/ridges
    n = len(params)
    f_original = -np.sum(np.sin(params) * np.sin(((np.arange(1, n + 1) * params**2) / np.pi))**(2*m))
    f_min = -1.8013  # Approximate minimum for 2D, m=10
    return -(f_original - f_min)


def michalewicz_4d(params):
    """
    4D Michalewicz function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with steep valleys and ridges
    - Non-separable
    - Tests fine-grained optimization

    Global optimum: approximately at (2.20, 1.57, 1.28, 1.92)
    Original minimum: ≈ -3.72 for m=10
    Domain: [0, π]^4
    """
    m = 10
    n = len(params)
    f_original = -np.sum(np.sin(params) * np.sin(((np.arange(1, n + 1) * params**2) / np.pi))**(2*m))
    f_min = -3.72  # Approximate minimum for 4D, m=10
    return -(f_original - f_min)


def michalewicz_6d(params):
    """
    6D Michalewicz function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with steep valleys and ridges
    - Non-separable
    - Tests fine-grained optimization

    Global optimum: location varies with m parameter
    Original minimum: ≈ -5.69 for m=10
    Domain: [0, π]^6
    """
    m = 10
    n = len(params)
    f_original = -np.sum(np.sin(params) * np.sin(((np.arange(1, n + 1) * params**2) / np.pi))**(2*m))
    f_min = -5.69  # Approximate minimum for 6D, m=10
    return -(f_original - f_min)


def michalewicz_10d(params):
    """
    10D Michalewicz function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with steep valleys and ridges
    - Non-separable
    - Tests fine-grained optimization

    Global optimum: location varies with m parameter
    Original minimum: ≈ -9.66 for m=10
    Domain: [0, π]^10
    """
    m = 10
    n = len(params)
    f_original = -np.sum(np.sin(params) * np.sin(((np.arange(1, n + 1) * params**2) / np.pi))**(2*m))
    f_min = -9.66  # Approximate minimum for 10D, m=10
    return -(f_original - f_min)


def styblinski_tang_2d(params):
    """
    2D Styblinski-Tang function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Separable
    - All global optima at same value

    Global optimum: f(-2.903534, -2.903534) = 0.0
    Original minimum: -39.16617*n (per dimension)
    Domain: [-5, 5]^2

    Reference: Styblinski, M. A. & Tang, T.-S. (1990). Experiments in
    nonconvex optimization.
    """
    n = len(params)
    f_original = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)
    f_min = -39.16617 * n
    return -(f_original - f_min)


def styblinski_tang_4d(params):
    """
    4D Styblinski-Tang function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Separable
    - All global optima at same value

    Global optimum: f(-2.903534, ..., -2.903534) = 0.0
    Original minimum: -39.16617*n
    Domain: [-5, 5]^4
    """
    n = len(params)
    f_original = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)
    f_min = -39.16617 * n
    return -(f_original - f_min)


def styblinski_tang_6d(params):
    """
    6D Styblinski-Tang function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Separable
    - All global optima at same value

    Global optimum: f(-2.903534, ..., -2.903534) = 0.0
    Original minimum: -39.16617*n
    Domain: [-5, 5]^6
    """
    n = len(params)
    f_original = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)
    f_min = -39.16617 * n
    return -(f_original - f_min)


def styblinski_tang_10d(params):
    """
    10D Styblinski-Tang function (negated and shifted for maximization).

    Characteristics:
    - Multimodal with many local optima
    - Separable
    - All global optima at same value

    Global optimum: f(-2.903534, ..., -2.903534) = 0.0
    Original minimum: -39.16617*n
    Domain: [-5, 5]^10
    """
    n = len(params)
    f_original = 0.5 * np.sum(params**4 - 16*params**2 + 5*params)
    f_min = -39.16617 * n
    return -(f_original - f_min)


def levy_2d(params):
    """
    2D Levy function (negated for maximization).

    Characteristics:
    - Multimodal with wave-like structure
    - Non-separable
    - Tests neighbor-based propagation

    Global optimum: f(1, 1) = 0.0
    Domain: [-10, 10]^2

    Reference: Levy, H. & Montalvo, A. (1985). The tunneling algorithm
    for the global minimization of functions.
    """
    w = 1 + (params - 1) / 4
    n = len(params)
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    f_original = term1 + term2 + term3
    return -f_original


def levy_4d(params):
    """
    4D Levy function (negated for maximization).

    Characteristics:
    - Multimodal with wave-like structure
    - Non-separable
    - Tests neighbor-based propagation

    Global optimum: f(1, 1, 1, 1) = 0.0
    Domain: [-10, 10]^4
    """
    w = 1 + (params - 1) / 4
    n = len(params)
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    f_original = term1 + term2 + term3
    return -f_original


def levy_6d(params):
    """
    6D Levy function (negated for maximization).

    Characteristics:
    - Multimodal with wave-like structure
    - Non-separable
    - Tests neighbor-based propagation

    Global optimum: f(1, ..., 1) = 0.0
    Domain: [-10, 10]^6
    """
    w = 1 + (params - 1) / 4
    n = len(params)
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    f_original = term1 + term2 + term3
    return -f_original


def levy_10d(params):
    """
    10D Levy function (negated for maximization).

    Characteristics:
    - Multimodal with wave-like structure
    - Non-separable
    - Tests neighbor-based propagation

    Global optimum: f(1, ..., 1) = 0.0
    Domain: [-10, 10]^10
    """
    w = 1 + (params - 1) / 4
    n = len(params)
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    f_original = term1 + term2 + term3
    return -f_original


def schwefel_2d(params):
    """
    2D Schwefel function (negated and shifted for maximization).

    Characteristics:
    - Highly multimodal, deceptive
    - Global optimum far from local optima
    - Separable
    - Very challenging for optimization

    Global optimum: f(420.9687, 420.9687) = 0.0
    Original minimum: -418.9829*n
    Domain: [-500, 500]^2

    Reference: Schwefel, H.-P. (1981). Numerical Optimization of Computer
    Models.
    """
    n = len(params)
    f_original = 418.9829 * n - np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return -f_original


def schwefel_4d(params):
    """
    4D Schwefel function (negated and shifted for maximization).

    Characteristics:
    - Highly multimodal, deceptive
    - Global optimum far from local optima
    - Separable
    - Very challenging for optimization

    Global optimum: f(420.9687, ..., 420.9687) = 0.0
    Original minimum: -418.9829*n
    Domain: [-500, 500]^4
    """
    n = len(params)
    f_original = 418.9829 * n - np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return -f_original


def schwefel_6d(params):
    """
    6D Schwefel function (negated and shifted for maximization).

    Characteristics:
    - Highly multimodal, deceptive
    - Global optimum far from local optima
    - Separable
    - Very challenging for optimization

    Global optimum: f(420.9687, ..., 420.9687) = 0.0
    Original minimum: -418.9829*n
    Domain: [-500, 500]^6
    """
    n = len(params)
    f_original = 418.9829 * n - np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return -f_original


def schwefel_10d(params):
    """
    10D Schwefel function (negated and shifted for maximization).

    Characteristics:
    - Highly multimodal, deceptive
    - Global optimum far from local optima
    - Separable
    - Very challenging for optimization

    Global optimum: f(420.9687, ..., 420.9687) = 0.0
    Original minimum: -418.9829*n
    Domain: [-500, 500]^10
    """
    n = len(params)
    f_original = 418.9829 * n - np.sum(params * np.sin(np.sqrt(np.abs(params))))
    return -f_original


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_test_function(name):
    """
    Factory function to get a test likelihood, its bounds, and true peaks.

    Parameters
    ----------
    name : str
        Name of the test function. Format: "function_Nd" where N is dimension.
        Examples: "sphere_2d", "rastrigin_4d", "eggholder_6d"

        Available functions:
        - Unimodal: sphere, rosenbrock
        - Multimodal (few peaks): himmelblau, beale, eggholder
        - Multimodal (many peaks): rastrigin, ackley, griewank
        - Multimodal (steep): michalewicz, styblinski_tang, levy, schwefel

    Returns
    -------
    func : callable
        The test function
    bounds : list of [min, max] pairs
        Parameter bounds for each dimension
    peaks : list of numpy arrays
        Known peak locations (empty list if not precisely known)
    """

    # Dictionary mapping function names to their configurations
    # Format: (function, bounds_per_dim, peaks)

    function_registry = {
        # Sphere functions
        "sphere_2d": (sphere_2d, [-5, 5], [np.zeros(2)]),
        "sphere_4d": (sphere_4d, [-5, 5], [np.zeros(4)]),
        "sphere_6d": (sphere_6d, [-5, 5], [np.zeros(6)]),
        "sphere_10d": (sphere_10d, [-5, 5], [np.zeros(10)]),

        # Rosenbrock functions
        "rosenbrock_2d": (rosenbrock_2d, [-6, 6], [np.ones(2)]),
        "rosenbrock_4d": (rosenbrock_4d, [-6, 6], [np.ones(4)]),
        "rosenbrock_6d": (rosenbrock_6d, [-6, 6], [np.ones(6)]),
        "rosenbrock_10d": (rosenbrock_10d, [-6, 6], [np.ones(10)]),

        # Himmelblau function
        "himmelblau_4d": (himmelblau_4d, [-6, 6], [
            np.array([3.0, 2.0, 3.0, 2.0]),
            np.array([-2.805118, 3.131312, -2.805118, 3.131312]),
            np.array([-3.779310, -3.283186, -3.779310, -3.283186]),
            np.array([3.584428, -1.848126, 3.584428, -1.848126])
        ]),

        # Beale function
        "beale_2d": (beale_2d, [-4.5, 4.5], [np.array([3.0, 0.5])]),

        # Eggholder functions
        "eggholder_2d": (eggholder_2d, [-512, 512], [np.array([512.0, 404.2319])]),
        "eggholder_4d": (eggholder_4d, [-512, 512], [np.array([512.0, 404.2319, 512.0, 404.2319])]),
        "eggholder_6d": (eggholder_6d, [-512, 512], [np.array([512.0, 404.2319, 512.0, 404.2319, 512.0, 404.2319])]),

        # Rastrigin functions
        "rastrigin_2d": (rastrigin_2d, [-5.12, 5.12], [np.zeros(2)]),
        "rastrigin_4d": (rastrigin_4d, [-5.12, 5.12], [np.zeros(4)]),
        "rastrigin_6d": (rastrigin_6d, [-5.12, 5.12], [np.zeros(6)]),
        "rastrigin_10d": (rastrigin_10d, [-5.12, 5.12], [np.zeros(10)]),

        # Ackley functions
        "ackley_2d": (ackley_2d, [-5, 5], [np.zeros(2)]),
        "ackley_4d": (ackley_4d, [-5, 5], [np.zeros(4)]),
        "ackley_6d": (ackley_6d, [-5, 5], [np.zeros(6)]),
        "ackley_10d": (ackley_10d, [-5, 5], [np.zeros(10)]),

        # Griewank functions
        "griewank_2d": (griewank_2d, [-100, 100], [np.zeros(2)]),
        "griewank_4d": (griewank_4d, [-100, 100], [np.zeros(4)]),
        "griewank_6d": (griewank_6d, [-100, 100], [np.zeros(6)]),
        "griewank_10d": (griewank_10d, [-100, 100], [np.zeros(10)]),

        # Michalewicz functions (peaks are approximate)
        "michalewicz_2d": (michalewicz_2d, [0, np.pi], []),
        "michalewicz_4d": (michalewicz_4d, [0, np.pi], []),
        "michalewicz_6d": (michalewicz_6d, [0, np.pi], []),
        "michalewicz_10d": (michalewicz_10d, [0, np.pi], []),

        # Styblinski-Tang functions
        "styblinski_tang_2d": (styblinski_tang_2d, [-5, 5], [np.full(2, -2.903534)]),
        "styblinski_tang_4d": (styblinski_tang_4d, [-5, 5], [np.full(4, -2.903534)]),
        "styblinski_tang_6d": (styblinski_tang_6d, [-5, 5], [np.full(6, -2.903534)]),
        "styblinski_tang_10d": (styblinski_tang_10d, [-5, 5], [np.full(10, -2.903534)]),

        # Levy functions
        "levy_2d": (levy_2d, [-10, 10], [np.ones(2)]),
        "levy_4d": (levy_4d, [-10, 10], [np.ones(4)]),
        "levy_6d": (levy_6d, [-10, 10], [np.ones(6)]),
        "levy_10d": (levy_10d, [-10, 10], [np.ones(10)]),

        # Schwefel functions
        "schwefel_2d": (schwefel_2d, [-500, 500], [np.full(2, 420.9687)]),
        "schwefel_4d": (schwefel_4d, [-500, 500], [np.full(4, 420.9687)]),
        "schwefel_6d": (schwefel_6d, [-500, 500], [np.full(6, 420.9687)]),
        "schwefel_10d": (schwefel_10d, [-500, 500], [np.full(10, 420.9687)]),
    }

    if name not in function_registry:
        available = sorted(function_registry.keys())
        raise ValueError(
            f"Unknown test function: '{name}'\n"
            f"Available functions:\n" +
            "\n".join(f"  - {fn}" for fn in available)
        )

    func, bounds_per_dim, peaks = function_registry[name]

    # Determine dimensionality from function
    if peaks:
        n_dims = len(peaks[0])
    else:
        # Extract dimension from function name
        n_dims = int(name.split('_')[-1].replace('d', ''))

    # Create bounds list
    bounds = [bounds_per_dim] * n_dims

    return func, bounds, peaks
