# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Adapted from mgcv 1.9-4, commit
# 1b6a4c8374612da27e36420b4459e93acb183f2d,
# R/smooth.r and src/mgcv.c.
# Python/JAX adaptation modified 2026-07-24.

"""One-dimensional smooth constructors."""

import jax.numpy as jnp
import numpy as np

from ._smooth import Array, ArrayLike, Smooth
from ._splines import (
    basis_matrix,
    cyclic_basis_matrix,
    cyclic_difference_penalty,
    equidistant_knots,
    linear_extrapolation_basis,
    pspline_penalty,
    wrap_periodic,
)


def _strict_knots(knots: ArrayLike, expected: int, name: str) -> np.ndarray:
    result = np.asarray(knots, dtype=float).reshape(-1)
    if result.size != expected:
        raise ValueError(f"There should be {expected} supplied knots for {name}.")
    result = np.sort(result)
    if not np.isfinite(result).all() or not np.all(np.diff(result) > 0.0):
        raise ValueError("Knots must be finite, unique, and strictly increasing.")
    return result


def _pspline_knots(
    x: ArrayLike, k: int, degree: int, knots: ArrayLike | None
) -> np.ndarray:
    expected = k + degree + 1
    if knots is None:
        return np.asarray(equidistant_knots(x, k, degree, eps=0.002))
    supplied = np.asarray(knots, dtype=float).reshape(-1)
    if supplied.size == 2:
        bounds = np.sort(supplied)
        if bounds[0] > np.min(x) or bounds[1] < np.max(x):
            raise ValueError("Knot range does not include the data.")
        return np.asarray(equidistant_knots(bounds, k, degree, eps=0.002))
    return _strict_knots(supplied, expected, "a P-spline")


def pspline(
    x: ArrayLike,
    *,
    k: int,
    degree: int,
    penalty_order: int,
    knots: ArrayLike | None = None,
) -> Smooth:
    """Construct a B-spline basis with a coefficient-difference penalty.

    This is the standard P-spline construction: a flexible local basis is
    regularized by finite differences between neighboring coefficients. Basis
    evaluation uses linear extrapolation beyond the interior knot range.

    Parameters
    ----------
    x
        One-dimensional values used to choose or validate the knots.
    k
        Number of basis functions and coefficients.
    degree
        Polynomial degree of the B-spline basis.
    penalty_order
        Order of the coefficient differences in the penalty.
    knots
        Knot specification. Use ``None`` for automatic knots, two values for
        boundary limits, or ``k + degree + 1`` values for the full sequence.

    Returns
    -------
    smooth :
        A smooth with ``k`` basis columns and penalty rank
        ``k - penalty_order``.

    Raises
    ------
    ValueError
        If the basis dimension or knot specification is invalid.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import pspline
    >>> x = jnp.linspace(0.0, 1.0, 10)
    >>> smooth = pspline(x, k=6, degree=3, penalty_order=2)
    >>> assert smooth.basis(x).shape == (10, 6)
    >>> assert smooth.rank == 4

    ```
    """
    if k <= degree - 1:
        raise ValueError("Basis dimension is too small for the B-spline degree.")
    knot_array = jnp.asarray(_pspline_knots(x, k, degree, knots))
    penalty = pspline_penalty(k, penalty_order)

    def evaluate(values: ArrayLike) -> Array:
        return linear_extrapolation_basis(jnp.ravel(values), knot_array, degree)

    return Smooth(evaluate, penalty, k - penalty_order, knot_array)


def _bspline_knots(
    x: ArrayLike, k: int, degree: int, knots: ArrayLike | None
) -> np.ndarray:
    expected = k + degree + 1
    n_interior = k - degree + 1
    if knots is None:
        return np.asarray(equidistant_knots(x, k, degree, eps=0.002))
    supplied = np.asarray(knots, dtype=float).reshape(-1)
    if supplied.size == 2:
        bounds = np.sort(supplied)
        if bounds[0] > np.min(x) or bounds[1] < np.max(x):
            raise ValueError("Knot range does not include the data.")
        return np.asarray(equidistant_knots(bounds, k, degree, eps=0.002))
    if supplied.size == 4 and supplied.size < expected:
        bounds = np.sort(supplied)
        step = (bounds[3] - bounds[0]) / (n_interior - 1)
        left = np.linspace(bounds[0] - step * degree, bounds[0], degree + 1)
        middle = np.linspace(bounds[1], bounds[2], max(0, n_interior - 2))
        right = np.linspace(bounds[3], bounds[3] + step * degree, degree + 1)
        return np.concatenate((left, middle, right))
    return _strict_knots(supplied, expected, "a B-spline")


def _derivative_penalty(knots: np.ndarray, degree: int, derivative: int) -> Array:
    if derivative > degree:
        raise ValueError("Requested non-existent derivative in B-spline penalty.")
    lower = degree
    upper = len(knots) - degree - 1
    intervals = zip(knots[lower:upper], knots[lower + 1 : upper + 1])
    quadrature_order = max(1, degree - derivative + 1)
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    penalty = None
    knot_array = jnp.asarray(knots)
    for left, right in intervals:
        points = (right - left) * (nodes + 1.0) / 2.0 + left
        local_weights = weights * (right - left) / 2.0
        derivatives = basis_matrix(
            jnp.asarray(points),
            knot_array,
            order=degree,
            outer_ok=True,
            derivative=derivative,
        )
        contribution = derivatives.T @ (
            jnp.asarray(local_weights)[:, None] * derivatives
        )
        penalty = contribution if penalty is None else penalty + contribution
    assert penalty is not None
    return (penalty + penalty.T) / 2.0


def bspline(
    x: ArrayLike,
    *,
    k: int,
    degree: int,
    penalty_order: int,
    knots: ArrayLike | None = None,
) -> Smooth:
    """Construct a B-spline with an integrated derivative penalty.

    Unlike a P-spline coefficient penalty, this construction numerically
    integrates the squared derivative of each basis function. Evaluation uses
    linear extrapolation beyond the interior knot range.

    Parameters
    ----------
    x
        One-dimensional values used to choose or validate the knots.
    k
        Number of basis functions and coefficients.
    degree
        Polynomial degree of the B-spline basis.
    penalty_order
        Derivative order used in the integrated squared penalty.
    knots
        Knot specification. Use ``None`` for automatic knots, two values for
        boundary limits, four mgcv-style boundary knots, or
        ``k + degree + 1`` values for the full sequence.

    Returns
    -------
    smooth :
        A smooth with ``k`` basis columns and penalty rank
        ``k - penalty_order``.

    Raises
    ------
    ValueError
        If the knot specification is invalid or ``penalty_order`` exceeds
        ``degree``.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import bspline
    >>> x = jnp.linspace(0.0, 1.0, 10)
    >>> smooth = bspline(x, k=6, degree=3, penalty_order=2)
    >>> assert smooth.basis(x).shape == (10, 6)
    >>> assert smooth.penalty.shape == (6, 6)

    ```
    """
    knot_values = _bspline_knots(x, k, degree, knots)
    knot_array = jnp.asarray(knot_values)
    penalty = _derivative_penalty(knot_values, degree, penalty_order)

    def evaluate(values: ArrayLike) -> Array:
        return linear_extrapolation_basis(jnp.ravel(values), knot_array, degree)

    return Smooth(evaluate, penalty, k - penalty_order, knot_array)


def cyclic_pspline(
    x: ArrayLike,
    *,
    k: int,
    degree: int,
    penalty_order: int,
    knots: ArrayLike | None = None,
) -> Smooth:
    """Construct a periodic B-spline with wrapped coefficient differences.

    Values outside the boundary interval are wrapped into it. Both the basis
    and the difference penalty join across the boundary, making this useful for
    seasonal or angular covariates.

    Parameters
    ----------
    x
        One-dimensional values used to choose or validate the boundary range.
    k
        Number of periodic basis functions and coefficients.
    degree
        Polynomial degree of the B-spline basis.
    penalty_order
        Order of the wrapped coefficient differences.
    knots
        Knot specification. Use ``None`` for automatic knots, two values for
        boundary limits, or exactly ``k + 1`` knot values.

    Returns
    -------
    smooth :
        A periodic smooth with ``k`` basis columns and rank ``k - 1``.

    Raises
    ------
    ValueError
        If the boundary range, knot specification, or penalty order is invalid.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import cyclic_pspline
    >>> x = jnp.linspace(0.0, 1.0, 10)
    >>> smooth = cyclic_pspline(x, k=6, degree=3, penalty_order=2)
    >>> endpoints = smooth.basis(jnp.array([0.0, 1.0]))
    >>> assert bool(jnp.allclose(endpoints[0], endpoints[1]))

    ```
    """
    if knots is None:
        knot_values = np.linspace(np.min(x), np.max(x), k + 1)
    else:
        supplied = np.asarray(knots, dtype=float).reshape(-1)
        if supplied.size == 2:
            bounds = np.sort(supplied)
            if bounds[0] > np.min(x) or bounds[1] < np.max(x):
                raise ValueError("Knot range does not include the data.")
            knot_values = np.linspace(bounds[0], bounds[1], k + 1)
        else:
            knot_values = _strict_knots(supplied, k + 1, "a cyclic P-spline")
    knot_array = jnp.asarray(knot_values)
    penalty = cyclic_difference_penalty(k, penalty_order)

    def evaluate(values: ArrayLike) -> Array:
        return cyclic_basis_matrix(jnp.ravel(values), knot_array, degree)

    return Smooth(evaluate, penalty, k - 1, knot_array)


def _cardinal_system(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_knots = knots.size
    spacing = np.diff(knots)
    differences = np.zeros((n_knots - 2, n_knots))
    for index in range(n_knots - 2):
        differences[index, index] = 1.0 / spacing[index]
        differences[index, index + 1] = -(
            1.0 / spacing[index] + 1.0 / spacing[index + 1]
        )
        differences[index, index + 2] = 1.0 / spacing[index + 1]
    system = np.diag((spacing[:-1] + spacing[1:]) / 3.0)
    if n_knots > 3:
        off_diagonal = spacing[1:-1] / 6.0
        system += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    interior_map = np.linalg.solve(system, differences)
    second_derivative_map = np.zeros((n_knots, n_knots))
    second_derivative_map[1:-1] = interior_map
    penalty = differences.T @ interior_map
    return second_derivative_map, (penalty + penalty.T) / 2.0


def _natural_cubic_basis(x: Array, knots: Array, second_derivative_map: Array) -> Array:
    x = jnp.ravel(x)
    n_knots = knots.shape[0]
    interval = jnp.searchsorted(knots, x, side="right") - 1
    interval = jnp.clip(interval, 0, n_knots - 2)
    left_knot = knots[interval]
    right_knot = knots[interval + 1]
    width = right_knot - left_knot
    left_weight = (right_knot - x) / width
    right_weight = (x - left_knot) / width
    identity = jnp.eye(n_knots, dtype=x.dtype)
    left_identity = identity[interval]
    right_identity = identity[interval + 1]
    left_second = second_derivative_map[interval]
    right_second = second_derivative_map[interval + 1]
    interior = (
        left_weight[:, None] * left_identity
        + right_weight[:, None] * right_identity
        + ((left_weight**3 - left_weight) * width**2 / 6.0)[:, None] * left_second
        + ((right_weight**3 - right_weight) * width**2 / 6.0)[:, None] * right_second
    )

    first_width = knots[1] - knots[0]
    left_slope = (identity[1] - identity[0]) / first_width - first_width * (
        2.0 * second_derivative_map[0] + second_derivative_map[1]
    ) / 6.0
    last_width = knots[-1] - knots[-2]
    right_slope = (identity[-1] - identity[-2]) / last_width + last_width * (
        second_derivative_map[-2] + 2.0 * second_derivative_map[-1]
    ) / 6.0
    left = identity[0] + (x - knots[0])[:, None] * left_slope
    right = identity[-1] + (x - knots[-1])[:, None] * right_slope
    return jnp.where(
        (x < knots[0])[:, None],
        left,
        jnp.where((x > knots[-1])[:, None], right, interior),
    )


def _shrink_penalty(penalty: np.ndarray, shrink: float, nullity: int) -> Array:
    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    base = eigenvalues[nullity]
    for offset in range(nullity):
        eigenvalues[nullity - offset - 1] = base * shrink ** (offset + 1)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    return jnp.asarray((result + result.T) / 2.0)


def cubic_regression(
    x: ArrayLike,
    *,
    k: int,
    knots: ArrayLike | None = None,
    shrinkage: bool = False,
) -> Smooth:
    """Construct a natural cardinal cubic regression spline.

    Coefficients represent function values at the knots. The curvature penalty
    has an unpenalized constant-and-linear null space unless ``shrinkage`` is
    enabled, and evaluation extrapolates linearly beyond the boundary knots.

    Parameters
    ----------
    x
        One-dimensional values used to place or validate the knots.
    k
        Number of knots, basis functions, and coefficients.
    knots
        Exact knot locations. ``None`` chooses quantiles of the unique values.
    shrinkage
        Whether to add small penalties to the two null-space directions.

    Returns
    -------
    smooth :
        A natural cubic smooth with rank ``k - 2``, or ``k`` with shrinkage.

    Raises
    ------
    ValueError
        If there are fewer than ``k`` unique values or the supplied knots are
        invalid.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import cubic_regression
    >>> x = jnp.linspace(0.0, 1.0, 12)
    >>> smooth = cubic_regression(x, k=6)
    >>> assert smooth.basis(x).shape == (12, 6)
    >>> assert smooth.rank == 4

    ```
    """
    unique = np.unique(np.asarray(x, dtype=float).reshape(-1))
    if unique.size < k:
        raise ValueError("Insufficient unique values to support the requested knots.")
    if knots is None:
        knot_values = np.quantile(unique, np.linspace(0.0, 1.0, k))
    else:
        knot_values = _strict_knots(knots, k, "a cubic regression spline")
    second_map, penalty = _cardinal_system(knot_values)
    knot_array = jnp.asarray(knot_values)
    second_array = jnp.asarray(second_map)
    penalty_array = (
        _shrink_penalty(penalty, 0.1, nullity=2) if shrinkage else jnp.asarray(penalty)
    )

    def evaluate(values: ArrayLike) -> Array:
        return _natural_cubic_basis(jnp.asarray(values), knot_array, second_array)

    return Smooth(evaluate, penalty_array, k if shrinkage else k - 2, knot_array)


def _cyclic_system(knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spacing = np.diff(knots)
    n_coef = knots.size - 1
    system = np.zeros((n_coef, n_coef))
    differences = np.zeros((n_coef, n_coef))
    for index in range(n_coef):
        previous = (index - 1) % n_coef
        following = (index + 1) % n_coef
        previous_width = spacing[previous]
        following_width = spacing[index]
        system[index, previous] = previous_width / 6.0
        system[index, index] = (previous_width + following_width) / 3.0
        system[index, following] = following_width / 6.0
        differences[index, previous] = 1.0 / previous_width
        differences[index, index] = -(1.0 / previous_width + 1.0 / following_width)
        differences[index, following] = 1.0 / following_width
    second_map = np.linalg.solve(system, differences)
    penalty = differences.T @ second_map
    return second_map, (penalty + penalty.T) / 2.0


def _cyclic_cubic_basis(x: Array, knots: Array, second_map: Array) -> Array:
    x = wrap_periodic(jnp.ravel(x), knots[0], knots[-1])
    n_coef = knots.shape[0] - 1
    upper = jnp.searchsorted(knots, x, side="left")
    upper = jnp.clip(upper, 1, n_coef)
    left = upper - 1
    right = jnp.where(upper == n_coef, 0, upper)
    width = knots[upper] - knots[left]
    left_distance = knots[upper] - x
    right_distance = x - knots[left]
    identity = jnp.eye(n_coef, dtype=x.dtype)
    return (
        second_map[left] * (left_distance**3 / (6.0 * width))[:, None]
        + second_map[right] * (right_distance**3 / (6.0 * width))[:, None]
        - second_map[left] * (width * left_distance / 6.0)[:, None]
        - second_map[right] * (width * right_distance / 6.0)[:, None]
        + identity[left] * (left_distance / width)[:, None]
        + identity[right] * (right_distance / width)[:, None]
    )


def cyclic_cubic(
    x: ArrayLike,
    *,
    k: int,
    knots: ArrayLike | None = None,
) -> Smooth:
    """Construct a periodic cardinal cubic regression spline.

    The first and last knot describe the same periodic boundary, so ``k``
    knots produce ``k - 1`` coefficients. Values outside that interval are
    wrapped into it.

    Parameters
    ----------
    x
        One-dimensional values used to place or validate the knots.
    k
        Number of knots, with a minimum of four.
    knots
        Knot specification. ``None`` chooses quantiles, two values contribute
        boundary information, and ``k`` values specify the complete sequence.

    Returns
    -------
    smooth :
        A periodic cubic smooth with ``k - 1`` basis columns and rank ``k - 2``.

    Raises
    ------
    ValueError
        If there are fewer than ``k`` unique values or the supplied knots are
        invalid.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import cyclic_cubic
    >>> x = jnp.linspace(0.0, 1.0, 12)
    >>> smooth = cyclic_cubic(x, k=6)
    >>> assert smooth.basis(x).shape == (12, 5)
    >>> assert smooth.rank == 4

    ```
    """
    k = max(k, 4)
    unique = np.unique(np.asarray(x, dtype=float).reshape(-1))
    if unique.size < k:
        raise ValueError("Insufficient unique values to support the requested knots.")
    if knots is None:
        knot_values = np.quantile(unique, np.linspace(0.0, 1.0, k))
    else:
        supplied = np.asarray(knots, dtype=float).reshape(-1)
        if supplied.size == 2:
            knot_values = np.quantile(
                np.unique(np.concatenate((supplied, unique))),
                np.linspace(0.0, 1.0, k),
            )
        else:
            knot_values = _strict_knots(supplied, k, "a cyclic cubic spline")
    second_map, penalty = _cyclic_system(knot_values)
    knot_array = jnp.asarray(knot_values)
    second_array = jnp.asarray(second_map)

    def evaluate(values: ArrayLike) -> Array:
        return _cyclic_cubic_basis(jnp.asarray(values), knot_array, second_array)

    return Smooth(evaluate, jnp.asarray(penalty), k - 2, knot_array)
