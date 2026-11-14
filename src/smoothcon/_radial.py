# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Adapted from mgcv 1.9-4, commit
# 1b6a4c8374612da27e36420b4459e93acb183f2d,
# R/smooth.r and src/tprs.c.
# Python/JAX adaptation modified 2026-07-24.

"""Build smooths from distances between input locations."""

import hashlib
import math
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._smooth import Array, ArrayLike, Smooth, _canonical_signs, _smooth_from_setup

XpArray = Any
"""A JAX or NumPy array, matching the ``xp`` module it is computed with."""

_ROW_BLOCK = 4096
"""Rows of the setup data evaluated at once, to bound float64 memory."""


def _as_matrix(x: ArrayLike) -> Array:
    array = jnp.asarray(x)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a vector or matrix, got shape {array.shape}.")
    return array


def _as_float64_matrix(x: ArrayLike) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a vector or matrix, got shape {array.shape}.")
    return array


def _eigen_largest_magnitude(
    matrix: ArrayLike, n_vectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the eigenpairs of largest magnitude, in float64."""
    array = np.asarray(matrix, dtype=np.float64)
    values, vectors = np.linalg.eigh((array + array.T) / 2.0)
    indices = np.argsort(np.abs(values), kind="stable")[::-1][:n_vectors]
    return values[indices], _canonical_signs(vectors[:, indices])


def _row_blocks(x: np.ndarray):
    for start in range(0, x.shape[0], _ROW_BLOCK):
        yield x[start : start + _ROW_BLOCK]


def _deterministic_sample(rows: np.ndarray, size: int) -> np.ndarray:
    canonical = np.asarray(rows, dtype="<f8")
    keyed: list[tuple[bytes, tuple[float, ...], int]] = []
    for index, row in enumerate(canonical):
        digest = hashlib.blake2b(
            row.tobytes(), digest_size=16, person=b"smoothcon-v1"
        ).digest()
        keyed.append((digest, tuple(float(value) for value in row), index))
    keyed.sort()
    return rows[[entry[2] for entry in keyed[:size]]]


def _unique_locations(x: np.ndarray, max_locations: int = 2000) -> np.ndarray:
    unique = np.unique(x, axis=0)
    if unique.shape[0] > max_locations:
        return _deterministic_sample(unique, max_locations)
    return unique


def _null_space_dimension(dimension: int, order: int) -> tuple[int, int]:
    if 2 * order <= dimension:
        order = 1
        while 2 * order < dimension + 2:
            order += 1
    return order, math.comb(order + dimension - 1, dimension)


def _polynomial_powers(dimension: int, order: int, count: int) -> np.ndarray:
    index = np.zeros(dimension, dtype=int)
    powers = np.zeros((count, dimension), dtype=int)
    for row in range(count):
        powers[row] = index
        total = int(np.sum(index))
        if total < order - 1:
            index[0] += 1
        else:
            total -= int(index[0])
            index[0] = 0
            for column in range(1, dimension):
                index[column] += 1
                total += 1
                if total == order:
                    total -= int(index[column])
                    index[column] = 0
                else:
                    break
    return powers


def _polynomial_basis(x: XpArray, powers: np.ndarray, xp: ModuleType = jnp) -> XpArray:
    powers_array = xp.asarray(powers)
    return xp.prod(x[:, None, :] ** powers_array[None, :, :], axis=2)


def _eta_constant(order: int, dimension: int) -> float:
    if dimension % 2 == 0:
        sign = -1.0 if (order + 1 + dimension // 2) % 2 else 1.0
        return sign / (
            2.0 ** (2 * order - 1)
            * math.pi ** (dimension / 2)
            * math.factorial(order - 1)
            * math.factorial(order - dimension // 2)
        )
    return math.gamma(dimension / 2 - order) / (
        2.0 ** (2 * order) * math.pi ** (dimension / 2) * math.factorial(order - 1)
    )


def _thin_plate_kernel(
    x: XpArray, centers: XpArray, order: int, xp: ModuleType = jnp
) -> XpArray:
    dimension = x.shape[1]
    squared_distance = xp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    constant = _eta_constant(order, dimension)
    if dimension % 2 == 0:
        exponent = order - dimension // 2
        safe_distance = xp.where(squared_distance > 0.0, squared_distance, 1.0)
        result = constant * 0.5 * xp.log(safe_distance) * safe_distance**exponent
    else:
        exponent = order - dimension // 2 - 1
        result = constant * squared_distance**exponent * xp.sqrt(squared_distance)
    return xp.where(squared_distance > 0.0, result, 0.0)


def _nullspace(matrix: np.ndarray, dimension: int) -> np.ndarray:
    _, _, right = np.linalg.svd(matrix, full_matrices=True)
    return right.T[:, dimension:]


def _shrink_null_space(penalty: np.ndarray, shrink: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((penalty + penalty.T) / 2.0)
    tolerance = np.max(values) * np.finfo(np.float64).eps ** 0.8
    smallest = np.min(values[values > tolerance])
    values = np.where(values > tolerance, values, smallest * shrink)
    return (vectors * values) @ vectors.T


def thin_plate(
    x: ArrayLike,
    *,
    k: int,
    penalty_order: int,
    knots: ArrayLike | None = None,
    shrinkage: bool = False,
    remove_null_space: bool = False,
) -> Smooth:
    """Create a smooth that can bend in one or more dimensions.

    A thin-plate spline penalizes bending equally in every direction, so there
    is no need to choose separate knot axes. This smaller, low-rank version
    follows Wood (2003). Constant and low-order polynomial patterns remain
    unpenalized unless they are shrunk or removed.

    Parameters
    ----------
    x
        Covariate vector or matrix with observations in rows and dimensions in
        columns.
    k
        Requested basis dimension. It is raised when necessary to leave at
        least one penalized direction beyond the polynomial null space.
    penalty_order
        Thin-plate derivative order. Invalid low orders are raised to the
        smallest valid order for the covariate dimension.
    knots
        Optional center locations, as a vector in one dimension or a matrix in
        multiple dimensions. ``None`` uses unique covariate locations.
    shrinkage
        Whether to add small penalties to the polynomial null space.
    remove_null_space
        Whether to drop the polynomial null-space columns and center the
        remaining basis over ``x``.

    Returns
    -------
    smooth :
        A low-rank thin-plate smooth whose ``knots`` contain its centers.

    Raises
    ------
    ValueError
        If the covariates or custom knots have invalid dimensions, or too few
        unique locations support the requested basis.

    References
    ----------
    Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
    Statistical Society: Series B*, 65(1), 95–114.
    https://doi.org/10.1111/1467-9868.00374

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import thin_plate
    >>> x = jnp.linspace(0.0, 1.0, 10)
    >>> smooth = thin_plate(x, k=6, penalty_order=2)
    >>> smooth.basis(x).shape
    (10, 6)
    >>> smooth.penalty.shape
    (6, 6)
    >>> smooth.rank
    4

    ```
    """
    x_array = _as_float64_matrix(x)
    dimension = x_array.shape[1]
    order, nullity = _null_space_dimension(dimension, penalty_order)
    k = max(k, nullity + 1)

    shift = np.mean(x_array, axis=0)
    centered = x_array - shift
    if knots is None:
        centers = _unique_locations(centered)
    else:
        supplied = np.asarray(knots, dtype=np.float64)
        if dimension != 1 and supplied.ndim != 2:
            raise ValueError("Multidimensional custom knots are not supported.")
        supplied = supplied.reshape(-1, dimension) - shift
        centers = (
            _unique_locations(centered)
            if supplied.shape[0] < k or supplied.shape[0] > x_array.shape[0]
            else np.unique(supplied, axis=0)
        )
    if centers.shape[0] < k:
        raise ValueError(
            "Fewer unique covariate combinations than the requested basis dimension."
        )

    powers = _polynomial_powers(dimension, order, nullity)
    kernel = _thin_plate_kernel(centers, centers, order, xp=np)
    eigenvalues, eigenvectors = _eigen_largest_magnitude(kernel, k)
    polynomial_at_centers = _polynomial_basis(centers, powers, xp=np)
    constraint = polynomial_at_centers.T @ eigenvectors
    reduced = _nullspace(constraint, nullity)

    transform = np.zeros((centers.shape[0] + nullity, k))
    transform[: centers.shape[0], : k - nullity] = eigenvectors @ reduced
    transform[centers.shape[0] :, k - nullity :] = np.eye(nullity)
    penalty = np.zeros((k, k))
    penalty[: k - nullity, : k - nullity] = reduced.T @ (eigenvalues[:, None] * reduced)

    def full_basis(points: XpArray, points_centers: XpArray, xp: ModuleType) -> XpArray:
        return xp.concatenate(
            (
                _thin_plate_kernel(points, points_centers, order, xp=xp),
                _polynomial_basis(points, powers, xp=xp),
            ),
            axis=1,
        )

    def setup_basis(points: np.ndarray) -> np.ndarray:
        return full_basis(points, centers, np)

    centers_jax = jnp.asarray(centers)
    shift_jax = jnp.asarray(shift)

    squares = np.zeros(k)
    for block in _row_blocks(centered):
        squares += np.sum((setup_basis(block) @ transform) ** 2, axis=0)
    rms = np.sqrt(squares / x_array.shape[0])
    transform = transform / rms
    penalty = penalty / rms[:, None] / rms[None, :]

    def root(values: ArrayLike) -> Array:
        return full_basis(_as_matrix(values) - shift_jax, centers_jax, jnp)

    if shrinkage:
        penalty = _shrink_null_space(penalty, 0.1)

    knot_array = jnp.asarray(centers + shift)
    if not remove_null_space:
        return _smooth_from_setup(root, penalty, transform=transform, knots=knot_array)

    kept = k - nullity
    kept_transform = transform[:, :kept]
    totals = np.zeros(kept)
    for block in _row_blocks(centered):
        totals += np.sum(setup_basis(block) @ kept_transform, axis=0)
    training_mean = jnp.asarray(totals / x_array.shape[0])
    kept_transform_jax = jnp.asarray(kept_transform)

    def centered_root(values: ArrayLike) -> Array:
        return root(values) @ kept_transform_jax - training_mean

    return _smooth_from_setup(centered_root, penalty[:kept, :kept], knots=knot_array)


def _pairwise_distance(x: XpArray, centers: XpArray, xp: ModuleType = jnp) -> XpArray:
    squared = xp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    safe_squared = xp.where(squared > 0.0, squared, 1.0)
    return xp.where(squared > 0.0, xp.sqrt(safe_squared), 0.0)


def _gp_kernel(
    x: XpArray,
    centers: XpArray,
    kernel_name: str,
    range_: float,
    power: float,
    xp: ModuleType = jnp,
) -> XpArray:
    distance = _pairwise_distance(x, centers, xp=xp) / range_
    exponential = xp.exp(-distance)
    if kernel_name == "spherical":
        return (1.0 - 1.5 * distance + 0.5 * distance**3) * (distance <= 1.0)
    if kernel_name == "power_exponential":
        return xp.exp(-(distance**power))
    if kernel_name == "matern1.5":
        return (1.0 + distance) * exponential
    if kernel_name == "matern2.5":
        return exponential + distance * exponential * (1.0 + distance / 3.0)
    if kernel_name == "matern3.5":
        return exponential + distance * exponential * (
            1.0 + 0.4 * distance + distance**2 / 15.0
        )
    raise ValueError(f"Unknown GP kernel {kernel_name!r}.")


def gaussian_process(
    x: ArrayLike,
    *,
    k: int,
    kernel_name: str,
    linear_trend: bool,
    range_: float | None,
    power: float,
    knots: ArrayLike | None = None,
) -> Smooth:
    """Create a smooth in which nearby inputs tend to have similar effects.

    The selected kernel and range control how quickly that similarity fades
    with distance. This deterministic, low-rank version includes a constant or
    linear trend and follows mgcv's fixed-range construction.

    Parameters
    ----------
    x
        Covariate vector or matrix with observations in rows and dimensions in
        columns.
    k
        Requested total basis dimension. It is raised when necessary to include
        the chosen trend and at least one penalized direction.
    kernel_name
        One of ``"spherical"``, ``"power_exponential"``, ``"matern1.5"``,
        ``"matern2.5"``, or ``"matern3.5"``.
    linear_trend
        Whether to include linear covariate columns in addition to an intercept.
    range_
        Positive kernel range. ``None`` or a non-positive value uses the maximum
        pairwise distance between centers.
    power
        Exponent for the ``"power_exponential"`` kernel; ignored by the other
        kernels.
    knots
        Optional kernel centers, as a vector in one dimension or a matrix in
        multiple dimensions. ``None`` uses unique covariate locations.

    Returns
    -------
    smooth :
        A low-rank Gaussian-process smooth whose ``knots`` contain its centers.

    Raises
    ------
    ValueError
        If the kernel name, covariates, or custom knots are invalid, or too few
        unique locations support the requested basis.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import gaussian_process
    >>> x = jnp.linspace(0.0, 1.0, 10)
    >>> smooth = gaussian_process(
    ...     x,
    ...     k=5,
    ...     kernel_name="matern1.5",
    ...     linear_trend=True,
    ...     range_=None,
    ...     power=1.5,
    ... )
    >>> smooth.basis(x).shape
    (10, 5)
    >>> smooth.penalty.shape
    (5, 5)
    >>> smooth.rank
    3

    ```
    """
    x_array = _as_float64_matrix(x)
    dimension = x_array.shape[1]
    shift = np.mean(x_array, axis=0)
    centered = x_array - shift
    if knots is None:
        centers = _unique_locations(centered)
    else:
        supplied = np.asarray(knots, dtype=np.float64)
        if dimension != 1 and supplied.ndim != 2:
            raise ValueError("Multidimensional custom knots are not supported.")
        centers = np.unique(supplied.reshape(-1, dimension) - shift, axis=0)
    if np.unique(centered, axis=0).shape[0] < k:
        raise ValueError(
            "Fewer unique covariate combinations than the requested basis dimension."
        )
    nullity = dimension + 1 if linear_trend else 1
    k = max(k, nullity + 1)
    penalized_dimension = k - nullity
    actual_range = (
        float(np.max(_pairwise_distance(centers, centers, xp=np)))
        if range_ is None or range_ <= 0.0
        else float(range_)
    )
    covariance = _gp_kernel(centers, centers, kernel_name, actual_range, power, xp=np)
    eigenvalues, eigenvectors = _eigen_largest_magnitude(
        covariance, penalized_dimension
    )
    penalty = np.diag(np.concatenate((eigenvalues, np.zeros(nullity))))
    transform = np.zeros((centers.shape[0] + nullity, k))
    transform[: centers.shape[0], :penalized_dimension] = eigenvectors
    transform[centers.shape[0] :, penalized_dimension:] = np.eye(nullity)
    centers_jax = jnp.asarray(centers)
    shift_jax = jnp.asarray(shift)

    def root(values: ArrayLike) -> Array:
        points = _as_matrix(values) - shift_jax
        radial = _gp_kernel(points, centers_jax, kernel_name, actual_range, power)
        trend = jnp.ones((points.shape[0], 1), dtype=points.dtype)
        if linear_trend:
            trend = jnp.concatenate((trend, points), axis=1)
        return jnp.concatenate((radial, trend), axis=1)

    # The penalty is the retained covariance spectrum, so every retained
    # eigen-direction is penalized, however small its eigenvalue.
    return _smooth_from_setup(
        root,
        penalty,
        rank=penalized_dimension,
        transform=transform,
        knots=jnp.asarray(centers + shift),
    )
