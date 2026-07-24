# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Adapted from mgcv 1.9-4, commit
# 1b6a4c8374612da27e36420b4459e93acb183f2d,
# R/smooth.r and src/tprs.c.
# Python/JAX adaptation modified 2026-07-24.

"""Thin-plate and Gaussian-process smooth constructors."""

import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import lobpcg_standard

from ._smooth import Array, ArrayLike, Smooth


def _as_matrix(x: ArrayLike) -> Array:
    array = jnp.asarray(x)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a vector or matrix, got shape {array.shape}.")
    return array


def _canonicalize_columns(vectors: Array) -> Array:
    pivots = jnp.argmax(jnp.abs(vectors), axis=0)
    values = jnp.take_along_axis(
        vectors, jnp.expand_dims(pivots, axis=0), axis=0
    ).squeeze(axis=0)
    return vectors * jnp.where(values < 0.0, -1.0, 1.0)


def _eigen_largest_magnitude(matrix: Array, n_vectors: int) -> tuple[Array, Array]:
    symmetric = (matrix + matrix.T) / 2.0
    dimension = symmetric.shape[0]
    if dimension <= 256 or 5 * n_vectors >= dimension:
        values, vectors = jnp.linalg.eigh(symmetric)
        indices = jnp.argsort(jnp.abs(values))[::-1][:n_vectors]
        values = values[indices]
        vectors = vectors[:, indices]
    else:
        initial = jax.random.normal(
            jax.random.key(0), (dimension, n_vectors), dtype=symmetric.dtype
        )

        def squared_action(value: Array) -> Array:
            return symmetric @ (symmetric @ value)

        _, subspace, _ = lobpcg_standard(squared_action, initial, m=100)
        projected = subspace.T @ symmetric @ subspace
        projected_values, projected_vectors = jnp.linalg.eigh(projected)
        indices = jnp.argsort(jnp.abs(projected_values))[::-1]
        values = projected_values[indices]
        vectors = (subspace @ projected_vectors)[:, indices]
    return values, _canonicalize_columns(vectors)


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


def _polynomial_basis(x: Array, powers: np.ndarray) -> Array:
    powers_array = jnp.asarray(powers)
    return jnp.prod(x[:, None, :] ** powers_array[None, :, :], axis=2)


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


def _thin_plate_kernel(x: Array, centers: Array, order: int) -> Array:
    dimension = x.shape[1]
    squared_distance = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    constant = _eta_constant(order, dimension)
    if dimension % 2 == 0:
        exponent = order - dimension // 2
        safe_distance = jnp.where(squared_distance > 0.0, squared_distance, 1.0)
        result = constant * 0.5 * jnp.log(safe_distance) * safe_distance**exponent
    else:
        exponent = order - dimension // 2 - 1
        result = constant * squared_distance**exponent * jnp.sqrt(squared_distance)
    return jnp.where(squared_distance > 0.0, result, 0.0)


def _nullspace(matrix: np.ndarray, dimension: int) -> np.ndarray:
    _, _, right = np.linalg.svd(matrix, full_matrices=True)
    return right.T[:, dimension:]


def _shrink_null_space(penalty: Array, shrink: float) -> Array:
    values, vectors = jnp.linalg.eigh((penalty + penalty.T) / 2.0)
    tolerance = jnp.max(values) * jnp.finfo(values.dtype).eps ** 0.8
    smallest = jnp.min(values[values > tolerance])
    values = jnp.where(values > tolerance, values, smallest * shrink)
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
    """Construct a low-rank thin-plate regression spline."""
    x_array = _as_matrix(x)
    dimension = x_array.shape[1]
    order, nullity = _null_space_dimension(dimension, penalty_order)
    k = max(k, nullity + 1)

    shift = jnp.mean(x_array, axis=0)
    centered = x_array - shift
    centered_np = np.asarray(centered, dtype=float)
    if knots is None:
        centers_np = _unique_locations(centered_np)
    else:
        supplied = np.asarray(knots, dtype=float)
        if dimension != 1 and supplied.ndim != 2:
            raise ValueError("Multidimensional custom knots are not supported.")
        supplied = supplied.reshape(-1, dimension) - np.asarray(shift)
        centers_np = (
            _unique_locations(centered_np)
            if supplied.shape[0] < k or supplied.shape[0] > x_array.shape[0]
            else np.unique(supplied, axis=0)
        )
    if centers_np.shape[0] < k:
        raise ValueError(
            "Fewer unique covariate combinations than the requested basis dimension."
        )

    centers = jnp.asarray(centers_np, dtype=x_array.dtype)
    powers = _polynomial_powers(dimension, order, nullity)
    kernel = _thin_plate_kernel(centers, centers, order)
    eigenvalues, eigenvectors = _eigen_largest_magnitude(kernel, k)
    polynomial_at_centers = _polynomial_basis(centers, powers)
    constraint = np.asarray(polynomial_at_centers.T @ eigenvectors)
    reduced = jnp.asarray(_nullspace(constraint, nullity), dtype=x_array.dtype)

    radial_transform = eigenvectors @ reduced
    transform = jnp.zeros((centers.shape[0] + nullity, k), dtype=x_array.dtype)
    transform = transform.at[: centers.shape[0], : k - nullity].set(radial_transform)
    transform = transform.at[centers.shape[0] :, k - nullity :].set(
        jnp.eye(nullity, dtype=x_array.dtype)
    )
    range_penalty = reduced.T @ (eigenvalues[:, None] * reduced)
    penalty = jnp.zeros((k, k), dtype=x_array.dtype)
    penalty = penalty.at[: k - nullity, : k - nullity].set(range_penalty)

    def raw_evaluate(values: ArrayLike, basis_transform: Array) -> Array:
        points = _as_matrix(values) - shift
        full = jnp.concatenate(
            (
                _thin_plate_kernel(points, centers, order),
                _polynomial_basis(points, powers),
            ),
            axis=1,
        )
        return full @ basis_transform

    training_basis = raw_evaluate(x_array, transform)
    rms = jnp.sqrt(jnp.mean(training_basis**2, axis=0))
    transform = transform / rms
    penalty = penalty / rms[:, None] / rms[None, :]

    def evaluate(values: ArrayLike) -> Array:
        return raw_evaluate(values, transform)

    rank = k - nullity
    if shrinkage:
        penalty = _shrink_null_space(penalty, 0.1)
        rank = k

    if remove_null_space:
        kept = k - nullity
        training_mean = jnp.mean(evaluate(x_array)[:, :kept], axis=0)
        unrestricted = evaluate

        def evaluate(values: ArrayLike) -> Array:
            return unrestricted(values)[:, :kept] - training_mean

        penalty = penalty[:kept, :kept]
        rank = kept

    return Smooth(evaluate, penalty, rank, centers + shift)


def _pairwise_distance(x: Array, centers: Array) -> Array:
    squared = jnp.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    safe_squared = jnp.where(squared > 0.0, squared, 1.0)
    return jnp.where(squared > 0.0, jnp.sqrt(safe_squared), 0.0)


def _gp_kernel(
    x: Array,
    centers: Array,
    kernel_name: str,
    range_: Array,
    power: float,
) -> Array:
    distance = _pairwise_distance(x, centers) / range_
    exponential = jnp.exp(-distance)
    if kernel_name == "spherical":
        return (1.0 - 1.5 * distance + 0.5 * distance**3) * (distance <= 1.0)
    if kernel_name == "power_exponential":
        return jnp.exp(-(distance**power))
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
    """Construct the low-rank fixed-range GP smooth used by mgcv."""
    x_array = _as_matrix(x)
    dimension = x_array.shape[1]
    shift = jnp.mean(x_array, axis=0)
    centered = x_array - shift
    centered_np = np.asarray(centered, dtype=float)
    if knots is None:
        centers_np = _unique_locations(centered_np)
    else:
        supplied = np.asarray(knots, dtype=float)
        if dimension != 1 and supplied.ndim != 2:
            raise ValueError("Multidimensional custom knots are not supported.")
        centers_np = np.unique(
            supplied.reshape(-1, dimension) - np.asarray(shift), axis=0
        )
    if np.unique(centered_np, axis=0).shape[0] < k:
        raise ValueError(
            "Fewer unique covariate combinations than the requested basis dimension."
        )
    centers = jnp.asarray(centers_np, dtype=x_array.dtype)
    nullity = dimension + 1 if linear_trend else 1
    k = max(k, nullity + 1)
    penalized_dimension = k - nullity
    raw_distances = _pairwise_distance(centers, centers)
    actual_range = (
        jnp.max(raw_distances)
        if range_ is None or range_ <= 0.0
        else jnp.asarray(range_, dtype=x_array.dtype)
    )
    covariance = _gp_kernel(centers, centers, kernel_name, actual_range, power)
    eigenvalues, eigenvectors = _eigen_largest_magnitude(
        covariance, penalized_dimension
    )
    penalty = jnp.diag(
        jnp.concatenate((eigenvalues, jnp.zeros(nullity, dtype=eigenvalues.dtype)))
    )

    def evaluate(values: ArrayLike) -> Array:
        points = _as_matrix(values) - shift
        radial = (
            _gp_kernel(points, centers, kernel_name, actual_range, power) @ eigenvectors
        )
        trend = jnp.ones((points.shape[0], 1), dtype=points.dtype)
        if linear_trend:
            trend = jnp.concatenate((trend, points), axis=1)
        return jnp.concatenate((radial, trend), axis=1)

    return Smooth(evaluate, penalty, penalized_dimension, centers + shift)
