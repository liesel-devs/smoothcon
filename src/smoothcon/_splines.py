# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Adapted from liesel.contrib.splines (MIT; copyright Paul Wiemann and
# Hannes Riebl) and mgcv 1.9-4, commit
# 1b6a4c8374612da27e36420b4459e93acb183f2d, R/smooth.r.
# Python/JAX adaptation modified 2026-07-24.

"""JAX B-spline primitives."""

import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def equidistant_knots(
    x: ArrayLike, n_param: int, order: int = 3, eps: float = 0.01
) -> Array:
    """Construct an extended equidistant knot sequence."""
    if order < 0:
        raise ValueError(f"Invalid {order=}.")
    if n_param < order:
        raise ValueError(f"{n_param=} must not be smaller than {order=}.")

    x = jnp.asarray(x)
    n_internal = n_param - order + 1
    lower = jnp.min(x)
    upper = jnp.max(x)
    data_range = upper - lower
    lower = lower - data_range * (eps / 2.0)
    upper = upper + data_range * (eps / 2.0)
    internal = jnp.linspace(lower, upper, n_internal)
    step = internal[1] - internal[0]
    left = lower - step * jnp.arange(order, 0, -1)
    right = upper + step * jnp.arange(1, order + 1)
    return jnp.concatenate((left, internal, right))


def _divide(numerator: Array, denominator: Array) -> Array:
    safe_denominator = jnp.where(denominator == 0.0, 1.0, denominator)
    return jnp.where(denominator == 0.0, 0.0, numerator / safe_denominator)


def _basis_vector_derivative(
    x: Array, knots: Array, degree: int, derivative: int
) -> Array:
    values: dict[tuple[int, int], Array] = {}
    values[(0, 0)] = ((x >= knots[:-1]) & (x < knots[1:])).astype(knots.dtype)
    for derivative_order in range(1, derivative + 1):
        values[(0, derivative_order)] = jnp.zeros_like(values[(0, 0)])

    for current_degree in range(1, degree + 1):
        previous = values[(current_degree - 1, 0)]
        n_basis = knots.shape[0] - current_degree - 1
        left_denominator = (
            knots[current_degree : current_degree + n_basis] - knots[:n_basis]
        )
        right_denominator = (
            knots[current_degree + 1 : current_degree + n_basis + 1]
            - knots[1 : n_basis + 1]
        )
        left = _divide((x - knots[:n_basis]) * previous[:n_basis], left_denominator)
        right = _divide(
            (knots[current_degree + 1 : current_degree + n_basis + 1] - x)
            * previous[1 : n_basis + 1],
            right_denominator,
        )
        values[(current_degree, 0)] = left + right

        for derivative_order in range(1, derivative + 1):
            previous_derivative = values[(current_degree - 1, derivative_order - 1)]
            left_derivative = _divide(
                current_degree * previous_derivative[:n_basis], left_denominator
            )
            right_derivative = _divide(
                current_degree * previous_derivative[1 : n_basis + 1],
                right_denominator,
            )
            values[(current_degree, derivative_order)] = (
                left_derivative - right_derivative
            )

    return values[(degree, derivative)]


def basis_matrix(
    x: ArrayLike,
    knots: ArrayLike,
    order: int = 3,
    *,
    outer_ok: bool = False,
    derivative: int = 0,
) -> Array:
    """Evaluate a B-spline basis or derivative matrix."""
    if order < 0:
        raise ValueError(f"Invalid {order=}.")
    if derivative < 0:
        raise ValueError(f"Invalid {derivative=}.")
    if derivative > order:
        x_array = jnp.atleast_1d(jnp.asarray(x))
        n_basis = jnp.asarray(knots).shape[0] - order - 1
        return jnp.zeros((x_array.shape[0], n_basis), dtype=x_array.dtype)

    x_array = jnp.atleast_1d(jnp.asarray(x))
    knots_array = jnp.sort(jnp.asarray(knots, dtype=x_array.dtype))
    if not outer_ok:
        lower = knots_array[order]
        upper = knots_array[-order - 1]
        if bool(jnp.any(x_array < lower) | jnp.any(x_array > upper)):
            raise ValueError(
                "Values of x are not inside the range of interior knots, "
                f"[{lower}, {upper}]."
            )

    def evaluate(value: Array) -> Array:
        return _basis_vector_derivative(value, knots_array, order, derivative)

    return jax.vmap(evaluate)(x_array)


def pspline_penalty(d: int, diff: int = 2) -> Array:
    """Construct a coefficient-difference penalty."""
    if diff < 0:
        raise ValueError(f"Invalid {diff=}.")
    differences = jnp.diff(jnp.eye(d), n=diff, axis=0)
    return differences.T @ differences


def linear_extrapolation_basis(
    x: ArrayLike, knots: ArrayLike, degree: int, derivative: int = 0
) -> Array:
    """Evaluate a B-spline with mgcv's linear boundary extrapolation."""
    x_array = jnp.atleast_1d(jnp.asarray(x))
    knots_array = jnp.asarray(knots, dtype=x_array.dtype)
    lower = knots_array[degree]
    upper = knots_array[-degree - 1]

    interior_x = jnp.clip(x_array, lower, upper)
    interior = basis_matrix(
        interior_x,
        knots_array,
        order=degree,
        outer_ok=True,
        derivative=derivative,
    )
    lower_value = basis_matrix(
        jnp.asarray([lower]),
        knots_array,
        order=degree,
        outer_ok=True,
    )[0]
    upper_value = basis_matrix(
        jnp.asarray([upper]),
        knots_array,
        order=degree,
        outer_ok=True,
    )[0]
    lower_slope = basis_matrix(
        jnp.asarray([lower]),
        knots_array,
        order=degree,
        outer_ok=True,
        derivative=1,
    )[0]
    upper_slope = basis_matrix(
        jnp.asarray([upper]),
        knots_array,
        order=degree,
        outer_ok=True,
        derivative=1,
    )[0]

    if derivative == 0:
        left = lower_value + (x_array[:, None] - lower) * lower_slope
        right = upper_value + (x_array[:, None] - upper) * upper_slope
    elif derivative == 1:
        left = jnp.broadcast_to(lower_slope, interior.shape)
        right = jnp.broadcast_to(upper_slope, interior.shape)
    else:
        left = jnp.zeros_like(interior)
        right = jnp.zeros_like(interior)

    return jnp.where(
        (x_array < lower)[:, None],
        left,
        jnp.where((x_array > upper)[:, None], right, interior),
    )


def wrap_periodic(x: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> Array:
    """Wrap points into a closed interval following mgcv's ``cwrap`` rules."""
    x_array = jnp.asarray(x)
    lower_array = jnp.asarray(lower, dtype=x_array.dtype)
    upper_array = jnp.asarray(upper, dtype=x_array.dtype)
    period = upper_array - lower_array
    above = lower_array + jnp.mod(x_array - upper_array, period)
    below = upper_array - jnp.mod(lower_array - x_array, period)
    return jnp.where(
        x_array > upper_array,
        above,
        jnp.where(x_array < lower_array, below, x_array),
    )


def cyclic_basis_matrix(x: ArrayLike, knots: ArrayLike, degree: int) -> Array:
    """Evaluate mgcv's cyclic B-spline construction."""
    x_array = jnp.atleast_1d(jnp.asarray(x))
    knots_array = jnp.sort(jnp.asarray(knots, dtype=x_array.dtype))
    lower = knots_array[0]
    upper = knots_array[-1]
    x_array = wrap_periodic(x_array, lower, upper)

    leading = lower - (upper - knots_array[-degree - 1 : -1])
    augmented = jnp.concatenate((leading, knots_array))
    wrap_from = knots_array[-degree - 1]
    primary = basis_matrix(x_array, augmented, order=degree, outer_ok=True)
    wrapped = basis_matrix(
        x_array - upper + lower,
        augmented,
        order=degree,
        outer_ok=True,
    )
    return primary + jnp.where((x_array > wrap_from)[:, None], wrapped, 0.0)


def cyclic_difference_penalty(d: int, diff: int) -> Array:
    """Construct mgcv's wrapped coefficient-difference penalty."""
    if diff < 0 or diff > d - 1:
        raise ValueError("Penalty order is incompatible with the basis dimension.")
    extended = jnp.diff(jnp.eye(d + diff), n=diff, axis=0)
    if diff == 0:
        differences = extended
    else:
        differences = extended[:, diff:]
        differences = differences.at[:, -diff:].add(extended[:, :diff])
    return differences.T @ differences
