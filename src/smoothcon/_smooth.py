# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem

"""The public numerical representation of a smooth term."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


@dataclass(frozen=True)
class Smooth:
    """A basis evaluator and its quadratic penalty."""

    basis: Callable[[ArrayLike], Array]
    penalty: Array
    rank: int
    knots: Array | None = None

    def _reparameterize(self, transform: Array) -> "Smooth":
        basis = self.basis
        penalty = transform.T @ self.penalty @ transform

        def evaluate(values: ArrayLike) -> Array:
            return basis(values) @ transform

        rank = int(np.linalg.matrix_rank(np.asarray(penalty)))
        return Smooth(evaluate, penalty, rank, self.knots)

    def constrain(
        self,
        constraint: str | ArrayLike,
        *,
        values: ArrayLike | None = None,
    ) -> "Smooth":
        """Apply a linear constraint using Kneib et al. (2019).

        Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular
        regression—A Lego system for building structured additive
        distributional regression models with tensor product interactions.
        TEST, 28(1), 1–39. https://doi.org/10.1007/s11749-019-00631-z
        """
        if not isinstance(constraint, str):
            matrix = jnp.asarray(constraint)
        else:
            if constraint == "sumzero_coef":
                matrix = jnp.ones((1, self.penalty.shape[0]))
            elif constraint == "sumzero_term":
                if values is None:
                    raise ValueError(
                        "'values' are required for a term sum-to-zero constraint."
                    )
                matrix = jnp.sum(self.basis(values), axis=0, keepdims=True)
            elif constraint == "constant_and_linear":
                if values is None:
                    raise ValueError(
                        "'values' are required to remove constant and linear trends."
                    )
                values_array = jnp.asarray(values)
                if values_array.ndim == 1:
                    values_array = values_array[:, None]
                linear_basis = jnp.column_stack(
                    (jnp.ones(values_array.shape[0]), values_array)
                )
                basis = self.basis(values)
                matrix = jnp.linalg.solve(
                    linear_basis.T @ linear_basis,
                    linear_basis.T @ basis,
                )
            else:
                raise ValueError(f"Unknown constraint {constraint!r}.")
        transform = _constraint_reparameterization(matrix)
        return self._reparameterize(transform)

    def scale_penalty(self, *, values: ArrayLike) -> "Smooth":
        """Scale the penalty relative to the evaluated basis."""
        design_size = jnp.linalg.norm(self.basis(values), ord=jnp.inf) ** 2
        penalty_size = jnp.linalg.norm(self.penalty, ord=1)
        if not bool(jnp.isfinite(design_size)) or float(design_size) <= 0.0:
            raise ValueError("Cannot scale a penalty for a zero or non-finite basis.")
        if not bool(jnp.isfinite(penalty_size)) or float(penalty_size) <= 0.0:
            raise ValueError("Cannot scale a zero or non-finite penalty matrix.")
        return Smooth(
            self.basis,
            self.penalty * design_size / penalty_size,
            self.rank,
            self.knots,
        )

    def diagonalize_penalty(self) -> "Smooth":
        """Reparameterize the penalty to ones followed by zeros."""
        eigenvalues, eigenvectors = jnp.linalg.eigh(
            (self.penalty + self.penalty.T) / 2.0
        )
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        pivots = jnp.argmax(jnp.abs(eigenvectors), axis=0)
        pivot_values = jnp.take_along_axis(
            eigenvectors, pivots[None, :], axis=0
        ).squeeze(axis=0)
        eigenvectors = eigenvectors * jnp.where(pivot_values < 0.0, -1.0, 1.0)
        penalized = jnp.arange(eigenvalues.shape[0]) < self.rank
        safe_eigenvalues = jnp.where(penalized, eigenvalues, 1.0)
        if not bool(jnp.all(safe_eigenvalues > 0.0)):
            raise ValueError("Penalty must be positive semidefinite.")
        transform = eigenvectors * jnp.where(
            penalized,
            1.0 / jnp.sqrt(safe_eigenvalues),
            1.0,
        )
        basis = self.basis

        def evaluate(values: ArrayLike) -> Array:
            return basis(values) @ transform

        target = jnp.diag(penalized.astype(self.penalty.dtype))
        return Smooth(evaluate, target, self.rank, self.knots)


def _constraint_reparameterization(constraint: Array) -> Array:
    if constraint.ndim != 2:
        raise ValueError("A constraint matrix must be two-dimensional.")
    n_constraints, n_coefficients = constraint.shape
    if not 0 < n_constraints < n_coefficients:
        raise ValueError("A constraint must leave at least one free coefficient.")
    if np.linalg.matrix_rank(np.asarray(constraint)) != n_constraints:
        raise ValueError("Constraint rows must be linearly independent.")

    _, eigenvectors = jnp.linalg.eigh(constraint.T @ constraint)
    signs = jnp.sign(eigenvectors[0])
    eigenvectors = eigenvectors * jnp.where(signs == 0, 1.0, signs)
    complement = eigenvectors[:, :-n_constraints].T
    stacked = jnp.concatenate((constraint, complement), axis=0)
    inverse = jnp.linalg.inv(stacked)
    return inverse[:, n_constraints:]
