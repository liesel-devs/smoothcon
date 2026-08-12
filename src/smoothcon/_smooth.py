# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem

"""Store and transform a smooth's basis and penalty."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


@dataclass(frozen=True)
class Smooth:
    """Hold the two pieces needed to use a smooth in a model.

    Call ``basis`` to turn input values into model columns. Use ``penalty`` to
    discourage unnecessarily wiggly coefficient patterns. A ``Smooth`` never
    changes in place; each transformation returns a new one.

    Parameters
    ----------
    basis
        Function mapping covariate values to a design matrix with one row per
        value and one column per coefficient.
    penalty
        Square matrix defining the coefficient penalty.
    rank
        Numerical rank of ``penalty``.
    knots
        Knot or center locations retained by the constructor, when applicable.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from smoothcon import pspline
    >>> x = jnp.linspace(0.0, 1.0, 8)
    >>> smooth = pspline(x, k=5, degree=3, penalty_order=2)
    >>> smooth.basis(x).shape
    (8, 5)
    >>> smooth.penalty.shape
    (5, 5)
    >>> smooth.rank
    3

    ```
    """

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
        """Remove patterns that the smooth should not be allowed to represent.

        ``"sumzero_coef"`` makes the coefficients add to zero,
        ``"sumzero_term"`` makes the fitted values at ``values`` add to zero,
        and ``"constant_and_linear"`` removes constant and linear trends. A
        matrix can describe any other rule as ``A @ coefficients == 0``.

        Parameters
        ----------
        constraint
            Built-in constraint name or a full-row-rank constraint matrix.
        values
            Covariate values used by term-based constraints. Required for
            ``"sumzero_term"`` and ``"constant_and_linear"``.

        Returns
        -------
        smooth :
            A reparameterized smooth with one column removed per constraint.

        Raises
        ------
        ValueError
            If the constraint is unknown, malformed, or does not leave a free
            coefficient, or if required values are missing.

        References
        ----------
        Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular
        regression—A Lego system for building structured additive
        distributional regression models with tensor product interactions.
        *TEST*, 28(1), 1–39. https://doi.org/10.1007/s11749-019-00631-z

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from smoothcon import pspline
        >>> x = jnp.linspace(0.0, 1.0, 8)
        >>> smooth = pspline(x, k=5, degree=3, penalty_order=2)
        >>> constrained = smooth.constrain("sumzero_coef")
        >>> smooth.basis(x).shape
        (8, 5)
        >>> constrained.basis(x).shape
        (8, 4)

        ```
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
        """Put the penalty on a scale that matches the evaluated basis.

        The penalty is multiplied by ``||B||_inf**2 / ||K||_1``. This makes its
        strength less dependent on how the basis happens to be parameterized.
        The basis itself stays unchanged.

        Parameters
        ----------
        values
            Covariate values at which to evaluate the basis scale.

        Returns
        -------
        smooth :
            A smooth with the scaled penalty.

        Raises
        ------
        ValueError
            If the evaluated basis or penalty has zero or non-finite norm.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from smoothcon import pspline
        >>> x = jnp.linspace(0.0, 1.0, 8)
        >>> smooth = pspline(x, k=5, degree=3, penalty_order=2)
        >>> scaled = smooth.scale_penalty(values=x)
        >>> scaled.basis(x).shape
        (8, 5)
        >>> scaled.penalty.shape
        (5, 5)
        >>> bool(jnp.all(jnp.isfinite(scaled.penalty)))
        True

        ```
        """
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
        """Rewrite the smooth so its penalty is diagonal and made of ones and zeros.

        Penalized directions receive a one; unpenalized directions receive a
        zero. The represented smooth stays the same, although the basis columns
        can change sign or rotate when penalty values are tied.

        Returns
        -------
        smooth :
            A reparameterized smooth with a diagonal penalty.

        Raises
        ------
        ValueError
            If a penalized eigenvalue is not positive.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from smoothcon import pspline
        >>> x = jnp.linspace(0.0, 1.0, 8)
        >>> smooth = pspline(x, k=5, degree=3, penalty_order=2)
        >>> diagonal = smooth.diagonalize_penalty()
        >>> jnp.diag(diagonal.penalty).astype(int).tolist()
        [1, 1, 1, 0, 0]

        ```
        """
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
