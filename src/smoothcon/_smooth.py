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

_SETUP_RTOL = float(np.finfo(np.float64).eps ** 0.8)
"""Relative eigenvalue cutoff for a penalty's rank in float64."""

_STORAGE_EPS_FACTOR = 10.0
"""Multiple of the storage precision's ``eps`` that a stored eigenvalue must exceed."""


@dataclass(frozen=True)
class _Setup:
    """Float64 state that transformations build on.

    ``penalty`` and ``rank`` describe the penalty in float64. ``root`` is the
    basis function before any reparameterization, and ``transform`` is the
    accumulated float64 reparameterization (``None`` for the identity), so
    ``basis(values) == root(values) @ transform``.
    """

    penalty: np.ndarray
    rank: int
    root: Callable[[ArrayLike], Array]
    transform: np.ndarray | None


@dataclass(frozen=True)
class Smooth:
    """Hold the two pieces needed to use a smooth in a model.

    Call ``basis`` to turn input values into model columns. Use ``penalty`` to
    discourage unnecessarily wiggly coefficient patterns. A ``Smooth`` never
    changes in place; each transformation returns a new one.

    Constructors and transformations do their setup in NumPy float64, whatever
    JAX's x64 mode is. ``basis`` and ``penalty`` then use JAX's default
    precision, which is float32 unless x64 is enabled.

    Parameters
    ----------
    basis
        Function mapping covariate values to a design matrix with one row per
        value and one column per coefficient.
    penalty
        Square matrix defining the coefficient penalty.
    rank
        Number of penalized directions that ``penalty`` resolves. When
        smoothcon builds a dense penalty that is stored in float32, directions
        whose eigenvalue is below ``10 * eps32`` times the largest one are not
        counted, so an ill-conditioned smooth can report a lower rank than its
        float64 setup. ``diagonalize_penalty`` keeps every direction of the
        float64 setup. A rank passed to ``Smooth`` directly is kept as given.
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

    @property
    def _state(self) -> _Setup:
        """Return the float64 setup record.

        It is kept outside the dataclass fields, so ``dataclasses.replace``
        and direct construction start from the public fields.
        """
        setup = self.__dict__.get("_setup")
        if setup is None:
            setup = _Setup(
                np.asarray(self.penalty, dtype=np.float64),
                int(self.rank),
                self.basis,
                None,
            )
            object.__setattr__(self, "_setup", setup)
        return setup

    def _design(self, values: ArrayLike) -> np.ndarray:
        """Evaluate the basis at ``values`` in float64."""
        setup = self._state
        array = np.asarray(values)
        if array.dtype.kind == "f":
            array = array.astype(np.float64)
        with jax.enable_x64(True):
            design = np.asarray(setup.root(array), dtype=np.float64)
        if setup.transform is not None:
            design = design @ setup.transform
        return design

    def _reparameterize(self, transform: np.ndarray) -> "Smooth":
        setup = self._state
        penalty = _symmetric(transform.T @ setup.penalty @ transform)
        eigenvalues = np.linalg.eigvalsh(penalty)
        rank = min(setup.rank, _rank_above(eigenvalues, _SETUP_RTOL))
        return _smooth_from_setup(
            setup.root,
            penalty,
            rank=rank,
            transform=_compose(setup.transform, transform),
            knots=self.knots,
            eigenvalues=eigenvalues,
        )

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
            Its rank drops when a constraint removes penalized directions.

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
            matrix = np.asarray(constraint, dtype=np.float64)
        else:
            if constraint == "sumzero_coef":
                matrix = np.ones((1, self._state.penalty.shape[0]))
            elif constraint == "sumzero_term":
                if values is None:
                    raise ValueError(
                        "'values' are required for a term sum-to-zero constraint."
                    )
                matrix = np.sum(self._design(values), axis=0, keepdims=True)
            elif constraint == "constant_and_linear":
                if values is None:
                    raise ValueError(
                        "'values' are required to remove constant and linear trends."
                    )
                values_array = np.asarray(values, dtype=np.float64)
                if values_array.ndim == 1:
                    values_array = values_array[:, None]
                linear_basis = np.column_stack(
                    (np.ones(values_array.shape[0]), values_array)
                )
                matrix = np.linalg.solve(
                    linear_basis.T @ linear_basis,
                    linear_basis.T @ self._design(values),
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
        setup = self._state
        design_size = np.linalg.norm(self._design(values), ord=np.inf) ** 2
        penalty_size = np.linalg.norm(setup.penalty, ord=1)
        if not np.isfinite(design_size) or design_size <= 0.0:
            raise ValueError("Cannot scale a penalty for a zero or non-finite basis.")
        if not np.isfinite(penalty_size) or penalty_size <= 0.0:
            raise ValueError("Cannot scale a zero or non-finite penalty matrix.")
        return _smooth_from_setup(
            setup.root,
            setup.penalty * (design_size / penalty_size),
            rank=setup.rank,
            transform=setup.transform,
            knots=self.knots,
        )

    def diagonalize_penalty(self, *, values: ArrayLike) -> "Smooth":
        """Rewrite the smooth so its penalty is diagonal and made of ones and zeros.

        Penalized directions receive a one; unpenalized directions receive a
        zero. Unpenalized directions are scaled to the average squared norm of
        the penalized design columns, following ``mgcv::nat.param(type=2)``.
        The represented smooth stays the same, although the basis columns can
        change sign or rotate when penalty values are tied.

        The eigendecomposition runs in float64 and uses every penalized
        direction of the float64 setup. The returned penalty holds only zeros
        and ones, so its rank survives float32 storage.

        Parameters
        ----------
        values
            Covariate values used to balance unpenalized and penalized design
            columns.

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
        >>> diagonal = smooth.diagonalize_penalty(values=x)
        >>> jnp.diag(diagonal.penalty).astype(int).tolist()
        [1, 1, 1, 0, 0]

        ```
        """
        setup = self._state
        rank = setup.rank
        eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(setup.penalty))
        eigenvalues = eigenvalues[::-1]
        eigenvectors = _canonical_signs(eigenvectors[:, ::-1])
        dimension = eigenvalues.shape[0]
        penalized = np.arange(dimension) < rank
        tolerance = np.sqrt(np.finfo(np.float64).eps) * np.max(np.abs(eigenvalues))
        if np.any(eigenvalues[:rank] < -tolerance):
            raise ValueError("Penalty must be positive semidefinite.")
        safe_eigenvalues = np.where(penalized, np.abs(eigenvalues), 1.0)
        if not np.all(safe_eigenvalues[:rank] > 0.0):
            raise ValueError("Penalized directions must have nonzero eigenvalues.")
        scales = np.where(penalized, 1.0 / np.sqrt(safe_eigenvalues), 1.0)
        transform = eigenvectors * scales
        if 0 < rank < dimension:
            design = self._design(values) @ transform
            column_norms = np.sum(design**2, axis=0)
            average_penalized_norm = np.mean(column_norms[:rank])
            scales[rank:] = np.sqrt(average_penalized_norm / column_norms[rank:])
            transform = eigenvectors * scales
        return _smooth_from_setup(
            setup.root,
            np.diag(penalized.astype(np.float64)),
            rank=rank,
            transform=_compose(setup.transform, transform),
            knots=self.knots,
        )


class _LazyEigenvalues:
    """Compute a symmetric matrix's float64 eigenvalues at most once."""

    def __init__(self, matrix: np.ndarray, eigenvalues: np.ndarray | None) -> None:
        self.matrix = matrix
        self._eigenvalues = eigenvalues

    def get(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvalsh(_symmetric(self.matrix))
        return self._eigenvalues


def _smooth_from_setup(
    root: Callable[[ArrayLike], Array],
    penalty: np.ndarray,
    *,
    rank: int | None = None,
    transform: np.ndarray | None = None,
    knots: Array | None = None,
    eigenvalues: np.ndarray | None = None,
) -> Smooth:
    """Build a smooth from float64 setup results.

    ``rank`` defaults to the float64 rank of ``penalty``. The returned
    ``penalty`` uses JAX's default precision, and the returned ``rank`` counts
    only the directions that stored copy resolves. Pass the eigenvalues of
    ``penalty`` when they are known, to skip recomputing them.
    """
    penalty = np.asarray(penalty, dtype=np.float64)
    lazy = _LazyEigenvalues(penalty, eigenvalues)
    setup_rank = _rank_above(lazy.get(), _SETUP_RTOL) if rank is None else int(rank)
    stored = jnp.asarray(penalty)
    if transform is None:
        basis = root
    else:
        transform_array = jnp.asarray(transform)

        def basis(values: ArrayLike) -> Array:
            return root(values) @ transform_array

    smooth = Smooth(basis, stored, _stored_rank(lazy, stored, setup_rank), knots)
    object.__setattr__(smooth, "_setup", _Setup(penalty, setup_rank, root, transform))
    return smooth


def _rank_above(eigenvalues: np.ndarray, rtol: float) -> int:
    """Count eigenvalues above ``rtol`` times the largest one."""
    if not eigenvalues.size:
        return 0
    largest = float(np.max(eigenvalues))
    if largest <= 0.0:
        return 0
    return int(np.sum(eigenvalues > rtol * largest))


def _setup_rank(penalty: np.ndarray) -> int:
    """Return the float64 rank of a penalty.

    Counts eigenvalues above ``eps64**0.8`` times the largest one. Every
    constructor and transformation uses this rule.
    """
    penalty = np.asarray(penalty, dtype=np.float64)
    return _rank_above(np.linalg.eigvalsh(_symmetric(penalty)), _SETUP_RTOL)


def _stored_rank(penalty: _LazyEigenvalues, stored: Array, setup_rank: int) -> int:
    """Cap ``setup_rank`` at the directions the stored penalty resolves.

    A diagonal penalty keeps its eigenvalues as entries, so it resolves every
    entry that stays positive. A dense penalty stored with ``eps`` coarser than
    float64 resolves eigenvalues above ``10 * eps`` times the largest, the same
    rule as in float64 otherwise. Eigenvalues come from the float64 penalty,
    ``penalty.matrix``.
    """
    if _is_diagonal(penalty.matrix):
        entries = np.diag(np.asarray(stored, dtype=np.float64))
        return min(setup_rank, int(np.sum(entries > 0.0)))
    eps = float(np.finfo(stored.dtype).eps)
    if eps <= np.finfo(np.float64).eps:
        return setup_rank
    rtol = max(_SETUP_RTOL, _STORAGE_EPS_FACTOR * eps)
    return min(setup_rank, _rank_above(penalty.get(), rtol))


def _is_diagonal(matrix: np.ndarray) -> bool:
    return not np.any(matrix - np.diag(np.diag(matrix)))


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _compose(first: np.ndarray | None, second: np.ndarray) -> np.ndarray:
    return second if first is None else first @ second


def _canonical_signs(vectors: np.ndarray) -> np.ndarray:
    """Flip columns so each one's largest-magnitude entry is positive."""
    pivots = np.argmax(np.abs(vectors), axis=0)
    pivot_values = vectors[pivots, np.arange(vectors.shape[1])]
    return vectors * np.where(pivot_values < 0.0, -1.0, 1.0)


def _constraint_reparameterization(constraint: np.ndarray) -> np.ndarray:
    if constraint.ndim != 2:
        raise ValueError("A constraint matrix must be two-dimensional.")
    n_constraints, n_coefficients = constraint.shape
    if not 0 < n_constraints < n_coefficients:
        raise ValueError("A constraint must leave at least one free coefficient.")
    if np.linalg.matrix_rank(constraint) != n_constraints:
        raise ValueError("Constraint rows must be linearly independent.")

    _, eigenvectors = np.linalg.eigh(constraint.T @ constraint)
    signs = np.sign(eigenvectors[0])
    eigenvectors = eigenvectors * np.where(signs == 0, 1.0, signs)
    # The complement's orthonormal columns span the constraint's null space, so
    # they are the last columns of the inverse of ``[constraint; complement.T]``.
    return eigenvectors[:, :-n_constraints]
