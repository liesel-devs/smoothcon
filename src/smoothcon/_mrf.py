# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem
#
# Adapted from mgcv 1.9-4, commit
# 1b6a4c8374612da27e36420b4459e93acb183f2d, R/smooth.r.
# Python/JAX adaptation modified 2026-07-24.

"""Markov-random-field smooth construction."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._smooth import Array, ArrayLike, Smooth


def polygon_neighbors(
    polygons: Mapping[str, ArrayLike],
) -> dict[str, list[str]]:
    """Infer polygon neighbors from shared vertices.

    Two regions are adjacent when their coordinate arrays contain at least one
    exactly equal vertex. Rows containing ``NaN`` are ignored, which permits
    common multipart-polygon separators.

    Parameters
    ----------
    polygons
        Mapping from region labels to two-column coordinate arrays.

    Returns
    -------
    neighbors :
        Symmetric mapping from each label to its neighboring labels.

    Raises
    ------
    ValueError
        If a polygon is not a two-column coordinate array.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from smoothcon import polygon_neighbors
    >>> polygons = {
    ...     "left": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    ...     "right": np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
    ... }
    >>> neighbors = polygon_neighbors(polygons)
    >>> assert neighbors == {"left": ["right"], "right": ["left"]}

    ```
    """
    vertices: dict[str, set[tuple[float, float]]] = {}
    for label, polygon in polygons.items():
        array = np.asarray(polygon, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(
                f"Polygon {label!r} must be a two-column coordinate array."
            )
        array = array[~np.isnan(array).any(axis=1)]
        vertices[label] = {tuple(row) for row in np.unique(array, axis=0)}

    result: dict[str, list[str]] = {label: [] for label in polygons}
    labels = list(polygons)
    for index, left in enumerate(labels):
        for right in labels[index + 1 :]:
            if vertices[left].intersection(vertices[right]):
                result[left].append(right)
                result[right].append(left)
    return result


def normalize_neighbors(
    neighbors: Mapping[str, ArrayLike | list[str] | list[int]],
    labels: Sequence[str],
    index_labels: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Normalize label- or index-valued neighborhood definitions.

    Integer neighbors are zero-based indices into ``index_labels``. If that
    ordering is omitted, sorted region labels are used. Label-valued neighbors
    are preserved after validation.

    Parameters
    ----------
    neighbors
        Mapping with one entry per region and one-dimensional neighbor values.
    labels
        Complete set of region labels.
    index_labels
        Label order used to interpret numeric indices.

    Returns
    -------
    normalized :
        Neighborhood mapping expressed entirely with string labels.

    Raises
    ------
    ValueError
        If labels, indices, or neighbor-array dimensions are invalid.
    TypeError
        If neighbor values use an unsupported dtype.

    Examples
    --------
    ```pycon
    >>> from smoothcon import normalize_neighbors
    >>> raw = {"a": [1], "b": [0, 2], "c": [1]}
    >>> neighbors = normalize_neighbors(raw, ["a", "b", "c"])
    >>> assert neighbors["b"] == ["a", "c"]

    ```
    """
    if set(neighbors) != set(labels):
        raise ValueError("Names in 'neighbors' must correspond to the labels.")
    indexed_labels = list(index_labels) if index_labels is not None else sorted(labels)
    if set(indexed_labels) != set(labels):
        raise ValueError("'index_labels' must contain exactly the region labels.")
    normalized: dict[str, list[str]] = {}
    for label, raw in neighbors.items():
        values = np.asarray(raw)
        if values.ndim != 1:
            raise ValueError(
                f"Expected 1d neighbor arrays, got {values.ndim=} for {label}."
            )
        if values.dtype.kind in "iu":
            indices = values.astype(int)
            if np.any(indices < 0) or np.any(indices >= len(labels)):
                raise ValueError(f"Neighbor index out of range for region {label!r}.")
            normalized[label] = [indexed_labels[index] for index in indices]
        elif values.dtype.kind == "f":
            if not np.equal(values, values.astype(int)).all():
                raise ValueError("Floating point neighbor indices must be integral.")
            indices = values.astype(int)
            if np.any(indices < 0) or np.any(indices >= len(labels)):
                raise ValueError(f"Neighbor index out of range for region {label!r}.")
            normalized[label] = [indexed_labels[index] for index in indices]
        elif values.dtype.kind in "OUS":
            neighbor_labels = values.astype(str).tolist()
            unknown = set(neighbor_labels).difference(labels)
            if unknown:
                raise ValueError(f"Unknown neighbor labels: {sorted(unknown)}.")
            normalized[label] = neighbor_labels
        else:
            raise TypeError(f"Unsupported dtype: {values.dtype!r}")
    return normalized


def laplacian(
    neighbors: Mapping[str, Sequence[str]], labels: Sequence[str]
) -> np.ndarray:
    """Construct a graph-Laplacian penalty from a neighborhood mapping.

    The diagonal records each region's number of distinct neighbors and a
    neighboring pair receives ``-1`` off the diagonal. The order of ``labels``
    determines the matrix rows and columns.

    Parameters
    ----------
    neighbors
        Symmetric mapping from region labels to neighboring labels.
    labels
        Region order for the output matrix.

    Returns
    -------
    penalty :
        Symmetric graph-Laplacian matrix.

    Raises
    ------
    ValueError
        If the neighborhood relation is not symmetric.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from smoothcon import laplacian
    >>> neighbors = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    >>> penalty = laplacian(neighbors, ["a", "b", "c"])
    >>> assert np.array_equal(penalty.sum(axis=1), np.zeros(3))
    >>> assert np.linalg.matrix_rank(penalty) == 2

    ```
    """
    lookup = {label: index for index, label in enumerate(labels)}
    penalty = np.zeros((len(labels), len(labels)), dtype=float)
    for label, adjacent in neighbors.items():
        row = lookup[label]
        distinct = list(dict.fromkeys(adjacent))
        penalty[row, row] = len(distinct)
        for neighbor in distinct:
            column = lookup[neighbor]
            if column != row:
                penalty[row, column] = -1.0
    if not np.array_equal(penalty, penalty.T):
        raise ValueError("The supplied neighborhood relation must be symmetric.")
    return penalty


def _rank(penalty: np.ndarray) -> int:
    values = np.linalg.eigvalsh((penalty + penalty.T) / 2.0)
    if not values.size or np.max(values) <= 0.0:
        return 0
    return int(np.sum(values > np.finfo(float).eps ** 0.8 * np.max(values)))


def _natural_low_rank(
    codes: np.ndarray, penalty: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, int]:
    n_regions = penalty.shape[0]
    observed = np.zeros((codes.size, n_regions), dtype=float)
    observed[np.arange(codes.size), codes] = 1.0
    missing = np.flatnonzero(np.sum(observed, axis=0) == 0.0)
    if missing.size:
        dummy = np.zeros((missing.size, n_regions), dtype=float)
        dummy[np.arange(missing.size), missing] = 1.0
        augmented = np.vstack((dummy, observed))
    else:
        augmented = observed

    _, triangular = np.linalg.qr(augmented, mode="reduced")
    inverse = np.linalg.inv(triangular)
    natural_penalty = inverse.T @ penalty @ inverse
    values, vectors = np.linalg.eigh((natural_penalty + natural_penalty.T) / 2.0)
    order = np.argsort(values)[::-1]
    values = values[order]
    transform = np.linalg.solve(triangular, vectors[:, order])
    natural_design = augmented @ transform
    rank = _rank(penalty)

    if rank:
        scale = 1.0 / np.sqrt(np.mean(natural_design[:, :rank] ** 2))
        natural_design[:, :rank] *= scale
        transform[:, :rank] *= scale
        values[:rank] *= scale**2
    if rank < n_regions:
        scale = 1.0 / np.sqrt(np.mean(natural_design[:, rank:] ** 2))
        transform[:, rank:] *= scale

    selected = np.arange(n_regions - k, n_regions)
    selected_values = np.where(selected < rank, values[selected], 0.0)
    return (
        transform[:, selected],
        np.diag(selected_values),
        int(np.sum(selected < rank)),
    )


def mrf(codes: ArrayLike, *, penalty: ArrayLike, k: int) -> Smooth:
    """Construct a full- or low-rank Markov-random-field smooth.

    The full basis is a one-hot encoding of region codes. A smaller ``k`` uses
    the natural low-rank parameterization of the supplied graph penalty, which
    is useful when the number of regions is large.

    Parameters
    ----------
    codes
        Zero-based integer region codes for the observed values.
    penalty
        Square graph penalty aligned with the region-code ordering.
    k
        Basis dimension. Use ``-1`` or the number of regions for the full basis,
        or a smaller positive value for a low-rank basis.

    Returns
    -------
    smooth :
        An MRF smooth whose basis evaluates integer region codes.

    Raises
    ------
    ValueError
        If ``k`` exceeds the number of regions.

    Examples
    --------
    ```pycon
    >>> import numpy as np
    >>> from smoothcon import laplacian, mrf
    >>> neighbors = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    >>> penalty = laplacian(neighbors, ["a", "b", "c"])
    >>> smooth = mrf(np.array([0, 1, 2, 1]), penalty=penalty, k=-1)
    >>> assert smooth.basis(np.array([0, 2])).shape == (2, 3)
    >>> assert smooth.rank == 2

    ```
    """
    codes_array = np.asarray(codes, dtype=int).reshape(-1)
    penalty_array = np.asarray(penalty, dtype=float)
    n_regions = penalty_array.shape[0]
    if k > n_regions:
        raise ValueError("MRF basis dimension is larger than the number of regions.")
    if k == -1 or k == n_regions:
        transform = np.eye(n_regions)
        output_penalty = penalty_array
        rank = _rank(penalty_array)
    else:
        transform, output_penalty, rank = _natural_low_rank(
            codes_array, penalty_array, k
        )
    transform_array = jnp.asarray(transform)

    def evaluate(values: ArrayLike) -> Array:
        integer_values = jnp.asarray(values, dtype=int).reshape(-1)
        return jax.nn.one_hot(integer_values, n_regions) @ transform_array

    return Smooth(
        evaluate,
        jnp.asarray(output_penalty),
        rank,
        None,
    )
