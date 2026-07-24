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
    """Infer neighbors from polygons sharing at least one exact vertex."""
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
    """Validate and normalize label- or index-valued neighbors."""
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
    """Construct and validate a graph-Laplacian penalty."""
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
    """Construct a full- or low-rank MRF basis from integer region codes."""
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
