"""Setup precision: float64 geometry whatever JAX's x64 mode is."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smoothcon

X64_MODES = pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])


def _chain_penalty(n_regions: int) -> np.ndarray:
    labels = [str(index) for index in range(n_regions)]
    neighbors = {
        str(index): [str(j) for j in (index - 1, index + 1) if 0 <= j < n_regions]
        for index in range(n_regions)
    }
    return smoothcon.build_mrf_penalty(neighbors, labels)


def _difference_penalty(k: int, order: int) -> np.ndarray:
    differences = np.diff(np.eye(k), n=order, axis=0)
    return differences.T @ differences


def _cr100() -> tuple[smoothcon.Smooth, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 400)
    smooth = smoothcon.cubic_regression(x, k=100)
    with jax.enable_x64(True):
        reference = np.asarray(smoothcon.cubic_regression(x, k=100).penalty)
    return smooth, reference, x


def _ps80() -> tuple[smoothcon.Smooth, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, 400)
    smooth = smoothcon.pspline(x, k=80, degree=3, penalty_order=4)
    return smooth, _difference_penalty(80, 4), x


def _mrf1000() -> tuple[smoothcon.Smooth, np.ndarray, np.ndarray]:
    penalty = _chain_penalty(1000)
    codes = np.arange(1000)
    return smoothcon.mrf(codes, penalty=penalty, k=-1), penalty, codes


CASES: dict[str, Callable[[], tuple[smoothcon.Smooth, np.ndarray, np.ndarray]]] = {
    "cr100": _cr100,
    "ps80_order4": _ps80,
    "mrf1000": _mrf1000,
}
SETUP_RANKS = {"cr100": 98, "ps80_order4": 76, "mrf1000": 999}


def _pseudo_inverse(penalty: np.ndarray, rank: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(penalty)
    values, vectors = values[::-1][:rank], vectors[:, ::-1][:, :rank]
    return (vectors / values) @ vectors.T


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@X64_MODES
@pytest.mark.parametrize("case", CASES)
def test_diagonalized_geometry_matches_float64_reference(case: str, x64: bool) -> None:
    with jax.enable_x64(x64):
        raw, reference, values = CASES[case]()
        diagonal = raw.diagonalize_penalty(values=jnp.asarray(values))
        rank = SETUP_RANKS[case]

        penalized = np.asarray(diagonal.basis(jnp.asarray(values)), float)[:, :rank]
        basis = np.asarray(raw.basis(jnp.asarray(values)), float)
        expected = basis @ _pseudo_inverse(reference, rank) @ basis.T

        assert diagonal.rank == rank
        np.testing.assert_array_equal(
            diagonal.penalty, np.diag(np.arange(reference.shape[0]) < rank)
        )
        assert diagonal.penalty.dtype == jnp.asarray(1.0).dtype
        assert _relative_error(penalized @ penalized.T, expected) < 1e-4


@X64_MODES
@pytest.mark.parametrize("case", CASES)
def test_raw_rank_is_resolved_by_the_stored_penalty(case: str, x64: bool) -> None:
    with jax.enable_x64(x64):
        raw, reference, _ = CASES[case]()
        stored = raw.penalty
        values = np.asarray(jnp.linalg.eigh(stored)[0], float)[::-1][: raw.rank]
        expected = np.linalg.eigvalsh(reference)[::-1][: raw.rank]

        assert stored.dtype == jnp.asarray(1.0).dtype
        assert raw.rank <= SETUP_RANKS[case]
        if x64:
            assert raw.rank == SETUP_RANKS[case]
        # The coefficient prior samples with 1 / sqrt of these eigenvalues.
        assert np.all(values > 0.0)
        np.testing.assert_allclose(values, expected, rtol=0.5)


def test_float32_storage_lowers_the_raw_rank_of_ill_conditioned_penalties() -> None:
    with jax.enable_x64(False):
        raw, _, values = _cr100()
        diagonal = raw.diagonalize_penalty(values=jnp.asarray(values))

    assert raw.rank < SETUP_RANKS["cr100"]
    assert diagonal.rank == SETUP_RANKS["cr100"]


@pytest.mark.parametrize(
    "build",
    [
        lambda x: smoothcon.cubic_regression(x, k=100),
        lambda x: smoothcon.pspline(x, k=80, degree=3, penalty_order=4),
        lambda x: smoothcon.bspline(x, k=60, degree=3, penalty_order=2),
        lambda x: smoothcon.cyclic_cubic(x, k=80),
        lambda x: smoothcon.cyclic_pspline(x, k=80, degree=3, penalty_order=3),
    ],
    ids=["cr", "ps", "bs", "cc", "cp"],
)
def test_transformed_smooth_matches_its_x64_construction(build) -> None:
    x = np.linspace(0.0, 1.0, 400)

    def construct() -> tuple[np.ndarray, np.ndarray, int]:
        values = jnp.asarray(x)
        smooth = (
            build(x)
            .scale_penalty(values=values)
            .constrain("sumzero_term", values=values)
            .diagonalize_penalty(values=values)
        )
        design = np.asarray(smooth.basis(values), float)
        return design, np.asarray(smooth.penalty, float), smooth.rank

    with jax.enable_x64(False):
        design32, penalty32, rank32 = construct()
    with jax.enable_x64(True):
        design64, penalty64, rank64 = construct()

    assert rank32 == rank64
    np.testing.assert_array_equal(penalty32, penalty64)
    penalized32 = design32[:, :rank32]
    penalized64 = design64[:, :rank64]
    assert (
        _relative_error(penalized32 @ penalized32.T, penalized64 @ penalized64.T)
        < 1e-4
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda x: smoothcon.thin_plate(x, k=40, penalty_order=2),
        lambda x: smoothcon.gaussian_process(
            x,
            k=40,
            kernel_name="matern2.5",
            linear_trend=True,
            range_=None,
            power=1.0,
        ),
    ],
    ids=["tp", "gp"],
)
def test_radial_geometry_matches_its_x64_construction(build) -> None:
    x = np.linspace(0.0, 1.0, 300)

    def construct() -> tuple[np.ndarray, np.ndarray, int]:
        values = jnp.asarray(x)
        smooth = build(x)
        design = np.asarray(smooth.basis(values), float)
        penalty = np.asarray(smooth.penalty, float)
        diagonal = smooth.diagonalize_penalty(values=values)
        penalized = np.asarray(diagonal.basis(values), float)[:, : diagonal.rank]
        return design, penalty, penalized @ penalized.T

    with jax.enable_x64(False):
        design32, penalty32, covariance32 = construct()
    with jax.enable_x64(True):
        design64, penalty64, covariance64 = construct()

    np.testing.assert_allclose(design32, design64, rtol=1e-4, atol=1e-5)
    assert _relative_error(penalty32, penalty64) < 1e-5
    assert _relative_error(covariance32, covariance64) < 1e-4


def test_constraint_that_removes_a_penalized_direction_lowers_the_rank() -> None:
    x = jnp.linspace(0.0, 1.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)
    # Orthogonal to the constant and linear null space, so it cuts the range.
    centered = jnp.arange(9.0) - 4.0
    constraint = (centered**2 - jnp.mean(centered**2))[None, :]

    constrained = smooth.constrain(constraint)

    assert constrained.penalty.shape == (8, 8)
    assert constrained.rank == 6


def test_directly_constructed_smooth_keeps_its_rank_and_transforms() -> None:
    penalty = jnp.asarray(_difference_penalty(6, 2))

    def basis(values):
        return jax.nn.one_hot(jnp.asarray(values, dtype=int), 6)

    smooth = smoothcon.Smooth(basis, penalty, 4)
    diagonal = smooth.constrain("sumzero_coef").diagonalize_penalty(
        values=jnp.arange(6)
    )

    assert smooth.rank == 4
    assert smooth.penalty is penalty
    assert diagonal.rank == 4
    np.testing.assert_array_equal(np.diag(diagonal.penalty), [1, 1, 1, 1, 0])
