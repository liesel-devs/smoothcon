from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smoothcon
from smoothcon._radial import _eigen_largest_magnitude, _unique_locations

ASSETS = Path(__file__).parent / "mgcv_reference" / "assets"


def test_large_location_sampling_is_stable_and_order_independent() -> None:
    locations = np.linspace(-4.0, 7.0, 2101)[:, None]
    forward = _unique_locations(locations)
    reverse = _unique_locations(locations[::-1])
    assert forward.shape == (2000, 1)
    np.testing.assert_array_equal(forward, reverse)


def test_large_eigendecomposition_uses_largest_magnitudes() -> None:
    diagonal = np.linspace(-20.0, 15.0, 270)
    matrix = jnp.diag(jnp.asarray(diagonal))
    values, vectors = _eigen_largest_magnitude(matrix, 8)
    expected = np.sort(np.abs(diagonal))[-8:][::-1]
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(values)))[::-1],
        expected,
        rtol=2e-4,
    )
    residual = matrix @ vectors - vectors * values
    np.testing.assert_allclose(residual, 0.0, atol=1e-2)


def test_two_dimensional_thin_plate_matches_mgcv() -> None:
    with np.load(ASSETS / "tp_2d.npz") as asset, jax.enable_x64(True):
        x = asset["x"]
        smooth = smoothcon.thin_plate(x, k=12, penalty_order=2)

        assert smooth.rank == 9
        assert smooth.knots is not None
        assert smooth.knots.shape[1] == 2
        basis = np.asarray(smooth.basis(jnp.asarray(x)))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        np.testing.assert_allclose(basis @ change, asset["basis"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            change.T @ np.asarray(smooth.penalty) @ change,
            asset["penalty"],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"])) @ change,
            asset["new_basis"],
            rtol=1e-6,
            atol=3e-7,
        )


def test_one_dimensional_thin_plate_matches_mgcv() -> None:
    with np.load(ASSETS / "tp_1d.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.thin_plate(x, k=9, penalty_order=2)

        assert smooth.rank == 7
        assert smooth.knots is not None
        basis = np.asarray(smooth.basis(jnp.asarray(x)))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        np.testing.assert_allclose(basis @ change, asset["basis"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            change.T @ np.asarray(smooth.penalty) @ change,
            asset["penalty"],
            rtol=1e-6,
            atol=1e-7,
        )


def test_thin_plate_shrinkage_matches_mgcv() -> None:
    with np.load(ASSETS / "ts_2d.npz") as asset, jax.enable_x64(True):
        x = asset["x"]
        smooth = smoothcon.thin_plate(x, k=12, penalty_order=2, shrinkage=True)

        assert smooth.rank == 12
        basis = np.asarray(smooth.basis(jnp.asarray(x)))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        mapped_penalty = change.T @ np.asarray(smooth.penalty) @ change
        np.testing.assert_allclose(
            mapped_penalty[:9, :9],
            asset["penalty"][:9, :9],
            rtol=1e-6,
            atol=1e-7,
        )
        actual = np.linalg.eigvalsh(np.asarray(smooth.penalty))
        expected = np.linalg.eigvalsh(asset["penalty"])
        np.testing.assert_allclose(
            actual[:3] / actual[3],
            expected[:3] / expected[3],
            rtol=1e-6,
            atol=1e-7,
        )


def test_thin_plate_can_remove_its_null_space() -> None:
    x = jnp.column_stack(
        (jnp.linspace(-1.0, 2.0, 40), jnp.linspace(-1.0, 2.0, 40) ** 2)
    )
    smooth = smoothcon.thin_plate(
        x,
        k=12,
        penalty_order=2,
        remove_null_space=True,
    )

    assert smooth.basis(x).shape == (40, 9)
    assert smooth.penalty.shape == (9, 9)
    assert smooth.rank == 9
    np.testing.assert_allclose(jnp.mean(smooth.basis(x), axis=0), 0.0, atol=1e-6)


def test_gaussian_process_matches_mgcv() -> None:
    with np.load(ASSETS / "gp_matern15.npz") as asset, jax.enable_x64(True):
        x = asset["x"]
        smooth = smoothcon.gaussian_process(
            x,
            k=12,
            kernel_name="matern1.5",
            linear_trend=True,
            range_=None,
            power=1.0,
        )

        assert smooth.rank == 9
        assert smooth.knots is not None
        basis = np.asarray(smooth.basis(jnp.asarray(x)))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        np.testing.assert_allclose(basis @ change, asset["basis"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            change.T @ np.asarray(smooth.penalty) @ change,
            asset["penalty"],
            rtol=1e-6,
            atol=5e-5,
        )
        jacobian = jax.jacfwd(smooth.basis)(jnp.asarray(x[:1]))
        assert bool(jnp.all(jnp.isfinite(jacobian)))


@pytest.mark.parametrize(
    ("case", "kernel_name", "linear_trend", "range_", "power"),
    [
        ("gp_spherical", "spherical", True, None, 1.0),
        ("gp_power", "power_exponential", True, None, 1.4),
        ("gp_matern25", "matern2.5", True, 2.3, 1.0),
        ("gp_matern35_stationary", "matern3.5", False, None, 1.0),
    ],
)
def test_gaussian_process_kernel_variants_match_mgcv(
    case: str,
    kernel_name: str,
    linear_trend: bool,
    range_: float | None,
    power: float,
) -> None:
    with np.load(ASSETS / f"{case}.npz") as asset, jax.enable_x64(True):
        smooth = smoothcon.gaussian_process(
            asset["x"],
            k=12,
            kernel_name=kernel_name,
            linear_trend=linear_trend,
            range_=range_,
            power=power,
        )
        basis = np.asarray(smooth.basis(jnp.asarray(asset["x"])))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        np.testing.assert_allclose(basis @ change, asset["basis"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"])) @ change,
            asset["new_basis"],
            rtol=1e-6,
            atol=3e-7,
        )
        np.testing.assert_allclose(
            change.T @ np.asarray(smooth.penalty) @ change,
            asset["penalty"],
            rtol=1e-6,
            atol=5e-5,
        )
