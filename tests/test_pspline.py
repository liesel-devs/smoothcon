from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import smoothcon

ASSETS = Path(__file__).parent / "mgcv_reference" / "assets"


def test_spline_primitives_are_public() -> None:
    x = jnp.linspace(0.0, 1.0, 13)
    knots = smoothcon.equidistant_knots(x, n_param=7, order=3)
    basis = smoothcon.bspline_basis(x, knots, order=3)
    penalty = smoothcon.pspline_penalty(7, diff=2)

    assert basis.shape == (13, 7)
    assert penalty.shape == (7, 7)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=2e-6)


def test_pspline_matches_mgcv() -> None:
    with np.load(ASSETS / "ps.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)

        assert isinstance(smooth, smoothcon.Smooth)
        assert smooth.rank == 7
        assert smooth.knots is not None
        assert smooth.knots.shape == (13,)
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.penalty, asset["penalty"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"][:, 0])),
            asset["new_basis"],
            rtol=1e-9,
            atol=1e-10,
        )


def test_bspline_matches_mgcv() -> None:
    with np.load(ASSETS / "bs.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.bspline(x, k=9, degree=3, penalty_order=2)

        assert smooth.rank == 7
        assert smooth.knots is not None
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.penalty, asset["penalty"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"][:, 0])),
            asset["new_basis"],
            rtol=1e-9,
            atol=1e-10,
        )


def test_cyclic_pspline_matches_mgcv() -> None:
    with np.load(ASSETS / "cp.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.cyclic_pspline(x, k=9, degree=3, penalty_order=2)

        assert smooth.rank == 8
        assert smooth.knots is not None
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.penalty, asset["penalty"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"][:, 0])),
            asset["new_basis"],
            rtol=1e-9,
            atol=1e-10,
        )


def test_cubic_regression_matches_mgcv() -> None:
    with np.load(ASSETS / "cr.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.cubic_regression(x, k=9)

        assert smooth.rank == 7
        assert smooth.knots is not None
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.penalty, asset["penalty"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"][:, 0])),
            asset["new_basis"],
            rtol=1e-9,
            atol=1e-10,
        )


def test_cubic_regression_shrinkage_matches_mgcv() -> None:
    with np.load(ASSETS / "cs.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.cubic_regression(x, k=9, shrinkage=True)

        assert smooth.rank == 9
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.eigvalsh(smooth.penalty),
            np.linalg.eigvalsh(asset["penalty"]),
            rtol=1e-9,
            atol=1e-10,
        )


def test_cyclic_cubic_matches_mgcv() -> None:
    with np.load(ASSETS / "cc.npz") as asset, jax.enable_x64(True):
        x = asset["x"][:, 0]
        smooth = smoothcon.cyclic_cubic(x, k=9)

        assert smooth.rank == 7
        assert smooth.knots is not None
        assert smooth.knots.shape == (9,)
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(x)), asset["basis"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.penalty, asset["penalty"], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(asset["new_x"][:, 0])),
            asset["new_basis"],
            rtol=1e-9,
            atol=1e-10,
        )
