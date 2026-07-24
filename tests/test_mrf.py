from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import smoothcon

ASSETS = Path(__file__).parent / "mgcv_reference" / "assets"


def test_full_rank_mrf_matches_mgcv() -> None:
    with np.load(ASSETS / "mrf_full.npz") as asset, jax.enable_x64(True):
        codes = asset["x"][:, 0].astype(int) - 1
        smooth = smoothcon.mrf(codes, penalty=asset["penalty"], k=-1)

        assert smooth.rank == 5
        assert smooth.knots is None
        np.testing.assert_allclose(
            smooth.basis(jnp.asarray(codes)), asset["basis"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(smooth.penalty, asset["penalty"], rtol=0.0, atol=0.0)


def test_low_rank_mrf_matches_mgcv() -> None:
    with (
        np.load(ASSETS / "mrf_full.npz") as full,
        np.load(ASSETS / "mrf_low_rank.npz") as asset,
        jax.enable_x64(True),
    ):
        codes = asset["x"][:, 0].astype(int) - 1
        smooth = smoothcon.mrf(codes, penalty=full["penalty"], k=4)

        assert smooth.rank == 3
        basis = np.asarray(smooth.basis(jnp.asarray(codes)))
        change, *_ = np.linalg.lstsq(basis, asset["basis"], rcond=None)
        np.testing.assert_allclose(basis @ change, asset["basis"], rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(
            change.T @ np.asarray(smooth.penalty) @ change,
            asset["penalty"],
            rtol=1e-6,
            atol=1e-7,
        )


def test_polygon_neighbors_construct_a_laplacian() -> None:
    polygons = {
        "a": np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        "b": np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
        "c": np.array([[2, 0], [3, 0], [3, 1], [2, 1]]),
    }
    neighbors = smoothcon.polygon_neighbors(polygons)
    normalized = smoothcon.normalize_neighbors(neighbors, ["a", "b", "c"])
    np.testing.assert_array_equal(
        smoothcon.laplacian(normalized, ["a", "b", "c"]),
        np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]]),
    )
