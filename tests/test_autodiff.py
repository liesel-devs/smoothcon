import jax
import jax.numpy as jnp
import numpy as np
import pytest

import smoothcon


@pytest.mark.parametrize(
    "constructor",
    [
        lambda x: smoothcon.pspline(x, k=9, degree=3, penalty_order=2),
        lambda x: smoothcon.bspline(x, k=9, degree=3, penalty_order=2),
        lambda x: smoothcon.cyclic_pspline(x, k=9, degree=3, penalty_order=2),
        lambda x: smoothcon.cubic_regression(x, k=9),
        lambda x: smoothcon.cubic_regression(x, k=9, shrinkage=True),
        lambda x: smoothcon.cyclic_cubic(x, k=9),
        lambda x: smoothcon.thin_plate(x, k=9, penalty_order=2),
        lambda x: smoothcon.gaussian_process(
            x,
            k=9,
            kernel_name="matern1.5",
            linear_trend=True,
            range_=None,
            power=1.0,
        ),
    ],
)
def test_continuous_basis_is_jittable_and_first_order_differentiable(
    constructor,
) -> None:
    x = jnp.linspace(-1.0, 2.0, 31)
    smooth = constructor(x)
    np.testing.assert_allclose(
        jax.jit(smooth.basis)(x), smooth.basis(x), rtol=1e-5, atol=5e-7
    )
    assert bool(jnp.all(jnp.isfinite(jax.jacfwd(smooth.basis)(x[:2]))))


def test_transformed_basis_remains_jittable_and_differentiable() -> None:
    x = jnp.linspace(-1.0, 2.0, 31)
    smooth = (
        smoothcon.pspline(x, k=9, degree=3, penalty_order=2)
        .constrain("constant_and_linear", values=x)
        .scale_penalty(values=x)
        .diagonalize_penalty()
    )

    np.testing.assert_allclose(
        jax.jit(smooth.basis)(x), smooth.basis(x), rtol=1e-5, atol=5e-7
    )
    assert bool(jnp.all(jnp.isfinite(jax.jacfwd(smooth.basis)(x[:2]))))
