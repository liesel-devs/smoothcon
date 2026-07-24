import jax.numpy as jnp
import numpy as np

import smoothcon


def test_sumzero_term_constrains_the_evaluated_term_immutably() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)

    constrained = smooth.constrain("sumzero_term", values=x)

    assert smooth.basis(x).shape == (40, 9)
    assert constrained.basis(x).shape == (40, 8)
    assert constrained.penalty.shape == (8, 8)
    assert constrained.rank == 7
    np.testing.assert_allclose(jnp.sum(constrained.basis(x), axis=0), 0.0, atol=2e-5)
    np.testing.assert_array_equal(constrained.knots, smooth.knots)


def test_sumzero_coef_constrains_the_coefficients() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)

    constrained = smooth.constrain("sumzero_coef")

    transform, *_ = np.linalg.lstsq(
        np.asarray(smooth.basis(x)),
        np.asarray(constrained.basis(x)),
        rcond=None,
    )
    np.testing.assert_allclose(np.sum(transform, axis=0), 0.0, atol=2e-5)


def test_constant_and_linear_constraint_removes_both_trends() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)

    constrained = smooth.constrain("constant_and_linear", values=x)

    linear_basis = jnp.column_stack((jnp.ones_like(x), x))
    np.testing.assert_allclose(linear_basis.T @ constrained.basis(x), 0.0, atol=2e-4)
    assert constrained.basis(x).shape == (40, 7)


def test_custom_constraint_removes_requested_coefficient_directions() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2)
    constraint = jnp.stack((jnp.ones(9), jnp.arange(9)))

    constrained = smooth.constrain(constraint)

    transform, *_ = np.linalg.lstsq(
        np.asarray(smooth.basis(x)),
        np.asarray(constrained.basis(x)),
        rcond=None,
    )
    np.testing.assert_allclose(constraint @ transform, 0.0, atol=2e-5)


def test_penalty_scaling_normalizes_the_current_smooth() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = smoothcon.pspline(x, k=9, degree=3, penalty_order=2).constrain(
        "constant_and_linear", values=x
    )

    scaled = smooth.scale_penalty(values=x)

    ratio = jnp.linalg.norm(scaled.penalty, ord=1) / (
        jnp.linalg.norm(scaled.basis(x), ord=jnp.inf) ** 2
    )
    np.testing.assert_allclose(ratio, 1.0, rtol=2e-6)
    np.testing.assert_array_equal(scaled.basis(x), smooth.basis(x))


def test_penalty_diagonalization_preserves_rank_and_null_space() -> None:
    x = jnp.linspace(-1.0, 2.0, 40)
    smooth = (
        smoothcon.pspline(x, k=9, degree=3, penalty_order=2)
        .constrain("sumzero_term", values=x)
        .scale_penalty(values=x)
    )

    diagonal = smooth.diagonalize_penalty()

    np.testing.assert_allclose(
        diagonal.penalty,
        jnp.diag(jnp.r_[jnp.ones(7), jnp.zeros(1)]),
        atol=1e-6,
    )
    assert diagonal.rank == smooth.rank == 7
    assert diagonal.basis(x).shape == smooth.basis(x).shape
