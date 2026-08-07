# smoothcon

[![pre-commit](https://github.com/liesel-devs/smoothcon2/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/smoothcon2/actions/workflows/pre-commit.yml) [![pytest](https://github.com/liesel-devs/smoothcon2/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/smoothcon2/actions/workflows/pytest.yml) [![doctest](https://github.com/liesel-devs/smoothcon2/actions/workflows/doctest.yml/badge.svg)](https://github.com/liesel-devs/smoothcon2/actions/workflows/doctest.yml)  [![coverage](https://raw.githubusercontent.com/liesel-devs/smoothcon2/pytest-cov/tests/coverage.svg)](https://github.com/liesel-devs/smoothcon2/actions/workflows/pytest.yml)

`smoothcon` constructs JAX-native basis matrices and quadratic penalties for
smooth terms. It is a standalone numerical library: arrays and construction
parameters go in, and an immutable `Smooth` comes out.

```python
import jax.numpy as jnp
import smoothcon

x = jnp.linspace(0.0, 1.0, 100)
smooth = smoothcon.pspline(x, k=20, degree=3, penalty_order=2)

basis = smooth.basis(x)
penalty = smooth.penalty
```

`Smooth.basis` supports `jax.jit` and finite first-order autodiff almost
everywhere for continuous smooth families. Exact derivatives at nonsmooth knot
or boundary locations and higher-order derivatives are not guaranteed. MRF
bases are discrete and excluded from the autodiff contract.

## Smooth families

- P-splines, integrated-derivative B-splines, and cyclic P-splines
- cubic regression and cyclic cubic regression splines
- thin-plate regression splines
- fixed-range Gaussian-process smooths
- Markov random fields

## Transformations

Transformations return new `Smooth` objects and can be composed explicitly:

```python
nonlinear = (
    smooth.constrain("constant_and_linear", values=x)
    .scale_penalty(values=x)
    .diagonalize_penalty()
)
```

Available constraints are evaluated-term sum-to-zero, coefficient sum-to-zero,
constant-and-linear trend removal, and arbitrary matrices `A` representing
`A @ coefficients == 0`.

## Installation and development

`smoothcon` requires Python 3.13 or 3.14, JAX, and NumPy.

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run ty check
uv run mypy
```

Normal tests do not require R. See
[`tests/mgcv_reference/README.md`](tests/mgcv_reference/README.md) for manual
oracle regeneration.

## Provenance and license

The numerical constructors include Python/JAX adaptations of algorithms from
Simon N. Wood's GPL-licensed
[`mgcv`](https://cran.r-project.org/package=mgcv), pinned for development and
regression testing to mgcv 1.9-4 at commit
`1b6a4c8374612da27e36420b4459e93acb183f2d`.

`smoothcon` is licensed under `GPL-3.0-or-later`. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for source provenance,
copyright notices, and academic citations.
