# smoothcon

`smoothcon` constructs JAX-native basis matrices and quadratic penalties for
smooth terms. Arrays and construction parameters go in, and an immutable
`Smooth` object comes out.

## Relationship to mgcv

Much of `smoothcon`'s numerical implementation is a Python/JAX adaptation of
algorithms from Simon N. Wood's GPL-licensed R package
[`mgcv`](https://cran.r-project.org/package=mgcv). Development and regression
testing are pinned to mgcv 1.9-4.

`smoothcon` is licensed under `GPL-3.0-or-later`. See the
[third-party notices](https://github.com/liesel-devs/smoothcon2/blob/main/THIRD_PARTY_NOTICES.md)
for detailed source provenance, copyright notices, and academic citations.

## Installation

```bash
pip install smoothcon
```

With uv, use `uv add smoothcon` instead.

## Quick start

```python
import jax.numpy as jnp
import smoothcon

x = jnp.linspace(0.0, 1.0, 100)
smooth = smoothcon.pspline(x, k=20, degree=3, penalty_order=2)

basis = smooth.basis(x)
penalty = smooth.penalty
```

`smooth.basis(values)` evaluates the design matrix, `smooth.penalty` contains
the coefficient penalty, and `smooth.rank` records its numerical rank. The
basis can be evaluated inside JAX transformations such as `jax.jit`.

## Smooth families

- P-splines and integrated-derivative B-splines
- cyclic P-splines and cyclic cubic splines
- natural cubic regression splines
- thin-plate regression splines
- fixed-range Gaussian-process smooths
- Markov random fields

## Transformations

Transformations return new `Smooth` objects, so their order remains explicit:

```python
nonlinear = (
    smooth.constrain("constant_and_linear", values=x)
    .scale_penalty(values=x)
    .diagonalize_penalty(values=x)
)
```

See the [API reference](api.md) for all constructors and the
[deviations from mgcv](mgcv_deviations.md) for numerical compatibility notes.
