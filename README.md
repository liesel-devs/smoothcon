# SmoothCon

[![pre-commit](https://github.com/liesel-devs/smoothcon/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml)
[![pytest-cov](https://raw.githubusercontent.com/liesel-devs/smoothcon/refs/heads/main/tests/coverage.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml)

This is a small wrapper that pulls basis and penalty matrices from the R packge [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html) and converts them to numpy arrays.

Although `smoothcon` is part of the [`liesel`](https://github.com/liesel-devs/liesel)
ecosystem, it has no dependence on Liesel and can be used independently.
`smoothcon` works well together with [`liesel_gam`](https://github.com/liesel-devs/liesel_gam)
for building generalized additive distributional regresion models, see also the
[notebooks](https://github.com/liesel-devs/smoothcon/tree/main/notebooks) for
examples.

## Disclaimer

This package is experimental and under active development. That means:

- The API cannot be considered stable. If you depend on this package, pin the version.
- Testing has not been extensive as of now. Please check and verify!
- There is currently no documentation beyond this readme.

In any case, this package comes with no warranty or guarantees.

## Installation

You can install `smoothcon` from pypi:

```bash
pip install smoothcon
```

You can install the development version from GitHub via pip:

```bash
pip install git+https://github.com/liesel-devs/smoothcon.git
```

Smoothcon requires the following R packages:

```r
install.packages("arrow") # for general usage of ryp
install.packages("svglite") # for plotting in jupyter notebooks
```

## Usage

We illustrate usage with random data:

```python
# import packages
import numpy as np
from smoothcon import SmoothCon


# generate some random data
rng = np.random.default_rng(seed=1)
n = 100
x = rng.uniform(-2.0, 2.0, size=n)
y = x + rng.normal(loc=0.0, scale=1.0, size=n)
mcycle = {"accel": y, "times": x}  # imitating the MASS:mcycle dataset
```

Now we initialize the smooth. What's special here is that the `spec` argument of the
`SmoothCon` class can simply be a string containing the R code that you would usually
use to specify a smooth in `mgcv`. Any smooth specification accepted by `mgcv::SmoothCon`
is permitted.

```python
# construct smooth
smooth = SmoothCon(
    spec="s(times, bs='ps', k=20, m=c(3,2))",   # mgcv smooth specification
    data=mcycle,                # dictionary, pandas dataframe, or polars dataframe
    knots=None,                 # knots; if None (default), mgcv will create the knots
    absorb_cons=True,           # If True, constraints (e.g. sum-to-zero) will be absorbed into the basis matrix
    diagonal_penalty=True,      # If True, the penalty will be diagonalized
    pass_to_r=None,             # dictionary of data that should be made available to the R environment
)
```

Access smooth information:

```python
# shortcuts to smooth information
smooth.basis        # if there is only one basis in the smooth
smooth.penalty      # if there is only one penalty in the smooth
smooth.knots

# full smooth information
smooth.all_bases()      # list of all bases in the smooth
smooth.all_penalties()  # list of all penalties in the smooth

# prediction
new_x = rng.uniform(-1.0, 2.0, size=5)
newdata = {"times": new_x}
smooth.predict(data=newdata)            # compute single basis at new covariate values
smooth.predict_all_bases(data=newdata)  # compute all bases at new covariate values
smooth(new_x)                           # alternative syntax for .predict
```

### SmoothFactory

If you want to initialize several smooths, you might not want to pass the data each time
to `SmoothCon`. Passing the data each time is not only cumbersome, but also inefficient,
because it will be converted to an R dataframe each time. So you probably want to
use the `SmoothFactory` class to initialize your `SmoothCon` objects in most cases:

```python
from smoothcon import SmoothFactory

sf = SmoothFactory(data=df, pass_to_r=None) # pass data to R only once
smooth_x = sf("s(x, bs='ps', k=20)")        # call to initialize a SmoothCon object
```

## Usage with `liesel_gam`: Example Notebooks

Advanced usage for building generalized additive distributional regression models with `liesel` and `liesel_gam` is illustrated in the following notebooks.

- [notebooks/test_gam_gibbs.ipynb](https://github.com/liesel-devs/liesel_gam/blob/main/notebooks/test_gam_gibbs.ipynb): A generalized addition location-scale model, using inverse gamma priors an Gibbs kernels for the inverse smoothing parameters.
- [notebooks/test_gam_manual.ipynb](https://github.com/liesel-devs/liesel_gam/blob/main/notebooks/test_gam_manual.ipynb): A generalized addition location-scale model, using a manually initialized inverse smoothing parameter with a Weibull prior.
