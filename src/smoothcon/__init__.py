# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Johannes Brachem

"""Build spline and spatial smooths for JAX models.

Each constructor returns a ``Smooth`` containing a callable basis and a
penalty matrix. Evaluate the basis at observed or new values, then use the
penalty to keep the fitted effect from becoming unnecessarily wiggly.

Examples
--------
```pycon
>>> import jax.numpy as jnp
>>> import smoothcon
>>> x = jnp.linspace(0.0, 1.0, 5)
>>> smooth = smoothcon.pspline(x, k=4, degree=3, penalty_order=2)
>>> smooth.basis(x).shape
(5, 4)
>>> smooth.penalty.shape
(4, 4)
>>> smooth.rank
2

```
"""

from ._mrf import laplacian as laplacian
from ._mrf import mrf as mrf
from ._mrf import normalize_neighbors as normalize_neighbors
from ._mrf import polygon_neighbors as polygon_neighbors
from ._radial import gaussian_process as gaussian_process
from ._radial import thin_plate as thin_plate
from ._smooth import Smooth as Smooth
from ._splines import basis_matrix as basis_matrix
from ._splines import equidistant_knots as equidistant_knots
from ._splines import pspline_penalty as pspline_penalty
from ._univariate import bspline as bspline
from ._univariate import cubic_regression as cubic_regression
from ._univariate import cyclic_cubic as cyclic_cubic
from ._univariate import cyclic_pspline as cyclic_pspline
from ._univariate import pspline as pspline

__version__ = "0.1.1"
