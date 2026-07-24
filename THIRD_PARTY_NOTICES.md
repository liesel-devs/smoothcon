# Third-party notices

## mgcv

Portions of the smooth constructors are Python/JAX adaptations of algorithms
from `mgcv` 1.9-4 at commit
`1b6a4c8374612da27e36420b4459e93acb183f2d`.

Relevant sources include:

- `R/smooth.r`, copyright Simon N. Wood
- `src/tprs.c`, copyright (C) 2000–2012 Simon N. Wood
- `src/mgcv.c`, copyright (C) 2000–2012 Simon N. Wood

`mgcv` is distributed under GPL version 2 or later. The Python/JAX adaptations
were modified by Johannes Brachem in 2026 and are distributed with this package
under GPL version 3 or later.

Relevant references include:

- Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal
  Statistical Society: Series B*, 65(1), 95–114.
  https://doi.org/10.1111/1467-9868.00374
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R*
  (2nd ed.). Chapman and Hall/CRC.

## Linear constraint reparameterization

The linear constraint reparameterization follows:

Kneib, T., Klein, N., Lang, S., & Umlauf, N. (2019). Modular regression—A Lego
system for building structured additive distributional regression models with
tensor product interactions. *TEST*, 28(1), 1–39.
https://doi.org/10.1007/s11749-019-00631-z

## Liesel spline primitives

The basic knot, B-spline basis, and P-spline penalty interfaces were adapted
from `liesel.contrib.splines`.

MIT License

Copyright (c) 2022 Paul Wiemann, Hannes Riebl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
