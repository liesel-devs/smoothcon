# Deviations from mgcv

`smoothcon` regression-tests raw basis and penalty constructions against
committed assets generated with mgcv 1.9-4 at commit
`1b6a4c8374612da27e36420b4459e93acb183f2d`. Some internal coordinates differ
deliberately. The complete scale, sum-to-zero constraint, diagonalization, and
prediction sequence is additionally oracle-tested for P-splines only.

## Transformations

Constraints, penalty scaling, and penalty diagonalization are explicit,
immutable `Smooth` transformations. Constraint coordinates can differ from
mgcv's QR-based coordinates while representing the same constrained function
space.

Penalty scaling uses

\[
K \leftarrow K\,\frac{\lVert B\rVert_\infty^2}{\lVert K\rVert_1}.
\]

Because scaling and constraints do not commute, callers choose their order.
For an exclusively nonlinear P-spline, first remove constant and linear
trends, then scale the resulting component, then diagonalize if desired.

Diagonalization scales penalized eigendirections to unit penalty and retains
the full null space. It requires the setup values and follows
`mgcv::nat.param(type=2, unit.fnorm=FALSE)` by scaling null-space design columns
to the average squared norm of the penalized columns. Eigenvector signs and
tied eigenspaces can differ from mgcv without changing the represented
quadratic form or this normalization. Post-transformation agreement with mgcv
is currently asserted only for the P-spline sequence described above.

## Smooth families

- Univariate raw matrices match the mgcv oracle assets up to floating-point
  roundoff. Integrated B-spline penalties use per-interval Gauss–Legendre
  quadrature.
- Thin-plate and Gaussian-process bases use equivalent eigenspaces rather than
  mgcv's exact eigenvector coordinates.
- Large radial eigenproblems use deterministic LOBPCG when appropriate, with
  dense decomposition as the fallback.
- More than 2,000 unique radial locations are reduced by a stable content hash;
  mgcv can select different locations.
- Multidimensional custom radial knots are unsupported.
- MRF polygons are adjacent when they share an exact vertex. Low-rank MRF
  bases can use different eigenvector coordinates while preserving predictions
  and penalty quadratic forms.

## Precision and oracle tests

Normal tests do not require R or mgcv. Oracle comparisons enable JAX 64-bit
mode locally and compare transformed basis Gram matrices so arbitrary signs
and rotations do not become part of the compatibility promise. Runtime
precision follows the user's JAX configuration.
