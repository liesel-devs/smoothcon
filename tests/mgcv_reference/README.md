# mgcv reference assets

The `.npz` files in `assets/` contain raw, unconstrained, unscaled, and
non-diagonalized mgcv smooths. The P-spline asset additionally contains the
fully scaled, sum-to-zero constrained, and diagonalized design and penalty.
Normal test runs only read these files and do not require R or mgcv.

To regenerate them from the checked-out mgcv source:

```bash
python tests/mgcv_reference/generate.py \
  --mgcv-source /path/to/mgcv
```

The Python driver installs that source tree into an isolated temporary R
library and runs the adjacent `generate.R`. The manifest records the mgcv
version, Git commit, array shapes, ranks, nullities, and asset checksums.
