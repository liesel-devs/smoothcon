# SmoothCon

[![pre-commit](https://github.com/liesel-devs/smoothcon/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pre-commit.yml)
[![notebooks](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest-notebooks.yml/badge.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest-notebooks.yml)
[![pytest](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml/badge.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml)
[![pytest-cov](tests/coverage.svg)](https://github.com/liesel-devs/smoothcon/actions/workflows/pytest.yml)

This is a small wrapper that pulls basis and penalty matrices from mgcv and converts them to numpy arrays.

## License

Due to its dependence on `rpy2`, [which is licensed under GPL-2.0](https://github.com/rpy2/rpy2/blob/master/LICENSE) `smoothcon` is also licensed under GPL-2.0. As a result, if you depend on `smoothcon`, your project also needs to be licensed under GPL-2.0.