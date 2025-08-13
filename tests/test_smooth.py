import numpy as np
import pandas as pd

from smoothcon import SmoothCon, SmoothFactory

rng = np.random.default_rng(seed=1)

n = 30
k = 1
x = rng.uniform(-2.0, 2.0, size=(n, k))
z = rng.binomial(n=1, p=0.5, size=n)
y = x.sum(axis=1) + rng.normal(loc=0.0, scale=1.0, size=n)
xdict = {"x0": x[:, 0]}
data = pd.DataFrame({"y": y, "z": z} | xdict)


class TestSmoothConBasics:
    def test_init(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)
        assert smooth is not None

    def test_basis(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)
        assert smooth.basis.shape == (n, 7)  # 7, because of sum-to-zero constraint
        assert not np.any(np.isnan(smooth.basis))

    def test_penalty(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)
        assert smooth.penalty.shape == (7, 7)  # 7, because of sum-to-zero constraint
        assert not np.any(np.isnan(smooth.penalty))

    def test_absorb_constraint_false(self) -> None:
        smooth = SmoothCon(
            "s(x0, bs='ps', k=8)",
            data=data,
            absorb_cons=False,
        )
        assert smooth.basis.shape == (n, 8)
        assert not np.any(np.isnan(smooth.basis))
        assert smooth.penalty.shape == (8, 8)
        assert not np.any(np.isnan(smooth.penalty))

    def test_diagonalize_penalty(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data, diagonal_penalty=True)

        pen = smooth.penalty
        assert np.allclose(pen[:-1, :-1], np.eye(6))
        assert np.allclose(pen[-1, :], np.zeros(7))

    def test_scale_penalty(self) -> None:
        smooth1 = SmoothCon(
            "s(x0, bs='ps', k=8)", data=data, scale_penalty=True, diagonal_penalty=False
        )
        smooth2 = SmoothCon(
            "s(x0, bs='ps', k=8)",
            data=data,
            scale_penalty=False,
            diagonal_penalty=False,
        )

        assert not np.allclose(smooth1.penalty, smooth2.penalty)

    def test_predict(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)

        xnew = rng.uniform(size=(3,))

        # predict with dictionary
        basis = smooth.predict({"x0": xnew})
        assert basis.shape == (3, 7)
        assert not np.any(np.isnan(basis))

        # predict with data frame
        df = pd.DataFrame({"x0": xnew})
        basis = smooth.predict(df)
        assert basis.shape == (3, 7)
        assert not np.any(np.isnan(basis))

    def test_call(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)
        xnew = rng.uniform(size=(3,))
        basis = smooth(xnew)
        assert basis.shape == (3, 7)
        assert not np.any(np.isnan(basis))

    def test_term(self) -> None:
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data)
        assert smooth.term == "x0"

    def test_custom_knots(self) -> None:
        knots = np.linspace(-4.4, 4.4, 12)
        smooth = SmoothCon("s(x0, bs='ps', k=8)", data=data, knots=knots)

        assert np.allclose(knots, smooth.knots)


class TestSmoothFactory:
    def test_init(self) -> None:
        sf = SmoothFactory(data)
        smooth = sf("s(x0, bs='ps', k=8)")

        assert smooth.basis.shape == (n, 7)  # 7, because of sum-to-zero constraint
        assert not np.any(np.isnan(smooth.basis))

    def test_custom_knots(self) -> None:
        knots = np.linspace(-4.4, 4.4, 12)
        sf = SmoothFactory(data)
        sf("s(x0, bs='ps', k=8)", knots=knots)
