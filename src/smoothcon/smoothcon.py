from __future__ import annotations

import typing
import uuid

import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike
from ryp import options, r, to_py, to_r

# Configure ryp to use polars format (currently the default)
options(to_py_format="polars")

# Load mgcv package (suppress startup messages)
r("suppressPackageStartupMessages(library(mgcv))")


def _convert_to_polars(
    data: pd.DataFrame | pl.DataFrame | dict[str, ArrayLike],
) -> pl.DataFrame:
    """Convert input data to a polars DataFrame."""
    match data:
        case dict():
            # convert JAX arrays to numpy arrays for polars compatibility
            converted_data: dict[str, ArrayLike] = {}
            for key, value in data.items():
                # check if it's a JAX array
                if (
                    hasattr(value, "__module__")
                    and value.__module__ is not None
                    and "jax" in value.__module__
                ):
                    converted_data[key] = np.asarray(value)
                else:
                    converted_data[key] = value
            return pl.DataFrame(converted_data)
        case pd.DataFrame():
            return pl.from_pandas(data)
        case pl.DataFrame():
            return data
        case _:
            typing.assert_never(data)


class SmoothCon:
    def __init__(
        self,
        spec: str,
        data: pd.DataFrame | pl.DataFrame | dict[str, ArrayLike],
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        pass_to_r: dict | None = None,
    ) -> None:
        self.pass_to_r = pass_to_r if pass_to_r is not None else {}
        self.spec = spec
        self.data = _convert_to_polars(data)
        self.knots = knots
        self.absorb_cons = absorb_cons
        self.diagonal_penalty = diagonal_penalty
        self.scale_penalty = scale_penalty

        # generate unique variable names for R environment
        self._data_var = f"smoothcon_data_{uuid.uuid4().hex[:8]}"
        self._knots_var = f"smoothcon_knots_{uuid.uuid4().hex[:8]}"
        self._smooth_var = f"smoothcon_smooth_{uuid.uuid4().hex[:8]}"

        # convert data to R
        self._convert_data_to_r()
        self._convert_knots_to_r()

        # create smooth
        knots_arg = f"knots={self._knots_var}" if knots is not None else "knots=NULL"
        r_cmd = f"""
        {self._smooth_var} <- smoothCon(
            {self.spec},
            data={self._data_var},
            {knots_arg},
            absorb.cons={str(absorb_cons).upper()},
            diagonal.penalty={str(diagonal_penalty).upper()},
            scale.penalty={str(scale_penalty).upper()}
        )
        """
        r(r_cmd)

    @property
    def pass_to_r(self) -> dict:
        return self._pass_to_r

    @pass_to_r.setter
    def pass_to_r(self, value: dict | None):
        value = value if value is not None else {}
        for key, val in value.items():
            to_r(val, key)
        self._pass_to_r = value

    def _convert_data_to_r(self) -> None:
        """convert data to R dataframe"""
        # data is already converted to polars in __init__
        to_r(self.data, self._data_var)

    def _convert_knots_to_r(self) -> None:
        """convert knots to R"""
        if self.knots is not None:
            to_r(self.knots, self._knots_var)

    def all_terms(self) -> list[str]:
        """get all smooth terms"""
        r(f"terms_list <- sapply({self._smooth_var}, function(x) x$term)")
        terms = [to_py("terms_list")]
        return terms

    def all_bases(self) -> list[np.ndarray]:
        """get all basis matrices"""
        r(f"bases_list <- lapply({self._smooth_var}, function(x) x$X)")
        bases_r: list[pl.DataFrame] = to_py("bases_list")
        bases_np = [base_r.to_numpy() for base_r in bases_r]
        print([type(base_r) for base_r in bases_r])
        return bases_np

    def all_penalties(self) -> list[list[np.ndarray]]:
        """get all penalty matrices"""
        r(f"penalties_list <- lapply({self._smooth_var}, function(x) x$S)")
        penalties_r: list[list[pl.DataFrame]] = to_py("penalties_list")

        penalties = [
            [penalty_r.to_numpy() for penalty_r in smooth_penalties]
            for smooth_penalties in penalties_r
        ]
        return penalties

    def single_basis(self, smooth_index: int = 0) -> np.ndarray:
        return self.all_bases()[smooth_index]

    def single_penalty(
        self, smooth_index: int = 0, penalty_index: int = 0
    ) -> np.ndarray:
        return self.all_penalties()[smooth_index][penalty_index]

    def predict_all_bases(
        self, data: pd.DataFrame | pl.DataFrame | dict[str, ArrayLike]
    ) -> list[np.ndarray]:
        """predict basis matrices for new data"""
        # convert new data to R
        pred_data_var = f"pred_data_{uuid.uuid4().hex[:8]}"
        df = _convert_to_polars(data)
        to_r(df, pred_data_var)

        # predict basis matrices
        pred_var = f"pred_bases_{uuid.uuid4().hex[:8]}"
        r(f"""
        {pred_var} <- lapply({self._smooth_var}, function(smooth) {{
            PredictMat(smooth, data={pred_data_var})
        }})
        """)

        bases_r = to_py(pred_var)
        bases = [base_r.to_numpy() for base_r in bases_r]
        return bases

    def predict_single_basis(
        self,
        data: pd.DataFrame | pl.DataFrame | dict[str, ArrayLike],
        smooth_index: int = 0,
    ) -> np.ndarray:
        return self.predict_all_bases(data)[smooth_index]

    @property
    def term(self) -> str:
        terms = self.all_terms()
        if len(terms) > 1:
            raise ValueError(
                "Smooth has more than one basis. Consider using .all_terms()."
            )
        return terms[0]

    @property
    def basis(self) -> np.ndarray:
        bases = self.all_bases()
        if len(bases) > 1:
            raise ValueError(
                "Smooth has more than one basis. Consider using "
                ".all_bases() or .single_basis()."
            )
        return bases[0]

    @property
    def penalty(self) -> np.ndarray:
        penalties = self.all_penalties()
        len_layer1 = len(penalties)
        len_layer2 = len(penalties[0])
        if (len_layer1 > 1) or (len_layer2 > 1):
            raise ValueError(
                "Smooth has more than one penalty. Consider using "
                ".all_penalties() or .single_penalty()."
            )
        return penalties[0][0]

    def predict(
        self, data: pd.DataFrame | pl.DataFrame | dict[str, ArrayLike]
    ) -> np.ndarray:
        bases = self.predict_all_bases(data)
        if len(bases) > 1:
            raise ValueError(
                "Smooth has more than one basis. Consider using"
                ".predict_all_bases() or .predict_single_basis()."
            )
        return np.concatenate(self.predict_all_bases(data), axis=1)

    def __call__(self, x: ArrayLike) -> np.ndarray:
        data = {self.term: x}
        return self.predict(data)


class SmoothFactory:
    def __init__(
        self,
        data: pl.DataFrame | dict[str, ArrayLike] | pd.DataFrame,
        pass_to_r: dict | None = None,
    ) -> None:
        self.data = data
        self.pass_to_r = pass_to_r if pass_to_r is not None else {}

    @property
    def pass_to_r(self) -> dict:
        return self._pass_to_r

    @pass_to_r.setter
    def pass_to_r(self, value: dict | None):
        value = value if value is not None else {}
        for key, val in value.items():
            to_r(val, key)
        self._pass_to_r = value

    def __call__(
        self,
        spec: str,
        knots: ArrayLike | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
    ) -> SmoothCon:
        smooth = SmoothCon(
            spec=spec,
            knots=knots,
            data=self.data,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )
        return smooth
