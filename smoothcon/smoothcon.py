from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from numpy.typing import ArrayLike
from rpy2.rinterface import NULL
from rpy2.robjects import conversion, default_converter, pandas2ri

mgcv = rpackages.importr("mgcv")


class SmoothCon:
    def __init__(
        self,
        spec: str | ro.vectors.ListVector,
        data: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike],
        knots: ArrayLike | ro.FloatVector | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
        pass_to_r: dict | None = None,
    ) -> None:
        self.pass_to_r = pass_to_r if pass_to_r is not None else {}
        self.spec = spec
        self.data_r = data
        self.knots_r = knots
        self.absorb_cons = absorb_cons
        self.diagonal_penalty = diagonal_penalty
        self.scale_penalty = scale_penalty

        self.smooth = mgcv.smoothCon(
            self.spec,
            data=self.data_r,
            knots=self._knots_r,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )

    @property
    def pass_to_r(self) -> dict:
        return self._pass_to_r

    @pass_to_r.setter
    def pass_to_r(self, value: dict | None):
        value = value if value is not None else {}
        for key, val in value.items():
            ro.globalenv[key] = val
        self._pass_to_r = value

    @property
    def spec(self) -> ro.vectors.ListVector:
        return self._spec

    @spec.setter
    def spec(self, value: str | ro.vectors.ListVector):
        if isinstance(value, str):
            spec = ro.r(value)
        else:
            spec = value

        self._spec = spec

    @property
    def data(self) -> pd.DataFrame:
        return pandas2ri.rpy2py(self.data_r)

    @property
    def data_r(self) -> ro.vectors.DataFrame:
        return self._data

    @data_r.setter
    def data_r(
        self, value: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike]
    ) -> None:
        self._data = self._convert_data(value)

    def _convert_data(
        self, value: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike]
    ) -> ro.vectors.DataFrame:
        with conversion.localconverter(default_converter + pandas2ri.converter):
            if isinstance(value, dict):
                data_r = pandas2ri.py2rpy(pd.DataFrame(value))
            elif isinstance(value, pd.DataFrame):
                data_r = pandas2ri.py2rpy(value)
            else:
                data_r = value

        return data_r

    @property
    def knots(self) -> ArrayLike:
        return np.asarray(self.knots_r)

    @property
    def knots_r(self) -> ro.FloatVector:
        if self._knots_r is NULL:
            return self.smooth[0].rx2("knots")
        return self._knots_r

    @knots_r.setter
    def knots_r(self, value: ArrayLike | ro.FloatVector | None) -> None:
        if value is None:
            knots = NULL
        elif isinstance(value, ro.FloatVector):
            knots = value
        else:
            knots = ro.FloatVector(value)

        self._knots_r = knots

    def all_terms(self) -> list[str]:
        terms_list = []
        for smooth in self.smooth:
            terms_list.append(smooth.rx2("term")[0])
        return terms_list

    def all_bases(self) -> list[np.ndarray]:
        bases_list = []
        for smooth in self.smooth:
            bases_list.append(np.asarray(smooth.rx2("X")))
        return bases_list

    def all_penalties(self) -> list[list[np.ndarray]]:
        pen_list = []
        for smooth in self.smooth:
            pen_list2 = []

            for penalty in smooth.rx2("S"):
                pen_list2.append(np.asarray(penalty))

            pen_list.append(pen_list2)
        return pen_list

    def single_basis(self, smooth_index: int = 0) -> np.ndarray:
        return self.all_bases()[smooth_index]

    def single_penalty(
        self, smooth_index: int = 0, penalty_index: int = 0
    ) -> np.ndarray:
        return self.all_penalties()[smooth_index][penalty_index]

    def predict_all_bases(
        self, data: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike]
    ) -> list[np.ndarray]:
        data_r = self._convert_data(data)
        bases_list = []

        for smooth in self.smooth:
            bases_list.append(np.asarray(mgcv.PredictMat(smooth, data=data_r)))
        return bases_list

    def predict_single_basis(
        self,
        data: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike],
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
        self, data: ro.vectors.DataFrame | pd.DataFrame | dict[str, ArrayLike]
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
        self, data: dict[str, ArrayLike] | pd.DataFrame, pass_to_r: dict | None = None
    ) -> None:
        with conversion.localconverter(default_converter + pandas2ri.converter):
            if isinstance(data, dict):
                data_r = pandas2ri.py2rpy(pd.DataFrame(data))
            elif isinstance(data, pd.DataFrame):
                data_r = pandas2ri.py2rpy(data)
            else:
                raise TypeError(f"Type {type(data)} not supported.")

        self.data_r = data_r
        self.pass_to_r = pass_to_r if pass_to_r is not None else {}

    @property
    def pass_to_r(self) -> dict:
        return self._pass_to_r

    @pass_to_r.setter
    def pass_to_r(self, value: dict | None):
        value = value if value is not None else {}
        for key, val in value.items():
            ro.globalenv[key] = val
        self._pass_to_r = value

    def __call__(
        self,
        spec: str | ro.vectors.ListVector,
        knots: ArrayLike | ro.FloatVector | None = None,
        absorb_cons: bool = True,
        diagonal_penalty: bool = True,
        scale_penalty: bool = True,
    ) -> SmoothCon:
        smooth = SmoothCon(
            spec=spec,
            knots=knots,
            data=self.data_r,
            absorb_cons=absorb_cons,
            diagonal_penalty=diagonal_penalty,
            scale_penalty=scale_penalty,
        )
        return smooth
