import numpy as np
import pandas as pd
import polars as pl
import pytest

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from smoothcon.smoothcon import _convert_to_polars


@pytest.fixture
def simple_numeric_data():
    """Fixture for simple numeric data tests"""
    return {
        "data": {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([10, 20, 30]),
            "z": np.array([0.1, 0.2, 0.3]),
        },
        "expected_shape": (3, 3),
    }


def _assert_result(result, simple_numeric_data):
    # check it's a polars DataFrame
    assert isinstance(result, pl.DataFrame)

    # check the data matches expectations
    assert result.shape == simple_numeric_data["expected_shape"]
    assert result.columns == list(simple_numeric_data["data"].keys())
    assert result["x"].to_list() == simple_numeric_data["data"]["x"].tolist()
    assert result["y"].to_list() == simple_numeric_data["data"]["y"].tolist()
    assert result["z"].to_list() == simple_numeric_data["data"]["z"].tolist()


class TestConvertToPolars:
    def test_pandas_dataframe_input(self, simple_numeric_data):
        """Test conversion from pandas DataFrame"""
        pd_df = pd.DataFrame.from_dict(simple_numeric_data["data"])
        result = _convert_to_polars(pd_df)
        _assert_result(result, simple_numeric_data)

    def test_dict_with_numpy_arrays(self, simple_numeric_data):
        """Test conversion from dict with numpy arrays"""
        data_dict = simple_numeric_data["data"]
        result = _convert_to_polars(data_dict)
        _assert_result(result, simple_numeric_data)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_dict_with_jax_arrays(self, simple_numeric_data):
        """Test conversion from dict with JAX arrays"""

        # create dict with JAX arrays from expected data
        simple_numeric_data["data"]["x"] = jnp.array(simple_numeric_data["data"]["x"])
        simple_numeric_data["data"]["y"] = jnp.array(simple_numeric_data["data"]["y"])
        simple_numeric_data["data"]["z"] = jnp.array(simple_numeric_data["data"]["z"])

        result = _convert_to_polars(simple_numeric_data["data"])
        _assert_result(result, simple_numeric_data)

    @pytest.mark.skipif(not HAS_JAX, reason="JAX not available")
    def test_mixed_numpy_jax_dict(self, simple_numeric_data):
        """Test dict with both numpy and JAX arrays"""
        simple_numeric_data["data"]["x"] = np.array(simple_numeric_data["data"]["x"])
        simple_numeric_data["data"]["z"] = jnp.array(simple_numeric_data["data"]["z"])

        result = _convert_to_polars(simple_numeric_data["data"])
        _assert_result(result, simple_numeric_data)

    def test_polars_dataframe_passthrough(self):
        """Test that polars DataFrame is returned unchanged (same object reference)"""
        pl_df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = _convert_to_polars(pl_df)
        # check it is the same object
        assert result is pl_df

    def test_empty_dict(self):
        """Test conversion of empty dict"""
        result = _convert_to_polars({})
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (0, 0)
