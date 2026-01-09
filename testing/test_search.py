import numpy as np
import pandas as pd
import pytest
from src.search import search

def test_empty_dataframe_returns_empty_array():
    df = pd.DataFrame()
    res = search(df)
    assert isinstance(res, np.ndarray)
    assert res.shape == (0, 0)
    assert res.size == 0

def test_single_column_returns_1x1_zero_float64():
    df = pd.DataFrame({"x": [1, 2, 3]})
    res = search(df)
    assert res.shape == (1, 1)
    assert np.array_equal(res, np.zeros((1, 1)))
    assert res.dtype == np.float64

def test_multiple_columns_shape_and_all_zeros():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6],
    })
    n = len(df.columns)
    res = search(df)
    assert res.shape == (n, n)
    assert np.all(res == 0)

def test_non_numeric_columns_do_not_affect_shape_or_values():
    df = pd.DataFrame({
        "s": ["x", "y"],
        "t": ["a", "b"],
        "u": [True, False],
    })
    n = len(df.columns)
    res = search(df)
    assert res.shape == (n, n)
    assert np.array_equal(res, np.zeros((n, n)))

def test_duplicate_column_names_counted_in_shape():
    df = pd.DataFrame([[1, 2, 3]], columns=["dup", "dup", "dup"])
    n = len(df.columns)
    res = search(df)
    assert n == 3
    assert res.shape == (3, 3)
    assert np.all(res == 0)