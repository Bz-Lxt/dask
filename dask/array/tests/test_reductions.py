from __future__ import annotations

import math
import os
import warnings
from contextlib import nullcontext as does_not_warn
from fractions import Fraction
from itertools import permutations, zip_longest

import pytest

np = pytest.importorskip("numpy")

import itertools

import dask.array as da
from dask import config
from dask.array.numpy_compat import NUMPY_GE_200, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps


@pytest.mark.parametrize("dtype", ["f4", "i4"])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("nan", [True, False])
def test_numel(dtype, keepdims, nan):
    x = np.ones((2, 3, 4))
    if nan:
        y = np.random.default_rng().uniform(-1, 1, size=(2, 3, 4))
        x[y < 0] = np.nan
        numel = da.reductions.nannumel

        def _sum(arr, **kwargs):
            n = np.sum(np.ma.masked_where(np.isnan(arr), arr), **kwargs)
            return n.filled(0) if isinstance(n, np.ma.MaskedArray) else n

    else:
        numel = da.reductions.numel
        _sum = np.sum

    assert_eq(
        numel(x, axis=(), keepdims=keepdims, dtype=dtype),
        _sum(x, axis=(), keepdims=keepdims, dtype=dtype),
    )
    assert_eq(
        numel(x, axis=0, keepdims=keepdims, dtype=dtype),
        _sum(x, axis=0, keepdims=keepdims, dtype=dtype),
    )

    for length in range(x.ndim):
        for sub in itertools.combinations([d for d in range(x.ndim)], length):
            assert_eq(
                numel(x, axis=sub, keepdims=keepdims, dtype=dtype),
                _sum(x, axis=sub, keepdims=keepdims, dtype=dtype),
            )

    for length in range(x.ndim):
        for sub in itertools.combinations([d for d in range(x.ndim)], length):
            ssub = np.random.default_rng().shuffle(list(sub))
            assert_eq(
                numel(x, axis=ssub, keepdims=keepdims, dtype=dtype),
                _sum(x, axis=ssub, keepdims=keepdims, dtype=dtype),
            )


def reduction_0d_test(da_func, darr, np_func, narr):
    expected = np_func(narr)
    actual = da_func(darr)

    assert_eq(actual, expected)
    assert_eq(da_func(narr), expected)  # Ensure Dask reductions work with NumPy arrays
    assert actual.size == 1


def test_reductions_0D():
    x = np.int_(3)  # np.int_ has a dtype attribute, np.int does not.
    a = da.from_array(x, chunks=(1,))

    reduction_0d_test(da.sum, a, np.sum, x)
    reduction_0d_test(da.prod, a, np.prod, x)
    reduction_0d_test(da.mean, a, np.mean, x)
    reduction_0d_test(da.var, a, np.var, x)
    reduction_0d_test(da.std, a, np.std, x)
    reduction_0d_test(da.min, a, np.min, x)
    reduction_0d_test(da.max, a, np.max, x)
    reduction_0d_test(da.any, a, np.any, x)
    reduction_0d_test(da.all, a, np.all, x)

    reduction_0d_test(da.nansum, a, np.nansum, x)
    reduction_0d_test(da.nanprod, a, np.nanprod, x)
    reduction_0d_test(da.nanmean, a, np.mean, x)
    reduction_0d_test(da.nanvar, a, np.var, x)
    reduction_0d_test(da.nanstd, a, np.std, x)
    reduction_0d_test(da.nanmin, a, np.nanmin, x)
    reduction_0d_test(da.nanmax, a, np.nanmax, x)


def reduction_1d_test(da_func, darr, np_func, narr, use_dtype=True, split_every=True):
    assert_eq(da_func(darr), np_func(narr))
    assert_eq(
        da_func(narr), np_func(narr)
    )  # Ensure Dask reductions work with NumPy arrays
    assert_eq(da_func(darr, keepdims=True), np_func(narr, keepdims=True))
    assert_eq(da_func(darr, axis=()), np_func(narr, axis=()))
    assert same_keys(da_func(darr), da_func(darr))
    assert same_keys(da_func(darr, keepdims=True), da_func(darr, keepdims=True))
    if use_dtype:
        with pytest.warns(ComplexWarning) if np.iscomplexobj(narr) else does_not_warn():
            assert_eq(da_func(darr, dtype="f8"), np_func(narr, dtype="f8"))
            assert_eq(da_func(darr, dtype="i8"), np_func(narr, dtype="i8"))
            assert same_keys(da_func(darr, dtype="i8"), da_func(darr, dtype="i8"))
    if split_every:
        a1 = da_func(darr, split_every=2)
        a2 = da_func(darr, split_every={0: 2})
        assert same_keys(a1, a2)
        assert_eq(a1, np_func(narr))
        assert_eq(a2, np_func(narr))
        assert_eq(
            da_func(darr, keepdims=True, split_every=2), np_func(narr, keepdims=True)
        )


@pytest.mark.parametrize("dtype", ["f4", "i4", "c8"])
def test_reductions_1D(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        x = (np.arange(5) + 1j * np.arange(5)).astype(dtype)
        a = da.from_array(x, chunks=(2,))

    reduction_1d_test(da.sum, a, np.sum, x)
    reduction_1d_test(da.prod, a, np.prod, x)
    reduction_1d_test(da.mean, a, np.mean, x)
    reduction_1d_test(da.var, a, np.var, x)
    reduction_1d_test(da.std, a, np.std, x)
    reduction_1d_test(da.min, a, np.min, x, False)
    reduction_1d_test(da.max, a, np.max, x, False)
    reduction_1d_test(da.any, a, np.any, x, False)
    reduction_1d_test(da.all, a, np.all, x, False)

    reduction_1d_test(da.nansum, a, np.nansum, x)
    reduction_1d_test(da.nanprod, a, np.nanprod, x)
    reduction_1d_test(da.nanmean, a, np.mean, x)
    reduction_1d_test(da.nanvar, a, np.var, x)
    reduction_1d_test(da.nanstd, a, np.std, x)
    reduction_1d_test(da.nanmin, a, np.nanmin, x, False)
    reduction_1d_test(da.nanmax, a, np.nanmax, x, False)


def test_reductions_1D_datetime():
    x = np.arange(5).astype("datetime64[ns]")
    a = da.from_array(x, chunks=(2,))
    reduction_1d_test(da.min, a, np.min, x, False)
    reduction_1d_test(da.max, a, np.max, x, False)
    reduction_1d_test(da.any, a, np.any, x, False)
    reduction_1d_test(da.all, a, np.all, x, False)
    reduction_1d_test(da.nanmin, a, np.nanmin, x, False)
    reduction_1d_test(da.nanmax, a, np.nanmax, x, False)


@pytest.mark.parametrize("dtype", ["f4", "c8"])
@pytest.mark.parametrize(
    "x", [np.array([np.inf, np.nan, -np.inf, 2]), np.array([np.nan, np.nan, 3, 2])]
)
def test_reductions_1D_nans(x, dtype):
    x = x.astype(dtype)
    a = da.from_array(x, chunks=(1,))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduction_1d_test(da.nansum, a, np.nansum, x)
        reduction_1d_test(da.nanprod, a, np.nanprod, x)
        reduction_1d_test(da.nanmean, a, np.nanmean, x, False)
        reduction_1d_test(da.nanvar, a, np.nanvar, x, False)
        reduction_1d_test(da.nanstd, a, np.nanstd, x, False)
        reduction_1d_test(da.nanmin, a, np.nanmin, x, False)
        reduction_1d_test(da.nanmax, a, np.nanmax, x, False)


def reduction_2d_test(da_func, darr, np_func, narr, use_dtype=True, split_every=True):
    assert_eq(da_func(darr), np_func(narr))
    assert_eq(da_func(darr, keepdims=True), np_func(narr, keepdims=True))
    assert_eq(da_func(darr, axis=()), np_func(narr, axis=()))
    assert_eq(da_func(darr, axis=0), np_func(narr, axis=0))
    assert_eq(da_func(darr, axis=1), np_func(narr, axis=1))
    assert_eq(da_func(darr, axis=-1), np_func(narr, axis=-1))
    assert_eq(da_func(darr, axis=-2), np_func(narr, axis=-2))
    assert_eq(
        da_func(darr, axis=1, keepdims=True), np_func(narr, axis=1, keepdims=True)
    )
    assert_eq(
        da_func(darr, axis=(), keepdims=True), np_func(narr, axis=(), keepdims=True)
    )
    assert_eq(da_func(darr, axis=(1, 0)), np_func(narr, axis=(1, 0)))

    assert same_keys(da_func(darr, axis=()), da_func(darr, axis=()))
    assert same_keys(da_func(darr, axis=1), da_func(darr, axis=1))
    assert same_keys(da_func(darr, axis=(1, 0)), da_func(darr, axis=(1, 0)))

    if use_dtype:
        with pytest.warns(ComplexWarning) if np.iscomplexobj(narr) else does_not_warn():
            assert_eq(da_func(darr, dtype="f8"), np_func(narr, dtype="f8"))
            assert_eq(da_func(darr, dtype="i8"), np_func(narr, dtype="i8"))

    if split_every:
        a1 = da_func(darr, split_every=4)
        a2 = da_func(darr, split_every={0: 2, 1: 2})
        assert same_keys(a1, a2)
        assert_eq(a1, np_func(narr))
        assert_eq(a2, np_func(narr))
        assert_eq(
            da_func(darr, keepdims=True, split_every=4),
            np_func(narr, keepdims=True),
        )
        assert_eq(da_func(darr, axis=(), split_every=2), np_func(narr, axis=()))
        assert_eq(da_func(darr, axis=0, split_every=2), np_func(narr, axis=0))
        assert_eq(
            da_func(darr, axis=(), keepdims=True, split_every=2),
            np_func(narr, axis=(), keepdims=True),
        )
        assert_eq(
            da_func(darr, axis=0, keepdims=True, split_every=2),
            np_func(narr, axis=0, keepdims=True),
        )
        assert_eq(da_func(darr, axis=1, split_every=2), np_func(narr, axis=1))
        assert_eq(
            da_func(darr, axis=1, keepdims=True, split_every=2),
            np_func(narr, axis=1, keepdims=True),
        )


def test_reduction_errors():
    x = da.ones((5, 5), chunks=(3, 3))
    with pytest.raises(ValueError):
        x.sum(axis=2)
    with pytest.raises(ValueError):
        x.sum(axis=-3)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["f4", "i4", "c8"])
def test_reductions_2D(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        x = (np.arange(1, 122) + 1j * np.arange(1, 122)).reshape((11, 11)).astype(dtype)
    a = da.from_array(x, chunks=(4, 4))

    b = a.sum(keepdims=True)
    assert b.__dask_keys__() == [[(b.name, 0, 0)]]

    reduction_2d_test(da.sum, a, np.sum, x)
    reduction_2d_test(da.mean, a, np.mean, x)
    reduction_2d_test(da.var, a, np.var, x, False)  # Difference in dtype algo
    reduction_2d_test(da.std, a, np.std, x, False)  # Difference in dtype algo
    reduction_2d_test(da.min, a, np.min, x, False)
    reduction_2d_test(da.max, a, np.max, x, False)
    reduction_2d_test(da.any, a, np.any, x, False)
    reduction_2d_test(da.all, a, np.all, x, False)

    reduction_2d_test(da.nansum, a, np.nansum, x)
    reduction_2d_test(da.nanmean, a, np.mean, x)
    reduction_2d_test(da.nanvar, a, np.nanvar, x, False)  # Difference in dtype algo
    reduction_2d_test(da.nanstd, a, np.nanstd, x, False)  # Difference in dtype algo
    reduction_2d_test(da.nanmin, a, np.nanmin, x, False)
    reduction_2d_test(da.nanmax, a, np.nanmax, x, False)

    # prod/nanprod overflow for data at this size, leading to warnings about
    # overflow/invalid values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduction_2d_test(da.prod, a, np.prod, x)
        reduction_2d_test(da.nanprod, a, np.nanprod, x)


def test_reductions_2D_datetime():
    x = np.arange(1, 122).reshape(11, 11).astype("datetime64[ns]")
    a = da.from_array(x, chunks=(4, 4))
    reduction_2d_test(da.min, a, np.min, x, False)
    reduction_2d_test(da.max, a, np.max, x, False)
    reduction_2d_test(da.any, a, np.any, x, False)
    reduction_2d_test(da.all, a, np.all, x, False)
    reduction_2d_test(da.nanmin, a, np.nanmin, x, False)
    reduction_2d_test(da.nanmax, a, np.nanmax, x, False)


@pytest.mark.parametrize(
    ["dfunc", "func"],
    [
        (da.argmin, np.argmin),
        (da.argmax, np.argmax),
        (da.nanargmin, np.nanargmin),
        (da.nanargmax, np.nanargmax),
    ],
)
def test_arg_reductions(dfunc, func):
    x = np.random.default_rng().random((10, 10, 10))
    a = da.from_array(x, chunks=(3, 4, 5))

    assert_eq(dfunc(a), func(x))
    assert_eq(dfunc(a, 0), func(x, 0))
    assert_eq(dfunc(a, 1), func(x, 1))
    assert_eq(dfunc(a, 2), func(x, 2))
    with config.set(split_every=2):
        assert_eq(dfunc(a), func(x))
        assert_eq(dfunc(a, 0), func(x, 0))
        assert_eq(dfunc(a, 1), func(x, 1))
        assert_eq(dfunc(a, 2), func(x, 2))
    assert_eq(dfunc(a, keepdims=True), func(x, keepdims=True))

    pytest.raises(ValueError, lambda: dfunc(a, 3))
    pytest.raises(TypeError, lambda: dfunc(a, (0, 1)))

    x2 = np.arange(10)
    a2 = da.from_array(x2, chunks=3)
    assert_eq(dfunc(a2), func(x2))
    assert_eq(dfunc(a2, 0), func(x2, 0))
    assert_eq(dfunc(a2, 0, split_every=2), func(x2, 0))

    x3 = np.array(1)
    a3 = da.from_array(x3)
    assert_eq(dfunc(a3), func(x3))


@pytest.mark.parametrize(
    ["dfunc", "func"], [(da.nanmin, np.nanmin), (da.nanmax, np.nanmax)]
)
def test_nan_reduction_warnings(dfunc, func):
    x = np.random.default_rng().random((10, 10, 10))
    x[5] = np.nan
    a = da.from_array(x, chunks=(3, 4, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice encountered
        expected = func(x, 1)
    assert_eq(dfunc(a, 1), expected)


@pytest.mark.parametrize(
    ["dfunc", "func"], [(da.nanargmin, np.nanargmin), (da.nanargmax, np.nanargmax)]
)
def test_nanarg_reductions(dfunc, func):
    x = np.random.default_rng().random((10, 10, 10))
    x[5] = np.nan
    a = da.from_array(x, chunks=(3, 4, 5))
    assert_eq(dfunc(a), func(x))
    assert_eq(dfunc(a, 0), func(x, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slice encountered
        with pytest.raises(ValueError):
            dfunc(a, 1).compute()

        with pytest.raises(ValueError):
            dfunc(a, 2).compute()

        x[:] = np.nan
        a = da.from_array(x, chunks=(3, 4, 5))
        with pytest.raises(ValueError):
            dfunc(a).compute()


@pytest.mark.parametrize(["dfunc", "func"], [(da.min, np.min), (da.max, np.max)])
def test_min_max_empty_chunks(dfunc, func):
    x1 = np.arange(10)
    a1 = da.from_array(x1, chunks=1)
    assert_eq(dfunc(a1[a1 < 2]), func(x1[x1 < 2]))

    x2 = np.arange(10)
    a2 = da.from_array(x2, chunks=((5, 0, 5),))
    assert_eq(dfunc(a2), func(x2))

    x3 = np.array([[1, 1, 2, 3], [1, 1, 4, 0]])
    a3 = da.from_array(x3, chunks=1)
    assert_eq(dfunc(a3[a3 >= 2]), func(x3[x3 >= 2]))

    a4 = da.arange(10)
    with pytest.raises(
        ValueError
    ):  # Checking it mimics numpy behavior when all chunks are empty
        dfunc(a4[a4 < 0]).compute()


@pytest.mark.parametrize("func", ["argmax", "nanargmax"])
def test_arg_reductions_unknown_chunksize(func):
    x = da.arange(10, chunks=5)
    x = x[x > 1]

    with pytest.raises(ValueError) as info:
        getattr(da, func)(x)

    assert "unknown chunksize" in str(info.value)


@pytest.mark.parametrize("func", ["argmax", "nanargmax"])
def test_arg_reductions_unknown_chunksize_2d(func):
    x = da.ones((10, 10), chunks=(5, 5))
    x = x[x[0, :] > 0, :]  # unknown chunks in first dimension only

    with pytest.raises(ValueError):
        getattr(da, func)(x, axis=0)

    getattr(da, func)(x, axis=1).compute()


@pytest.mark.parametrize("func", ["argmax", "nanargmax"])
def test_arg_reductions_unknown_single_chunksize(func):
    x = da.ones((10, 10), chunks=(10, 10))
    x = x[x[0, :] > 0, :]  # unknown chunks in first dimension only

    getattr(da, func)(x, axis=0).compute()
    getattr(da, func)(x, axis=1).compute()


def test_reductions_2D_nans():
    # chunks are a mix of some/all/no NaNs
    x = np.full((4, 4), np.nan)
    x[:2, :2] = np.array([[1, 2], [3, 4]])
    x[2, 2] = 5
    x[3, 3] = 6
    a = da.from_array(x, chunks=(2, 2))

    reduction_2d_test(da.sum, a, np.sum, x, False, False)
    reduction_2d_test(da.prod, a, np.prod, x, False, False)
    reduction_2d_test(da.mean, a, np.mean, x, False, False)
    reduction_2d_test(da.var, a, np.var, x, False, False)
    reduction_2d_test(da.std, a, np.std, x, False, False)
    reduction_2d_test(da.min, a, np.min, x, False, False)
    reduction_2d_test(da.max, a, np.max, x, False, False)
    reduction_2d_test(da.any, a, np.any, x, False, False)
    reduction_2d_test(da.all, a, np.all, x, False, False)

    reduction_2d_test(da.nansum, a, np.nansum, x)
    reduction_2d_test(da.nanprod, a, np.nanprod, x)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reduction_2d_test(da.nanmean, a, np.nanmean, x, False, False)
        reduction_2d_test(da.nanvar, a, np.nanvar, x, False, False)
        reduction_2d_test(da.nanstd, a, np.nanstd, x, False, False)
        reduction_2d_test(da.nanmin, a, np.nanmin, x, False, False)
        reduction_2d_test(da.nanmax, a, np.nanmax, x, False, False)

        assert_eq(da.argmax(a), np.argmax(x))
        assert_eq(da.argmin(a), np.argmin(x))
        assert_eq(da.nanargmax(a), np.nanargmax(x))
        assert_eq(da.nanargmin(a), np.nanargmin(x))

        assert_eq(da.argmax(a, axis=0), np.argmax(x, axis=0))
        assert_eq(da.argmin(a, axis=0), np.argmin(x, axis=0))
        assert_eq(da.nanargmax(a, axis=0), np.nanargmax(x, axis=0))
        assert_eq(da.nanargmin(a, axis=0), np.nanargmin(x, axis=0))

        assert_eq(da.argmax(a, axis=1), np.argmax(x, axis=1))
        assert_eq(da.argmin(a, axis=1), np.argmin(x, axis=1))
        assert_eq(da.nanargmax(a, axis=1), np.nanargmax(x, axis=1))
        assert_eq(da.nanargmin(a, axis=1), np.nanargmin(x, axis=1))


def test_moment():
    def moment(x, n, axis=None):
        return ((x - x.mean(axis=axis, keepdims=True)) ** n).sum(
            axis=axis
        ) / np.ones_like(x).sum(axis=axis)

    # Poorly conditioned
    x = np.array([1.0, 2.0, 3.0] * 10).reshape((3, 10)) + 1e8
    a = da.from_array(x, chunks=5)
    assert_eq(a.moment(2), moment(x, 2))
    assert_eq(a.moment(3), moment(x, 3))
    assert_eq(a.moment(4), moment(x, 4))

    x = np.arange(1, 122).reshape((11, 11)).astype("f8")
    a = da.from_array(x, chunks=(4, 4))
    assert_eq(a.moment(4, axis=1), moment(x, 4, axis=1))
    assert_eq(a.moment(4, axis=(1, 0)), moment(x, 4, axis=(1, 0)))

    # Tree reduction
    assert_eq(a.moment(order=4, split_every=4), moment(x, 4))
    assert_eq(a.moment(order=4, axis=0, split_every=4), moment(x, 4, axis=0))
    assert_eq(a.moment(order=4, axis=1, split_every=4), moment(x, 4, axis=1))


def test_reductions_with_negative_axes():
    x = np.random.default_rng().random((4, 4, 4))
    a = da.from_array(x, chunks=2)

    assert_eq(a.argmin(axis=-1), x.argmin(axis=-1))
    assert_eq(a.argmin(axis=-1, split_every=2), x.argmin(axis=-1))

    assert_eq(a.sum(axis=-1), x.sum(axis=-1))
    assert_eq(a.sum(axis=(0, -1)), x.sum(axis=(0, -1)))


def test_nan():
    x = np.array([[1, np.nan, 3, 4], [5, 6, 7, np.nan], [9, 10, 11, 12]])
    d = da.from_array(x, chunks=(2, 2))

    assert_eq(np.nansum(x), da.nansum(d))
    assert_eq(np.nansum(x, axis=0), da.nansum(d, axis=0))
    assert_eq(np.nanmean(x, axis=1), da.nanmean(d, axis=1))
    assert_eq(np.nanmin(x, axis=1), da.nanmin(d, axis=1))
    assert_eq(np.nanmax(x, axis=(0, 1)), da.nanmax(d, axis=(0, 1)))
    assert_eq(np.nanvar(x), da.nanvar(d))
    assert_eq(np.nanstd(x, axis=0), da.nanstd(d, axis=0))
    assert_eq(np.nanargmin(x, axis=0), da.nanargmin(d, axis=0))
    assert_eq(np.nanargmax(x, axis=0), da.nanargmax(d, axis=0))
    assert_eq(np.nanprod(x), da.nanprod(d))


@pytest.mark.parametrize("func", ["nansum", "sum", "nanmin", "min", "nanmax", "max"])
def test_nan_object(func):
    with warnings.catch_warnings():
        if os.name == "nt" and func in {"min", "max"}:
            # RuntimeWarning: invalid value encountered in reduce in wrapreduction
            # from NumPy.
            warnings.simplefilter("ignore", RuntimeWarning)

        x = np.array([[1, np.nan, 3, 4], [5, 6, 7, np.nan], [9, 10, 11, 12]]).astype(
            object
        )
        d = da.from_array(x, chunks=(2, 2))

        if func in {"nanmin", "nanmax"}:
            warnings.simplefilter("ignore", RuntimeWarning)

        assert_eq(getattr(np, func)(x, axis=()), getattr(da, func)(d, axis=()))

        if func in {"nanmin", "nanmax"}:
            warnings.simplefilter("default", RuntimeWarning)

        if func in {"min", "max"}:
            warnings.simplefilter("ignore", RuntimeWarning)
        assert_eq(getattr(np, func)(x, axis=0), getattr(da, func)(d, axis=0))
        if os.name != "nt" and func in {"min", "max"}:
            warnings.simplefilter("default", RuntimeWarning)

        assert_eq(getattr(np, func)(x, axis=1), getattr(da, func)(d, axis=1))
        # wrap the scalar in a numpy array since the dask version cannot know dtype
        assert_eq(np.array(getattr(np, func)(x)).astype(object), getattr(da, func)(d))


def test_0d_array():
    x = da.mean(da.ones(4, chunks=4), axis=()).compute()
    x = da.mean(da.ones(4, chunks=4), axis=0).compute()
    y = np.mean(np.ones(4))
    assert type(x) == type(y)

    x = da.sum(da.zeros(4, chunks=1)).compute()
    y = np.sum(np.zeros(4))
    assert type(x) == type(y)


def test_reduction_on_scalar():
    x = da.from_array(np.array(1.0), chunks=())
    assert (x == x).all()


def test_reductions_with_empty_array():
    dx1 = da.ones((10, 0, 5), chunks=4)
    x1 = dx1.compute()
    dx2 = da.ones((0, 0, 0), chunks=4)
    x2 = dx2.compute()

    for dx, x in [(dx1, x1), (dx2, x2)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Mean of empty slice
            assert_eq(dx.mean(), x.mean())
            assert_eq(dx.mean(axis=()), x.mean(axis=()))
            assert_eq(dx.mean(axis=0), x.mean(axis=0))
            assert_eq(dx.mean(axis=1), x.mean(axis=1))
            assert_eq(dx.mean(axis=2), x.mean(axis=2))


def assert_max_deps(x, n, eq=True):
    dependencies, dependents = get_deps(x.dask)
    if eq:
        assert max(map(len, dependencies.values())) == n
    else:
        assert max(map(len, dependencies.values())) <= n


def test_tree_reduce_depth():
    # 2D
    x = da.from_array(np.arange(242).reshape((11, 22)), chunks=(3, 4))
    thresh = {0: 2, 1: 3}
    assert_max_deps(x.sum(split_every=thresh), 2 * 3)
    assert_max_deps(x.sum(axis=(), split_every=thresh), 1)
    assert_max_deps(x.sum(axis=0, split_every=thresh), 2)
    assert_max_deps(x.sum(axis=1, split_every=thresh), 3)
    assert_max_deps(x.sum(split_every=20), 20, False)
    assert_max_deps(x.sum(axis=(), split_every=20), 1)
    assert_max_deps(x.sum(axis=0, split_every=20), 4)
    assert_max_deps(x.sum(axis=1, split_every=20), 6)

    # 3D
    x = da.from_array(np.arange(11 * 22 * 29).reshape((11, 22, 29)), chunks=(3, 4, 5))
    thresh = {0: 2, 1: 3, 2: 4}
    assert_max_deps(x.sum(split_every=thresh), 2 * 3 * 4)
    assert_max_deps(x.sum(axis=(), split_every=thresh), 1)
    assert_max_deps(x.sum(axis=0, split_every=thresh), 2)
    assert_max_deps(x.sum(axis=1, split_every=thresh), 3)
    assert_max_deps(x.sum(axis=2, split_every=thresh), 4)
    assert_max_deps(x.sum(axis=(0, 1), split_every=thresh), 2 * 3)
    assert_max_deps(x.sum(axis=(0, 2), split_every=thresh), 2 * 4)
    assert_max_deps(x.sum(axis=(1, 2), split_every=thresh), 3 * 4)
    assert_max_deps(x.sum(split_every=20), 20, False)
    assert_max_deps(x.sum(axis=(), split_every=20), 1)
    assert_max_deps(x.sum(axis=0, split_every=20), 4)
    assert_max_deps(x.sum(axis=1, split_every=20), 6)
    assert_max_deps(x.sum(axis=2, split_every=20), 6)
    assert_max_deps(x.sum(axis=(0, 1), split_every=20), 20, False)
    assert_max_deps(x.sum(axis=(0, 2), split_every=20), 20, False)
    assert_max_deps(x.sum(axis=(1, 2), split_every=20), 20, False)
    assert_max_deps(x.sum(axis=(0, 1), split_every=40), 4 * 6)
    assert_max_deps(x.sum(axis=(0, 2), split_every=40), 4 * 6)
    assert_max_deps(x.sum(axis=(1, 2), split_every=40), 6 * 6)


def test_tree_reduce_set_options():
    x = da.from_array(np.arange(242).reshape((11, 22)), chunks=(3, 4))
    with config.set(split_every={0: 2, 1: 3}):
        assert_max_deps(x.sum(), 2 * 3)
        assert_max_deps(x.sum(axis=()), 1)
        assert_max_deps(x.sum(axis=0), 2)


def test_reduction_names():
    x = da.ones(5, chunks=(2,))
    assert x.sum().name.startswith("sum")
    assert "max" in x.max().name.split("-")[0]
    assert x.var().name.startswith("var")
    assert x.all().name.startswith("all")
    assert any(k[0].startswith("nansum") for k in da.nansum(x).dask)
    assert x.mean().name.startswith("mean")


def test_general_reduction_names():
    dtype = int
    a = da.reduction(
        da.ones(10, dtype, chunks=2), np.sum, np.sum, dtype=dtype, name="foo"
    )
    names, tokens = list(zip_longest(*[key[0].rsplit("-", 1) for key in a.dask]))
    assert set(names) == {"ones_like", "foo", "foo-partial", "foo-aggregate"}
    assert all(tokens)


@pytest.mark.parametrize("func", [np.sum, np.argmax])
def test_array_reduction_out(func):
    x = da.arange(10, chunks=(5,))
    y = da.ones((10, 10), chunks=(4, 4))
    func(y, axis=0, out=x)
    assert_eq(x, func(np.ones((10, 10)), axis=0))


@pytest.mark.parametrize("func", ["cumsum", "cumprod", "nancumsum", "nancumprod"])
@pytest.mark.parametrize("use_nan", [False, True])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("method", ["sequential", "blelloch"])
def test_array_cumreduction_axis(func, use_nan, axis, method):
    np_func = getattr(np, func)
    da_func = getattr(da, func)

    s = (10, 11, 12)
    a = np.arange(np.prod(s), dtype=float).reshape(s)
    if use_nan:
        a[1] = np.nan
    d = da.from_array(a, chunks=(4, 5, 6))
    if func in ["cumprod", "nancumprod"] and method == "blelloch" and axis is None:
        with pytest.warns(RuntimeWarning):
            da_func(d, axis=axis, method=method).compute()
            return

    a_r = np_func(a, axis=axis)
    d_r = da_func(d, axis=axis, method=method)

    assert_eq(a_r, d_r)


@pytest.mark.parametrize("func", ["cumsum", "cumprod", "nancumsum", "nancumprod"])
@pytest.mark.parametrize("method", ["sequential", "blelloch"])
@pytest.mark.parametrize("target_dtype", [None, int, float])
def test_array_cumreduction_dtype(func, method, target_dtype):
    np_func = getattr(np, func)
    da_func = getattr(da, func)

    a = np.linspace(0, 1, num=10, dtype=float)
    d = da.from_array(a)

    a_r = np_func(a, axis=0, dtype=target_dtype)
    d_r = da_func(d, method=method, axis=0, dtype=target_dtype)

    assert_eq(a_r, d_r)


@pytest.mark.parametrize("ufunc", ["add", "multiply", "maximum"])
@pytest.mark.parametrize("target_dtype", [None, int, float])
def test_array_cumreduction_ufunc(ufunc, target_dtype):
    ufunc_obj = getattr(np, ufunc)
    accumulate = ufunc_obj.accumulate
    identity = ufunc_obj.identity

    a = np.linspace(0, 1, num=10, dtype=float)
    d = da.from_array(a)

    cumreduction = da.reductions.cumreduction

    a_r = accumulate(a, dtype=target_dtype)
    d_r = cumreduction(accumulate, ufunc_obj, identity, d, dtype=target_dtype)

    assert_eq(a_r, d_r)


@pytest.mark.parametrize("func", [np.cumsum, np.cumprod])
def test_array_cumreduction_out(func):
    x = da.ones((10, 10), chunks=(4, 4))
    func(x, axis=0, out=x)
    assert_eq(x, func(np.ones((10, 10)), axis=0))


@pytest.mark.parametrize(
    "npfunc,daskfunc", [(np.sort, da.topk), (np.argsort, da.argtopk)]
)
@pytest.mark.parametrize("split_every", [None, 2, 4, 8])
def test_topk_argtopk1(npfunc, daskfunc, split_every):
    # Test data
    k = 5
    # Test at least 3 levels of aggregation when split_every=2
    # to stress the different chunk, combine, aggregate kernels
    rng = np.random.default_rng()
    npa = rng.random(800)
    npb = rng.random((10, 20, 30))

    a = da.from_array(npa, chunks=((120, 80, 100, 200, 300),))
    b = da.from_array(npb, chunks=(4, 8, 8))

    # 1-dimensional arrays
    # top 5 elements, sorted descending
    assert_eq(npfunc(npa)[-k:][::-1], daskfunc(a, k, split_every=split_every))
    # bottom 5 elements, sorted ascending
    assert_eq(npfunc(npa)[:k], daskfunc(a, -k, split_every=split_every))

    # n-dimensional arrays
    # also testing when k > chunk
    # top 5 elements, sorted descending
    assert_eq(
        npfunc(npb, axis=0)[-k:, :, :][::-1, :, :],
        daskfunc(b, k, axis=0, split_every=split_every),
    )
    assert_eq(
        npfunc(npb, axis=1)[:, -k:, :][:, ::-1, :],
        daskfunc(b, k, axis=1, split_every=split_every),
    )
    assert_eq(
        npfunc(npb, axis=-1)[:, :, -k:][:, :, ::-1],
        daskfunc(b, k, axis=-1, split_every=split_every),
    )
    with pytest.raises(ValueError):
        daskfunc(b, k, axis=3, split_every=split_every)

    # bottom 5 elements, sorted ascending
    assert_eq(
        npfunc(npb, axis=0)[:k, :, :], daskfunc(b, -k, axis=0, split_every=split_every)
    )
    assert_eq(
        npfunc(npb, axis=1)[:, :k, :], daskfunc(b, -k, axis=1, split_every=split_every)
    )
    assert_eq(
        npfunc(npb, axis=-1)[:, :, :k],
        daskfunc(b, -k, axis=-1, split_every=split_every),
    )
    with pytest.raises(ValueError):
        daskfunc(b, -k, axis=3, split_every=split_every)


@pytest.mark.parametrize(
    "npfunc,daskfunc", [(np.sort, da.topk), (np.argsort, da.argtopk)]
)
@pytest.mark.parametrize("split_every", [None, 2, 3, 4])
@pytest.mark.parametrize("chunksize", [1, 2, 3, 4, 5, 10])
def test_topk_argtopk2(npfunc, daskfunc, split_every, chunksize):
    """Fine test use cases when k is larger than chunk size"""
    npa = np.random.default_rng().random((10,))
    a = da.from_array(npa, chunks=chunksize)
    k = 5

    # top 5 elements, sorted descending
    assert_eq(npfunc(npa)[-k:][::-1], daskfunc(a, k, split_every=split_every))
    # bottom 5 elements, sorted ascending
    assert_eq(npfunc(npa)[:k], daskfunc(a, -k, split_every=split_every))


def test_topk_argtopk3():
    a = da.random.default_rng().random((10, 20, 30), chunks=(4, 8, 8))

    # As Array methods
    assert_eq(a.topk(5, axis=1, split_every=2), da.topk(a, 5, axis=1, split_every=2))
    assert_eq(
        a.argtopk(5, axis=1, split_every=2), da.argtopk(a, 5, axis=1, split_every=2)
    )


@pytest.mark.parametrize(
    "func",
    [da.cumsum, da.cumprod, da.argmin, da.argmax, da.min, da.max, da.nansum, da.nanmax],
)
@pytest.mark.parametrize("method", ["sequential", "blelloch"])
def test_regres_3940(func, method):
    if func in {da.cumsum, da.cumprod}:
        kwargs = {"method": method}
    else:
        kwargs = {}
    a = da.ones((5, 2), chunks=(2, 2))
    assert func(a, **kwargs).name != func(a + 1, **kwargs).name
    assert func(a, axis=0, **kwargs).name != func(a, **kwargs).name
    assert func(a, axis=0, **kwargs).name != func(a, axis=1, **kwargs).name
    if func not in {da.cumsum, da.cumprod, da.argmin, da.argmax}:
        assert func(a, axis=()).name != func(a).name
        assert func(a, axis=()).name != func(a, axis=0).name


def test_trace():
    def _assert(a, b, *args, **kwargs):
        return assert_eq(a.trace(*args, **kwargs), b.trace(*args, **kwargs))

    b = np.arange(12).reshape((3, 4))
    a = da.from_array(b, 1)
    _assert(a, b)
    _assert(a, b, 0)
    _assert(a, b, 1)
    _assert(a, b, -1)

    b = np.arange(8).reshape((2, 2, 2))
    a = da.from_array(b, 2)
    _assert(a, b)
    _assert(a, b, 0)
    _assert(a, b, 1)
    _assert(a, b, -1)
    _assert(a, b, 0, 0, 1)
    _assert(a, b, 0, 0, 2)
    _assert(a, b, 0, 1, 2, int)
    _assert(a, b, 0, 1, 2, float)
    _assert(a, b, offset=1, axis1=0, axis2=2, dtype=int)
    _assert(a, b, offset=1, axis1=0, axis2=2, dtype=float)


@pytest.mark.parametrize("func", ["median", "nanmedian"])
@pytest.mark.parametrize("axis", [0, [0, 1], 1, -1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_median(axis, keepdims, func):
    x = np.arange(100).reshape((2, 5, 10))
    d = da.from_array(x, chunks=2)
    assert_eq(
        getattr(da, func)(d, axis=axis, keepdims=keepdims),
        getattr(np, func)(x, axis=axis, keepdims=keepdims),
    )


@pytest.mark.parametrize("func", ["median", "nanmedian"])
@pytest.mark.parametrize("axis", [0, [0, 2], 1])
def test_median_does_not_rechunk_if_whole_axis_in_one_chunk(axis, func):
    x = np.arange(100).reshape((2, 5, 10))
    d = da.from_array(x, chunks=(2, 1, 10))

    actual = getattr(da, func)(d, axis=axis)
    expected = getattr(np, func)(x, axis=axis)
    assert_eq(actual, expected)
    does_rechunk = "rechunk" in str(dict(actual.__dask_graph__()))
    if axis == 1:
        assert does_rechunk
    else:
        assert not does_rechunk


@pytest.mark.parametrize("method", ["sum", "mean", "prod"])
def test_object_reduction(method):
    arr = da.ones(1).astype(object)
    result = getattr(arr, method)().compute()
    assert result == 1


@pytest.mark.parametrize("func", ["nanmin", "nanmax"])
def test_empty_chunk_nanmin_nanmax(func):
    # see https://github.com/dask/dask/issues/8352
    x = np.arange(10).reshape(2, 5)
    d = da.from_array(x, chunks=2)
    x = x[x > 4]
    d = d[d > 4]
    block_lens = np.array([len(x.compute()) for x in d.blocks])
    assert 0 in block_lens
    with pytest.raises(ValueError) as err:
        getattr(da, func)(d)
    assert "Arrays chunk sizes are unknown" in str(err)
    d = d.compute_chunk_sizes()
    assert_eq(getattr(da, func)(d), getattr(np, func)(x))


@pytest.mark.parametrize("func", ["nanmin", "nanmax"])
def test_empty_chunk_nanmin_nanmax_raise(func):
    # see https://github.com/dask/dask/issues/8352
    x = np.arange(10).reshape(2, 5)
    d = da.from_array(x, chunks=2)
    d = d[d > 9]
    x = x[x > 9]
    d = d.compute_chunk_sizes()
    with pytest.raises(ValueError) as err_np:
        getattr(np, func)(x)
    with pytest.raises(ValueError) as err_da:
        d = getattr(da, func)(d)
        d.compute()
    assert str(err_np.value) == str(err_da.value)


def test_mean_func_does_not_warn():
    # non-regression test for https://github.com/pydata/xarray/issues/5151
    xr = pytest.importorskip("xarray")
    a = xr.DataArray(da.from_array(np.full((10, 10), np.nan)))

    with warnings.catch_warnings(record=True) as rec:
        a.mean().compute()
    assert not rec  # did not warn


@pytest.mark.parametrize("func", ["nanvar", "nanstd"])
def test_nan_func_does_not_warn(func):
    # non-regression test for #6105
    x = np.ones((10,)) * np.nan
    x[0] = 1
    x[1] = 2
    d = da.from_array(x, chunks=2)
    with warnings.catch_warnings(record=True) as rec:
        getattr(da, func)(d).compute()
    assert not rec  # did not warn


@pytest.mark.parametrize("chunks", list(permutations(((2, 1) * 8, (3,) * 8, (6,) * 4))))
@pytest.mark.parametrize("split_every", [2, 4])
@pytest.mark.parametrize(
    "axes", list(permutations((0, 1, 2), 2)) + list(permutations((0, 1, 2)))
)
def test_chunk_structure_independence(axes, split_every, chunks):
    # Reducing an array should not depend on its chunk-structure!!!
    # See Issue #8541: https://github.com/dask/dask/issues/8541
    shape = tuple(np.sum(s) for s in chunks)
    np_array = np.arange(np.prod(shape)).reshape(*shape)
    x = da.from_array(np_array, chunks=chunks)
    reduced_x = da.reduction(
        x,
        lambda x, axis, keepdims: x,
        lambda x, axis, keepdims: x,
        keepdims=True,
        axis=axes,
        split_every=split_every,
        dtype=x.dtype,
        meta=x._meta,
    )
    assert_eq(reduced_x, np_array, check_chunks=False, check_shape=False)


def test_weighted_reduction():
    # Weighted reduction
    def w_sum(x, weights=None, dtype=None, computing_meta=False, **kwargs):
        """`chunk` callable for (weighted) sum"""
        if computing_meta:
            return x
        if weights is not None:
            x = x * weights
        return np.sum(x, dtype=dtype, **kwargs)

    # Arrays
    a = 1 + np.ma.arange(60).reshape(6, 10)
    a[2, 2] = np.ma.masked
    dx = da.from_array(a, chunks=(4, 5))
    # Weights
    w = np.linspace(1, 2, 6).reshape(6, 1)

    # No weights (i.e. normal sum)
    x = da.reduction(dx, w_sum, np.sum, dtype=dx.dtype)
    assert_eq(x, np.sum(a), check_shape=True)

    # Weighted sum
    x = da.reduction(dx, w_sum, np.sum, dtype="f8", weights=w)
    assert_eq(x, np.sum(a * w), check_shape=True)

    # Non-broadcastable weights (short axis)
    with pytest.raises(ValueError):
        da.reduction(dx, w_sum, np.sum, weights=[1, 2, 3])

    # Non-broadcastable weights (too many dims)
    with pytest.raises(ValueError):
        da.reduction(dx, w_sum, np.sum, weights=[[[2]]])


def test_cumreduction_no_rechunk_on_1d_array():
    x = da.ones((5,))
    y = da.cumsum(x)
    no_rechunk = "rechunk" not in str(dict(y.__dask_graph__()))
    assert no_rechunk


@pytest.mark.parametrize("axis", [3, 0, [1, 3]])
@pytest.mark.parametrize("q", [0.75, [0.75], [0.75, 0.4]])
@pytest.mark.parametrize("rechunk", [True, False])
def test_nanquantile(rechunk, q, axis):
    shape = 7, 10, 7, 10
    arr = np.random.randn(*shape)
    indexer = np.random.randint(0, 10, size=shape)
    arr[indexer >= 8] = np.nan
    arr[:, :, :, 1] = 1
    arr[1, :, :, :] = 1

    darr = da.from_array(arr, chunks=(2, 3, 4, (5 if rechunk else -1)))
    assert_eq(da.nanquantile(darr, q, axis=axis), np.nanquantile(arr, q, axis=axis))
    assert_eq(
        da.nanquantile(darr, q, axis=axis, keepdims=True),
        np.nanquantile(arr, q, axis=axis, keepdims=True),
    )
    assert_eq(
        da.nanpercentile(darr, q * 100, axis=axis),
        np.nanpercentile(arr, q * 100, axis=axis),
    )
    assert_eq(
        da.nanpercentile(darr, q * 100, axis=axis, keepdims=True),
        np.nanpercentile(arr, q * 100, axis=axis, keepdims=True),
    )


@pytest.mark.parametrize("axis", [3, [1, 3]])
@pytest.mark.parametrize("q", [0.75, [0.75]])
@pytest.mark.parametrize("rechunk", [True, False])
def test_quantile(rechunk, q, axis):
    shape = 10, 15, 20, 15
    arr = np.random.randn(*shape)
    indexer = np.random.randint(0, 10, size=shape)
    arr[indexer >= 8] = np.nan

    darr = da.from_array(arr, chunks=(2, 3, 4, (5 if rechunk else -1)))
    assert_eq(da.quantile(darr, q, axis=axis), np.quantile(arr, q, axis=axis))
    assert_eq(
        da.quantile(darr, q, axis=axis, keepdims=True),
        np.quantile(arr, q, axis=axis, keepdims=True),
    )
    assert_eq(da.percentile(darr, q, axis=axis), np.percentile(arr, q, axis=axis))
    assert_eq(
        da.percentile(darr, q, axis=axis, keepdims=True),
        np.percentile(arr, q, axis=axis, keepdims=True),
    )


quantile_nanquantile = pytest.mark.parametrize(
    "np_func,da_func",
    [
        pytest.param(np.quantile, da.quantile, id="quantile"),
        pytest.param(np.nanquantile, da.nanquantile, id="nanquantile"),
    ],
)


@pytest.mark.skipif(not NUMPY_GE_200, reason="weights requires numpy>=2.0")
@pytest.mark.parametrize("q", [0.75, [0.75, 0.4]])
@pytest.mark.parametrize("rechunk", [True, False])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dask_weights", [True, False])
@quantile_nanquantile
def test_quantile_weights(np_func, da_func, rechunk, q, keepdims, dask_weights):
    shape = 10, 15, 20, 15
    chunks = (2, 3, 4, (5 if rechunk else -1))
    arr = np.random.randn(*shape)
    # 1-D weights whose length matches the reduction axis size
    weights = np.random.uniform(0.5, 2.0, size=shape[-1])
    if dask_weights:
        dweights = da.from_array(weights, chunks=(chunks[-1],))
    else:
        dweights = weights
    # np.nanquantile has a bug with 1-D weights on multi-D arrays
    if np_func is np.nanquantile:
        weights = np.broadcast_to(weights, arr.shape)

    darr = da.from_array(arr, chunks=chunks)

    assert_eq(
        da_func(
            darr,
            q,
            axis=-1,
            weights=dweights,
            method="inverted_cdf",
            keepdims=keepdims,
        ),
        np_func(
            arr,
            q,
            axis=-1,
            weights=weights,
            method="inverted_cdf",
            keepdims=keepdims,
        ),
    )


@pytest.mark.skipif(not NUMPY_GE_200, reason="weights requires numpy>=2.0")
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("q", [0.75, [0.75, 0.4]])
@pytest.mark.parametrize("rechunk", [True, False])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dask_weights", [True, False])
@quantile_nanquantile
def test_quantile_weights_non_last_axis(
    np_func, da_func, axis, rechunk, q, keepdims, dask_weights
):
    shape = 10, 15, 20, 15
    chunks = (2, 3, 4, (5 if rechunk else -1))
    arr = np.random.randn(*shape)
    weights = np.random.uniform(0.5, 2.0, size=shape[axis])
    if dask_weights:
        dweights = da.from_array(weights, chunks=(chunks[axis],))
    else:
        dweights = weights
    # np.nanquantile has a bug with 1-D weights on multi-D arrays
    if np_func is np.nanquantile:
        weights_shape = [1, 1, 1, 1]
        weights_shape[axis] = -1
        weights = np.broadcast_to(weights.reshape(tuple(weights_shape)), arr.shape)

    darr = da.from_array(arr, chunks=chunks)

    assert_eq(
        da_func(
            darr,
            q,
            axis=axis,
            weights=dweights,
            method="inverted_cdf",
            keepdims=keepdims,
        ),
        np_func(
            arr,
            q,
            axis=axis,
            weights=weights,
            method="inverted_cdf",
            keepdims=keepdims,
        ),
    )


@quantile_nanquantile
def test_quantile_dask_q(np_func, da_func):
    shape = 10, 15, 20, 15
    arr = np.random.randn(*shape)
    darr = da.from_array(arr, chunks=(2, 3, 4, -1))
    q = np.array([0.25, 0.5, 0.75])
    dq = da.from_array(q)

    assert_eq(da_func(darr, dq, axis=-1), np_func(arr, q, axis=-1))
    assert_eq(da_func(darr, dq, axis=0), np_func(arr, q, axis=0))


@pytest.mark.parametrize("axis", [-1, 0])
@pytest.mark.parametrize("rechunk", [True, False])
@quantile_nanquantile
def test_quantile_nd_q(xfail, np_func, da_func, axis, rechunk):
    if not NUMPY_GE_200 and np_func is np.nanquantile and axis == -1:
        xfail("Incorrect outputs on numpy<2")

    shape = 10, 15, 20, 15
    arr = np.random.randn(*shape)
    darr = da.from_array(arr, chunks=(2, 3, 4, (5 if rechunk else -1)))
    q = np.array([[0.1, 0.5], [0.9, 0.95]])  # shape (2, 2)

    assert_eq(da_func(darr, q, axis=axis), np_func(arr, q, axis=axis))
    assert_eq(
        da_func(darr, q, axis=axis, keepdims=True),
        np_func(arr, q, axis=axis, keepdims=True),
    )


@pytest.mark.parametrize(
    "a_dtype", [np.float32, np.float64, np.int16, np.int32, np.uint16, np.uint32]
)
@pytest.mark.parametrize("q_dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no_weights"),
        pytest.param(
            {
                "weights": np.asarray([0.4, 0.6, 1.0], dtype=np.float32),
                "method": "inverted_cdf",
            },
            marks=pytest.mark.skipif(
                not NUMPY_GE_200, reason="weights requires numpy>=2.0"
            ),
            id="weights",
        ),
    ],
)
@quantile_nanquantile
def test_quantile_dtype(np_func, da_func, kwargs, a_dtype, q_dtype):
    arr = np.asarray([100, 200, 300], dtype=a_dtype)
    darr = da.from_array(arr)
    q = np.asarray([0.5], dtype=q_dtype)

    assert_eq(
        da_func(darr, q, **kwargs),
        np_func(arr, q, **kwargs),
    )


@pytest.mark.parametrize("func", [da.quantile, da.nanquantile, da.nanpercentile])
def test_quantile_func_family_with_axis_none(func):
    # Check that these functions raise a NotImplementedError
    # when axis=None and more than one chunk is present
    # along at least one dimension
    darr = da.ones((3, 3), chunks=(2, 2))
    with pytest.raises(
        NotImplementedError, match="The full algorithm is difficult to do in parallel"
    ):
        func(darr, 0.5, axis=None)

    # Check that the functions behave as expected
    # when axis=None and the array is a single chunk
    darr = da.from_array([-1, 0, 1])
    out = func(darr, 0.0, axis=None)
    assert_eq(out, -1.0)


def test_nanquantile_all_nan():
    shape = 10, 15, 20, 15
    arr = np.random.randn(*shape)
    arr[:] = np.nan
    darr = da.from_array(arr, chunks=(2, 3, 4, -1))
    da.nanquantile(darr, 0.75, axis=-1).compute()
    with pytest.raises(RuntimeWarning):
        assert_eq(
            da.nanquantile(darr, 0.75, axis=-1), np.nanquantile(arr, 0.75, axis=-1)
        )
        assert_eq(da.percentile(darr, 0.75, axis=-1), np.percentile(arr, 0.75, axis=-1))


def test_nanquantile_method():
    shape = 10, 15, 20, 15
    arr = np.random.randn(*shape)
    indexer = np.random.randint(0, 10, size=shape)
    arr[indexer >= 8] = np.nan
    darr = da.from_array(arr, chunks=(2, 3, 4, -1))
    assert_eq(
        da.nanquantile(darr, 0.75, axis=-1, method="weibull"),
        np.nanquantile(arr, 0.75, axis=-1, method="weibull"),
    )
    assert_eq(
        da.nanpercentile(darr, 0.75, axis=-1, method="weibull"),
        np.nanpercentile(arr, 0.75, axis=-1, method="weibull"),
    )


def test_nanquantile_one_dim():
    arr = np.random.randn(10)
    darr = da.from_array(arr, chunks=(2,))
    assert_eq(da.nanquantile(darr, 0.75, axis=-1), np.nanquantile(arr, 0.75, axis=-1))


def test_nanquantile_two_dims():
    arr = np.random.randn(10, 10)
    darr = da.from_array(arr, chunks=(2, -1))
    assert_eq(da.nanquantile(darr, 0.75, axis=-1), np.nanquantile(arr, 0.75, axis=-1))
    assert_eq(
        da.nanpercentile(darr, 0.75, axis=-1), np.nanpercentile(arr, 0.75, axis=-1)
    )


# --------------------------------------------------------------------------
# Exact, chunking-independent reductions (exact=True)
# --------------------------------------------------------------------------


def _reduced_columns(x, axis):
    """Reshape ``x`` so every reduced slice is a row (columns of the result)."""
    ndim = x.ndim
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    axis = tuple(ax % ndim for ax in axis)
    work = np.moveaxis(x, axis, tuple(range(ndim - len(axis), ndim)))
    reduced_sizes = [x.shape[i] for i in axis]
    kept_shape = tuple(s for i, s in enumerate(x.shape) if i not in axis)
    k = int(np.prod(reduced_sizes, dtype=int)) if reduced_sizes else 1
    m = int(np.prod(kept_shape, dtype=int)) if kept_shape else 1
    return work.reshape(m, k), axis


def _fsum_reference(x, axis=None, keepdims=False):
    """Correctly rounded reference using ``math.fsum`` (float64/complex128)."""
    cols, axis = _reduced_columns(x, axis)
    if cols.dtype.kind == "c":
        values = [
            complex(math.fsum(col.real.tolist()), math.fsum(col.imag.tolist()))
            for col in cols
        ]
    else:
        values = [math.fsum(col.tolist()) for col in cols]
    out = np.array(values, dtype=x.dtype).reshape(
        tuple(1 if i in axis else x.shape[i] for i in range(x.ndim))
    )
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _fraction_mean_reference(x, axis=None, keepdims=False):
    cols, axis = _reduced_columns(x, axis)
    values = []
    for col in cols:
        total = sum((Fraction(float(v)) for v in col), Fraction(0))
        values.append(float(total / col.size))
    out = np.array(values, dtype=x.dtype).reshape(
        tuple(1 if i in axis else x.shape[i] for i in range(x.ndim))
    )
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


@pytest.mark.normal_and_array_expr
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2), (1, 2), (0, 1, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_exact_sum_reference_f64(axis, keepdims):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 7, 8))
    d = da.from_array(x, chunks=(2, 3, 4))
    assert_eq(
        da.sum(d, axis=axis, keepdims=keepdims, exact=True),
        _fsum_reference(x, axis, keepdims),
    )


@pytest.mark.normal_and_array_expr
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_exact_sum_reference_complex(axis, keepdims):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 7)) + 1j * rng.normal(size=(6, 7))
    d = da.from_array(x, chunks=(2, 3))
    assert_eq(
        da.sum(d, axis=axis, keepdims=keepdims, exact=True),
        _fsum_reference(x, axis, keepdims),
    )


@pytest.mark.normal_and_array_expr
@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_exact_mean_reference(axis, keepdims):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 7, 8))
    d = da.from_array(x, chunks=(2, 3, 4))
    assert_eq(
        da.mean(d, axis=axis, keepdims=keepdims, exact=True),
        _fraction_mean_reference(x, axis, keepdims),
    )


@pytest.mark.normal_and_array_expr
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16"])
@pytest.mark.parametrize("funcname", ["sum", "nansum", "mean", "nanmean"])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 2)])
def test_exact_reductions_are_chunk_independent(dtype, funcname, axis):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 7, 8))
    if dtype.startswith("c"):
        x = x + 1j * rng.normal(size=(6, 7, 8))
    x = x.astype(dtype)
    func = getattr(da, funcname)
    chunkings = [(2, 3, 4), (6, 7, 8), (3, 1, 5), (4, 2, 3)]
    results = [
        func(da.from_array(x, chunks=chunks), axis=axis, exact=True).compute()
        for chunks in chunkings
    ]
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)
    # The tree width must not matter either.
    shallow = func(
        da.from_array(x, chunks=(2, 3, 4)), axis=axis, split_every=8, exact=True
    ).compute()
    deep = func(
        da.from_array(x, chunks=(2, 3, 4)), axis=axis, split_every=2, exact=True
    ).compute()
    np.testing.assert_array_equal(shallow, deep)


@pytest.mark.normal_and_array_expr
def test_exact_sum_dtype_semantics():
    rng = np.random.default_rng(4)
    x32 = rng.normal(size=100).astype(np.float32)
    d = da.from_array(x32, chunks=7)
    assert da.sum(d, exact=True).compute().dtype == np.float32
    assert da.mean(d, exact=True).compute().dtype == np.float32
    promoted = da.sum(d, dtype="f8", exact=True).compute()
    assert promoted.dtype == np.float64
    assert promoted == math.fsum(x32.tolist())
    z64 = (rng.normal(size=20) + 1j * rng.normal(size=20)).astype(np.complex64)
    assert da.sum(da.from_array(z64, chunks=4), exact=True).compute().dtype == np.complex64


@pytest.mark.normal_and_array_expr
def test_exact_reductions_nan_and_inf():
    y = np.array([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0])
    d = da.from_array(y, chunks=2)
    assert np.isnan(da.sum(d, exact=True).compute())
    assert np.isnan(da.mean(d, exact=True).compute())
    # NaN skipped, but +inf and -inf still cancel to NaN
    assert np.isnan(da.nansum(d, exact=True).compute())
    assert np.isnan(da.nanmean(d, exact=True).compute())

    clean = np.array([1.0, np.nan, 2.0, 3.0])
    dc = da.from_array(clean, chunks=2)
    assert da.nansum(dc, exact=True).compute() == 6.0
    assert da.nanmean(dc, exact=True).compute() == 2.0

    infs = np.array([np.inf, 1.0, np.nan])
    di = da.from_array(infs, chunks=2)
    assert da.nansum(di, exact=True).compute() == np.inf
    assert da.nanmean(di, exact=True).compute() == np.inf

    z = np.array([1 + 2j, complex(np.nan, 3.0), 4 + 0j], dtype=np.complex128)
    dz = da.from_array(z, chunks=2)
    # Plain propagation is componentwise
    propagated = da.sum(dz, exact=True).compute()
    assert np.isnan(propagated.real) and propagated.imag == 5.0
    # nansum drops the whole NaN element, matching np.nansum
    got = da.nansum(dz, exact=True).compute()
    assert got == np.nansum(z) == (5 + 2j)
    assert da.nanmean(dz, exact=True).compute() == np.nanmean(z)

    # Axis-wise flags
    x = np.array([[1.0, np.nan], [np.inf, 2.0]])
    dx = da.from_array(x, chunks=(1, 1))
    np.testing.assert_array_equal(
        da.nansum(dx, axis=0, exact=True).compute(), [np.inf, 2.0]
    )


@pytest.mark.normal_and_array_expr
def test_exact_reductions_empty():
    empty = da.from_array(np.zeros((0,), "f8"), chunks=())
    assert da.sum(empty, exact=True).compute() == 0.0
    assert da.nansum(empty, exact=True).compute() == 0.0
    assert np.isnan(da.mean(empty, exact=True).compute())
    assert np.isnan(da.nanmean(empty, exact=True).compute())
    assert da.sum(empty, exact=True, keepdims=True).compute().shape == (1,)

    x = da.from_array(np.zeros((0, 4), "f8"), chunks=(0, 4))
    np.testing.assert_array_equal(da.sum(x, axis=0, exact=True).compute(), np.zeros(4))
    assert np.all(np.isnan(da.mean(x, axis=0, exact=True).compute()))
    assert da.sum(x, axis=1, exact=True).compute().shape == (0,)

    all_nan = da.from_array(np.full(3, np.nan), chunks=2)
    assert da.nansum(all_nan, exact=True).compute() == 0.0
    assert np.isnan(da.nanmean(all_nan, exact=True).compute())


@pytest.mark.normal_and_array_expr
def test_exact_reductions_integer_fallback_and_errors():
    x = da.from_array(np.arange(6).reshape(2, 3), chunks=2)
    # Integer arithmetic is already order-independent; exact=True is a no-op.
    assert_eq(da.sum(x, exact=True), da.sum(x))
    assert_eq(da.mean(x, exact=True), da.mean(x))
    assert x.sum(exact=True).compute() == 15
    assert x.mean(exact=True).compute() == 2.5

    with pytest.raises(TypeError, match="floating-point and complex"):
        da.sum(da.from_array(np.array([object()], object), chunks=1), exact=True)
    # Masked arrays keep their mask outside of ndarray semantics
    masked = da.from_array(np.ma.array([1.0, 2.0], mask=[0, 1]), chunks=1)
    with pytest.raises(TypeError, match="plain NumPy"):
        da.sum(masked, exact=True)
    rng = np.random.default_rng(5)
    z = rng.normal(size=4) + 1j * rng.normal(size=4)
    with pytest.raises(TypeError, match="complex"):
        da.sum(da.from_array(z, chunks=2), exact=True, dtype="f8")
    with pytest.raises(TypeError, match="dtype"):
        da.sum(da.from_array(rng.normal(size=4), chunks=2), exact=True, dtype="i8")


@pytest.mark.normal_and_array_expr
def test_exact_reductions_lazy_out_and_methods():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(6, 7, 8))
    d = da.from_array(x, chunks=3)
    # Graph construction never materializes data
    lazy = da.sum(d, exact=True)
    assert isinstance(lazy, da.Array)
    total = math.fsum(x.ravel().tolist())
    assert lazy.compute() == total
    assert d.sum(exact=True).compute() == total
    assert d.mean(exact=True).compute() == float(
        sum((Fraction(float(v)) for v in x.ravel()), Fraction(0)) / x.size
    )

    out = da.zeros((), dtype="f8", chunks=())
    result = da.sum(d, out=out, exact=True)
    assert result.compute() == total
    # The rejected/ordinary path still builds and computes independently
    ordinary = da.sum(d)
    assert isinstance(ordinary, da.Array)
    ordinary.compute()


@pytest.mark.normal_and_array_expr
def test_fraction_to_float_rounding():
    from dask.array.reductions import _fraction_to_float as round_fraction

    assert round_fraction(Fraction(0), "f8") == 0.0
    assert round_fraction(Fraction(1, 10), "f4") == np.float32(0.1)

    # Exact ties resolve to even mantissa at every precision
    tie32 = Fraction(1) + Fraction(1, 1 << 24)
    assert round_fraction(tie32, "f4") == np.float32(1.0)
    assert round_fraction(tie32 + Fraction(1, 1 << 30), "f4") > np.float32(1.0)
    assert round_fraction(Fraction(1) + Fraction(1, 1 << 11), "f2") == np.float16(1.0)

    # Double rounding (f64 then f32) would pick the wrong side here, but the
    # direct rounding must not:
    trap32 = Fraction(1) + Fraction(1, 1 << 24) + Fraction(1, 1 << 53)
    assert round_fraction(trap32, "f4") == np.float32(1.0) + np.finfo(np.float32).eps

    # Smallest subnormals and their zero ties
    assert round_fraction(Fraction(1, 1 << 1075), "f8") == 0.0
    assert round_fraction(Fraction(3, 1 << 1076), "f8") == np.float64(2.0**-1074)
    assert round_fraction(Fraction(1, 1 << 25), "f2") == np.float16(0.0)
    assert round_fraction(Fraction(3, 1 << 26), "f2") == np.float16(2.0**-24)

    # Overflow
    assert round_fraction(Fraction(1 << 1024), "f8") == np.inf
    assert round_fraction(-Fraction(1 << 1024), "f8") == -np.inf
    assert round_fraction(Fraction(1 << 1023), "f8") == np.float64(2.0**1023)
