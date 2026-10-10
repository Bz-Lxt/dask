from __future__ import annotations

import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from operator import add

import pytest

import dask
from dask import compute, delayed
from dask.multiprocessing import (
    _dumps,
    _loads,
    _SharedArray,
    get,
    get_context,
    remote_exception,
    share_graph_inputs,
)
from dask.system import CPU_COUNT
from dask.utils_test import inc


def unrelated_function_global(a):
    np = pytest.importorskip("numpy")
    return np.array([a])


def my_small_function_global(a, b):
    return a + b


def test_pickle_globals():
    """Unrelated globals should not be included in serialized bytes"""
    b = _dumps(my_small_function_global)
    assert b"my_small_function_global" in b
    assert b"unrelated_function_global" not in b
    assert b"numpy" not in b


def test_pickle_locals():
    """Unrelated locals should not be included in serialized bytes"""
    np = pytest.importorskip("numpy")

    def unrelated_function_local(a):
        return np.array([a])

    def my_small_function_local(a, b):
        return a + b

    b = _dumps(my_small_function_local)
    assert b"my_small_function_global" not in b
    assert b"my_small_function_local" in b
    assert b"unrelated_function_local" not in b


@pytest.mark.skipif(pickle.HIGHEST_PROTOCOL < 5, reason="requires pickle protocol 5")
def test_out_of_band_pickling():
    """Test that out-of-band pickling works"""
    np = pytest.importorskip("numpy")
    pytest.importorskip("cloudpickle", minversion="1.3.0")

    a = np.arange(5)

    l = []
    b = _dumps(a, buffer_callback=l.append)
    assert len(l) == 1
    assert isinstance(l[0], pickle.PickleBuffer)
    assert memoryview(l[0]) == memoryview(a)

    a2 = _loads(b, buffers=l)
    assert np.all(a == a2)


def bad():
    raise ValueError("12345")


def test_errors_propagate():
    dsk = {"x": (bad,)}

    with pytest.raises(ValueError) as e:
        get(dsk, "x")
    assert "12345" in str(e.value)


def test_remote_exception():
    e = TypeError("hello")
    a = remote_exception(e, "traceback-body")
    b = remote_exception(e, "traceback-body")

    assert type(a) == type(b)
    assert isinstance(a, TypeError)
    assert "hello" in str(a)
    assert "Traceback" in str(a)
    assert "traceback-body" in str(a)


def test_lambda_with_cloudpickle():
    dsk = {"x": 2, "y": (lambda x: x + 1, "x")}
    assert get(dsk, "y") == 3


def lambda_result():
    return lambda x: x + 1


def test_lambda_results_with_cloudpickle():
    dsk = {"x": (lambda_result,)}
    f = get(dsk, "x")
    assert f(2) == 3


class NotUnpickleable:
    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        raise ValueError("Can't unpickle me")


def test_unpicklable_args_generate_errors():
    a = NotUnpickleable()

    dsk = {"x": (bool, a)}

    with pytest.raises(ValueError):
        get(dsk, "x")

    dsk = {"x": (bool, "a"), "a": a}

    with pytest.raises(ValueError):
        get(dsk, "x")


@pytest.mark.parametrize("pool_typ", [multiprocessing.Pool, ProcessPoolExecutor])
def test_reuse_pool(pool_typ):
    with pool_typ(CPU_COUNT) as pool:
        with dask.config.set(pool=pool):
            assert get({"x": (inc, 1)}, "x") == 2
            assert get({"x": (inc, 1)}, "x") == 2


def test_dumps_loads():
    with dask.config.set(func_dumps=pickle.dumps, func_loads=pickle.loads):
        assert get({"x": 1, "y": (add, "x", 2)}, "y") == 3


def test_fuse_doesnt_clobber_intermediates():
    d = {"x": 1, "y": (inc, "x"), "z": (add, 10, "y")}
    assert get(d, ["y", "z"]) == (2, 12)


def test_optimize_graph_false():
    from dask.callbacks import Callback

    d = {"x": 1, "y": (inc, "x"), "z": (add, 10, "y")}
    keys = []
    with Callback(pretask=lambda key, *args: keys.append(key)):
        get(d, "z", optimize_graph=False)
    assert len(keys) == 2


def test_works_with_highlevel_graph():
    """Previously `dask.multiprocessing.get` would accidentally forward
    `HighLevelGraph` graphs through the dask optimization/scheduling routines,
    resulting in odd errors. One way to trigger this was to have a
    non-indexable object in a task. This is just a smoketest to ensure that
    things work properly even if `HighLevelGraph` objects get passed to
    `dask.multiprocessing.get`. See https://github.com/dask/dask/issues/7190.
    """

    class NoIndex:
        def __init__(self, x):
            self.x = x

        def __getitem__(self, key):
            raise Exception("Oh no!")

    x = delayed(lambda x: x)(NoIndex(1))
    (res,) = get(x.dask, x.__dask_keys__())
    assert isinstance(res, NoIndex)
    assert res.x == 1


@pytest.mark.parametrize("random", ["numpy", "random"])
def test_random_seeds(random):
    if random == "numpy":
        np = pytest.importorskip("numpy")
        random = np.random
    else:
        import random

    @delayed(pure=False)
    def f():
        return tuple(random.randint(0, 10000) for i in range(5))

    N = 10
    with dask.config.set(scheduler="processes"):
        (results,) = compute([f() for _ in range(N)])

    assert len(set(results)) == N


class global_:
    value = 0


def proc_init():
    global_.value = 1


@pytest.mark.parametrize(
    "scheduler, initializer, expected_results",
    [
        ("threading", None, [1] * 10),
        ("processes", None, [0] * 10),
        ("processes", proc_init, [1] * 10),
    ],
)
def test_process_initializer(scheduler, initializer, expected_results):
    @delayed(pure=False)
    def f():
        return global_.value

    global_.value = 1

    with dask.config.set(
        {"scheduler": scheduler, "multiprocessing.initializer": initializer}
    ):
        (results,) = compute([f() for _ in range(10)])
    assert results == expected_results

    (results2,) = compute(
        [f() for _ in range(10)],
        scheduler=scheduler,
        initializer=initializer,
    )
    assert results2 == expected_results


def check_for_pytest():
    """We check for spawn by ensuring subprocess doesn't have modules only
    parent process should have:
    """
    import sys

    return "FAKE_MODULE_FOR_TEST" in sys.modules


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows doesn't support different contexts"
)
def test_custom_context_used_python3_posix():
    """The 'multiprocessing.context' config is used to create the pool.

    We assume default is 'spawn', and therefore test for 'fork'.
    """
    # We check for 'fork' by ensuring subprocess doesn't have modules only
    # parent process should have:

    def check_for_pytest():
        import sys

        return "FAKE_MODULE_FOR_TEST" in sys.modules

    import sys

    sys.modules["FAKE_MODULE_FOR_TEST"] = 1
    try:
        with dask.config.set({"multiprocessing.context": "fork"}):
            result = get({"x": (check_for_pytest,)}, "x")
        assert result
    finally:
        del sys.modules["FAKE_MODULE_FOR_TEST"]


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows doesn't support different contexts"
)
def test_get_context_using_python3_posix():
    """get_context() respects configuration.

    If default context is changed this test will need to change too.
    """
    assert get_context() is multiprocessing.get_context("spawn")
    with dask.config.set({"multiprocessing.context": "forkserver"}):
        assert get_context() is multiprocessing.get_context("forkserver")
    with dask.config.set({"multiprocessing.context": "fork"}):
        assert get_context() is multiprocessing.get_context("fork")


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX supports different contexts")
def test_custom_context_ignored_elsewhere():
    """On Windows, setting 'multiprocessing.context' doesn't explode.

    Presumption is it's not used since it's unsupported, but mostly we care about
    not breaking anything.
    """
    assert get({"x": (inc, 1)}, "x") == 2
    with pytest.warns(UserWarning):
        with dask.config.set({"multiprocessing.context": "forkserver"}):
            assert get({"x": (inc, 1)}, "x") == 2


@pytest.mark.skipif(sys.platform != "win32", reason="POSIX supports different contexts")
def test_get_context_always_default():
    """On Python 2/Windows, get_context() always returns same context."""
    assert get_context() is multiprocessing
    with pytest.warns(UserWarning):
        with dask.config.set({"multiprocessing.context": "forkserver"}):
            assert get_context() is multiprocessing


# -- Shared read-only input storage ---------------------------------------

np = pytest.importorskip("numpy")


def _big_array():
    return np.arange(500_000, dtype=np.float64)  # 4 MB


def _sum_all(x):
    return float(x.sum())


def _sum_strided(x):
    return float(x[::2].sum())


def _sum_nested(payload):
    lst, tup, dct = payload
    return float(lst.sum() + tup[0][1::3].sum() + dct["k"].sum())


def _mutate_input(x):
    x[0] = -1
    return float(x.sum())


def _bad_with_arg(x):
    raise ValueError("12345")


def _raise_keyboard_interrupt(x):
    raise KeyboardInterrupt


def _count_attachments(x):
    from dask import multiprocessing as dmp

    return float(x.sum()), len(dmp._SHARED_ATTACHMENTS)


_recorded_payload_sizes: list[int] = []


def _recording_dumps(obj):
    payload = _dumps(obj)
    _recorded_payload_sizes.append(len(payload))
    return payload


def test_share_graph_inputs_selection():
    big = _big_array()
    small = np.arange(3, dtype=np.float64)
    obj = np.empty(2, dtype=object)

    dsk = {
        "big": big,
        "small": small,
        "obj": obj,
        "task": (_sum_all, big),
        "nested": [big, {"k": (_sum_all, big)}],
    }
    shared, manager = share_graph_inputs(dsk, min_bytes=1_048_576)
    try:
        assert isinstance(shared["big"], _SharedArray)
        # Arrays below the threshold and object arrays are left untouched
        assert shared["small"] is small
        assert shared["obj"] is obj
        # Handles are threaded into tasks and nested containers
        assert isinstance(shared["task"][1], _SharedArray)
        assert isinstance(shared["nested"][0], _SharedArray)
        assert isinstance(shared["nested"][1]["k"][1], _SharedArray)
        # The same array is shared into a single segment
        assert len(manager.names) == 1
    finally:
        manager.close()


def test_shared_inputs_values():
    big = _big_array()
    # Non-contiguous (strided) view of a 2-D array
    strided = big.reshape(500, 1000)[::2]

    dsk = {
        "big": big,
        "strided": strided,
        "full": (_sum_all, "big"),
        "sliced": (_sum_strided, "strided"),
        # Literal arrays nested in list/tuple/dict arguments
        "nested": (_sum_nested, [big, (strided,), {"k": big}]),
    }
    results = get(
        dsk, ["full", "sliced", "nested"], share_inputs=True, num_workers=2
    )
    expected = (
        float(big.sum()),
        float(strided[::2].sum()),
        float(big.sum() + strided[1::3].sum() + big.sum()),
    )
    assert results == expected


def test_shared_inputs_single_transfer():
    big = _big_array()
    dsk = {
        "big": big,
        "r1": (_sum_all, "big"),
        "r2": (_sum_all, "big"),
    }

    _recorded_payload_sizes.clear()
    results = get(
        dsk,
        ["r1", "r2"],
        share_inputs=True,
        num_workers=2,
        chunksize=1,
        func_dumps=_recording_dumps,
    )
    assert results == (float(big.sum()),) * 2
    # The 4 MB array must never travel with a task payload
    assert max(_recorded_payload_sizes) < big.nbytes // 10

    _recorded_payload_sizes.clear()
    get(
        dsk,
        ["r1", "r2"],
        share_inputs=False,
        num_workers=2,
        chunksize=1,
        func_dumps=_recording_dumps,
    )
    assert max(_recorded_payload_sizes) >= big.nbytes


def test_shared_inputs_readonly():
    big = _big_array()
    dsk = {"big": big, "r": (_mutate_input, "big")}
    with pytest.raises(ValueError, match="read-only"):
        get(dsk, "r", share_inputs=True, num_workers=1)


def test_shared_inputs_config():
    big = _big_array()
    dsk = {"big": big, "r": (_sum_all, "big")}
    with dask.config.set({"multiprocessing.share-inputs": True}):
        assert get(dsk, "r", num_workers=1) == float(big.sum())


def test_shared_inputs_cleanup_after_failure():
    big = _big_array()
    dsk = {"big": big, "r": (_bad_with_arg, "big")}
    shared, manager = share_graph_inputs(dsk, min_bytes=1_048_576)
    name = manager.names[0]
    try:
        with pytest.raises(ValueError, match="12345"):
            get(shared, "r", num_workers=1)
    finally:
        manager.close()
        manager.close()  # closing twice must not raise
    # The segment was unlinked and cannot be reattached
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, create=False)


def test_shared_inputs_cleanup_after_interrupt():
    big = _big_array()
    dsk = {"big": big, "r": (_raise_keyboard_interrupt, "big")}
    shared, manager = share_graph_inputs(dsk, min_bytes=1_048_576)
    name = manager.names[0]
    try:
        with pytest.raises(KeyboardInterrupt):
            get(shared, "r", num_workers=1)
    finally:
        manager.close()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, create=False)


def test_shared_inputs_followup_compute_unaffected():
    big = _big_array()
    dsk = {"big": big, "r": (_bad_with_arg, "big")}
    with pytest.raises(ValueError, match="12345"):
        get(dsk, "r", share_inputs=True, num_workers=1)

    # A later successful computation is unaffected by the failed one
    other = _big_array() + 1
    dsk2 = {"other": other, "r": (_sum_all, "other")}
    assert get(dsk2, "r", share_inputs=True, num_workers=1) == float(other.sum())


def test_shared_inputs_attachment_cached_per_worker():
    big = _big_array()
    keys = [f"r{i}" for i in range(8)]
    dsk = {"big": big, **{key: (_count_attachments, "big") for key in keys}}
    results = get(dsk, keys, share_inputs=True, num_workers=1, chunksize=1)
    assert [value for value, _ in results] == [float(big.sum())] * 8
    # One segment attachment per worker, no matter how many tasks read it
    assert all(n_attachments == 1 for _, n_attachments in results)
