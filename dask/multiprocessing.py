from __future__ import annotations

import atexit
import copyreg
import multiprocessing
import multiprocessing.pool
import os
import pickle
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from warnings import warn

import cloudpickle
import numpy as np

from dask import config
from dask.core import istask
from dask.local import MultiprocessingPoolExecutor, get_async, reraise
from dask.optimization import cull, fuse
from dask.system import CPU_COUNT
from dask.typing import Key
from dask.utils import ensure_dict


def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)


# type(set.union) is used as a proxy to <class 'method_descriptor'>
copyreg.pickle(type(set.union), _reduce_method_descriptor)

_dumps = partial(cloudpickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
_loads = cloudpickle.loads


def _process_get_id():
    return multiprocessing.current_process().ident


# -- Remote Exception Handling --
# By default, tracebacks can't be serialized using pickle. However, the
# `tblib` library can enable support for this. Since we don't mandate
# that tblib is installed, we do the following:
#
# - If tblib is installed, use it to serialize the traceback and reraise
#   in the scheduler process
# - Otherwise, use a ``RemoteException`` class to contain a serialized
#   version of the formatted traceback, which will then print in the
#   scheduler process.
#
# To enable testing of the ``RemoteException`` class even when tblib is
# installed, we don't wrap the class in the try block below
class RemoteException(Exception):
    """Remote Exception

    Contains the exception and traceback from a remotely run task
    """

    def __init__(self, exception, traceback):
        self.exception = exception
        self.traceback = traceback

    def __str__(self):
        return f"{self.exception}\n\nTraceback\n---------\n{self.traceback}"

    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) + dir(self.exception)))

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            return getattr(self.exception, key)


exceptions: dict[type[Exception], type[Exception]] = {}


def remote_exception(exc: Exception, tb) -> Exception:
    """Metaclass that wraps exception type in RemoteException"""
    if type(exc) in exceptions:
        typ = exceptions[type(exc)]
        return typ(exc, tb)
    else:
        try:
            typ = type(
                exc.__class__.__name__,
                (RemoteException, type(exc)),
                {"exception_type": type(exc)},
            )
            exceptions[type(exc)] = typ
            return typ(exc, tb)
        except TypeError:
            return exc


try:
    import tblib.pickling_support

    tblib.pickling_support.install()

    def _pack_traceback(tb):
        return tb

except ImportError:

    def _pack_traceback(tb):
        return "".join(traceback.format_tb(tb))

    def reraise(exc, tb=None):
        exc = remote_exception(exc, tb)
        raise exc


def pack_exception(e, dumps):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = _pack_traceback(exc_traceback)
    try:
        result = dumps((e, tb))
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = _pack_traceback(exc_traceback)
        result = dumps((e, tb))
    return result


_CONTEXT_UNSUPPORTED = """\
The 'multiprocessing.context' configuration option will be ignored on Python 2
and on Windows, because they each only support a single context.
"""


def get_context():
    """Return the current multiprocessing context."""
    # fork context does fork()-without-exec(), which can lead to deadlocks,
    # so default to "spawn".
    context_name = config.get("multiprocessing.context", "spawn")
    if sys.platform == "win32":
        if context_name != "spawn":
            # Only spawn is supported on Win32, can't change it:
            warn(_CONTEXT_UNSUPPORTED, UserWarning)
        return multiprocessing
    else:
        return multiprocessing.get_context(context_name)


# -- Shared read-only input storage ----------------------------------------
#
# When several tasks read the same large input array, pickling the array
# with every task submission copies it to the workers once per task.
# Optionally (see the ``multiprocessing.share-inputs`` config), such inputs
# are instead copied once into a ``SharedMemory`` segment in the scheduler
# process; tasks are sent a small handle and workers attach to the segment
# zero-copy, reconstructing a read-only ``numpy.ndarray``. This preserves the
# numerical semantics of both slicing (views) and inputs nested inside
# tuples/lists/dicts. The scheduler process owns the segments and unlinks
# them only after the worker pool has shut down (on success, task failure,
# or interrupt), so workers never outlive the storage they read.


class _SharedArray:
    """Scheduler-side placeholder for an array stored in shared memory.

    Pickles to a lightweight handle (never the array data); on unpickling in
    a worker process it is reconstructed by :func:`_reconstruct_shared_array`
    as a read-only ``numpy.ndarray`` backed by the shared segment.
    """

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str, nbytes: int):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def __reduce__(self):
        return (
            _reconstruct_shared_array,
            (self.name, self.shape, self.dtype, self.nbytes),
        )


# Per-worker-process cache of attached segments, so that a segment used by
# many tasks in the same worker is attached once rather than per task.
_SHARED_ATTACHMENTS: dict[str, SharedMemory] = {}
_atexit_registered = False


def _reconstruct_shared_array(name, shape, dtype, nbytes):
    """Reattach to a shared input segment in a worker process.

    Returns a read-only ``numpy.ndarray`` view on the shared memory. The
    underlying segment attachment is cached for the lifetime of the worker
    process and released at exit; it is never unlinked here (the scheduler
    process owns unlinking).
    """
    global _atexit_registered
    shm = _SHARED_ATTACHMENTS.get(name)
    if shm is None:
        shm = SharedMemory(name=name, create=False)
        _SHARED_ATTACHMENTS[name] = shm
        if not _atexit_registered:
            atexit.register(_release_shared_attachments)
            _atexit_registered = True
    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    assert arr.nbytes == nbytes
    arr.setflags(write=False)
    return arr


def _release_shared_attachments():
    """Close (but do not unlink) this process's shared segment attachments."""
    for shm in _SHARED_ATTACHMENTS.values():
        try:
            shm.close()
        except (FileNotFoundError, BufferError):
            pass
    _SHARED_ATTACHMENTS.clear()


class _SharedInputManager:
    """Owns the shared memory segments created for one ``get`` call."""

    def __init__(self, min_bytes: int):
        self.min_bytes = min_bytes
        self._segments: dict[str, SharedMemory] = {}
        self._arrays: dict[int, _SharedArray] = {}

    @property
    def names(self) -> list[str]:
        return list(self._segments)

    def share_value(self, value):
        """Return a copy of ``value`` with eligible arrays replaced by handles."""
        if isinstance(value, np.ndarray):
            return self._share_array(value)
        if istask(value):
            return (value[0],) + tuple(self.share_value(a) for a in value[1:])
        if type(value) is list:
            return [self.share_value(v) for v in value]
        if type(value) is tuple:
            return tuple(self.share_value(v) for v in value)
        if type(value) is dict:
            return {k: self.share_value(v) for k, v in value.items()}
        if type(value) is set:
            return {self.share_value(v) for v in value}
        return value

    def _share_array(self, arr: np.ndarray):
        dtype = arr.dtype
        # Object/void arrays cannot live in a flat shared segment, and
        # empty or small arrays are not worth sharing.
        if (
            dtype.hasobject
            or dtype.kind == "V"
            or arr.nbytes == 0
            or arr.nbytes < self.min_bytes
        ):
            return arr
        handle = self._arrays.get(id(arr))
        if handle is not None:
            return handle
        data = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr)
        shm = SharedMemory(create=True, size=data.nbytes)
        try:
            np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)[...] = data
        except BaseException:
            shm.close()
            shm.unlink()
            raise
        self._segments[shm.name] = shm
        handle = _SharedArray(shm.name, data.shape, data.dtype.str, data.nbytes)
        self._arrays[id(arr)] = handle
        return handle

    def close(self) -> None:
        """Close and unlink all segments. Safe to call more than once."""
        for shm in self._segments.values():
            try:
                shm.close()
            except (FileNotFoundError, BufferError):
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        self._segments.clear()
        self._arrays.clear()


def share_graph_inputs(dsk: Mapping, min_bytes: int):
    """Replace shareable array inputs in ``dsk`` with shared-memory handles.

    Returns the rewritten graph together with the :class:`_SharedInputManager`
    that owns the backing segments. ``manager.close()`` must be called once
    all tasks are finished.
    """
    manager = _SharedInputManager(min_bytes)
    try:
        shared = {key: manager.share_value(value) for key, value in dsk.items()}
    except BaseException:
        manager.close()
        raise
    return shared, manager


def get(
    dsk: Mapping,
    keys: Sequence[Key] | Key,
    num_workers=None,
    func_loads=None,
    func_dumps=None,
    optimize_graph=True,
    pool=None,
    initializer=None,
    chunksize=None,
    share_inputs=None,
    **kwargs,
):
    """Multiprocessed get function appropriate for Bags

    Parameters
    ----------
    dsk : dict
        dask graph
    keys : object or list
        Desired results from graph
    num_workers : int
        Number of worker processes (defaults to number of cores)
    func_dumps : function
        Function to use for function serialization (defaults to cloudpickle.dumps)
    func_loads : function
        Function to use for function deserialization (defaults to cloudpickle.loads)
    optimize_graph : bool
        If True [default], `fuse` is applied to the graph before computation.
    pool : Executor or Pool
        Some sort of `Executor` or `Pool` to use
    initializer: function
        Ignored if ``pool`` has been set.
        Function to initialize a worker process before running any tasks in it.
    chunksize: int, optional
        Size of chunks to use when dispatching work.
        Defaults to 6 as some batching is helpful.
        If -1, will be computed to evenly divide ready work across workers.
    share_inputs: bool, optional
        If True, large read-only array inputs in the graph are placed in
        shared memory once instead of being pickled for every task. Defaults
        to the ``multiprocessing.share-inputs`` configuration (``False``).
        Arrays smaller than ``multiprocessing.shared-input-min-bytes`` are
        not shared.
    """
    chunksize = chunksize or config.get("chunksize", 6)
    pool = pool or config.get("pool", None)
    initializer = initializer or config.get("multiprocessing.initializer", None)
    num_workers = num_workers or config.get("num_workers", None) or CPU_COUNT
    if share_inputs is None:
        share_inputs = config.get("multiprocessing.share-inputs", False)
    # Re-running failed tasks locally would pass the shared-array handles
    # (rather than real arrays) to the task, so sharing is incompatible with
    # ``rerun_exceptions_locally``.
    if share_inputs and config.get("rerun_exceptions_locally", False):
        share_inputs = False
    if pool is None:
        # In order to get consistent hashing in subprocesses, we need to set a
        # consistent seed for the Python hash algorithm. Unfortunately, there
        # is no way to specify environment variables only for the Pool
        # processes, so we have to rely on environment variables being
        # inherited.
        if os.environ.get("PYTHONHASHSEED") in (None, "0"):
            # This number is arbitrary; it was chosen to commemorate
            # https://github.com/dask/dask/issues/6640.
            os.environ["PYTHONHASHSEED"] = "6640"
        context = get_context()
        initializer = partial(initialize_worker_process, user_initializer=initializer)
        pool = ProcessPoolExecutor(
            num_workers, mp_context=context, initializer=initializer
        )
        cleanup = True
    else:
        if initializer is not None:
            warn(
                "The ``initializer`` argument is ignored when ``pool`` is provided. "
                "The user should configure ``pool`` with the needed ``initializer`` "
                "on creation."
            )
        if isinstance(pool, multiprocessing.pool.Pool):
            pool = MultiprocessingPoolExecutor(pool)
        cleanup = False

    if hasattr(dsk, "__dask_graph__"):
        dsk = dsk.__dask_graph__()

    dsk = ensure_dict(dsk)
    dsk2, dependencies = cull(dsk, keys)
    if optimize_graph:
        dsk3, dependencies = fuse(dsk2, keys, dependencies)
    else:
        dsk3 = dsk2

    # We specify marshalling functions in order to catch serialization
    # errors and report them to the user.
    loads = func_loads or config.get("func_loads", None) or _loads
    dumps = func_dumps or config.get("func_dumps", None) or _dumps

    shared_inputs = None

    # Note former versions used a multiprocessing Manager to share
    # a Queue between parent and workers, but this is fragile on Windows
    # (issue #1652).
    try:
        if share_inputs:
            min_bytes = config.get("multiprocessing.shared-input-min-bytes", 1_048_576)
            dsk3, shared_inputs = share_graph_inputs(dsk3, min_bytes)
        # Run
        result = get_async(
            pool.submit,
            pool._max_workers,
            dsk3,
            keys,
            get_id=_process_get_id,
            dumps=dumps,
            loads=loads,
            pack_exception=pack_exception,
            raise_exception=reraise,
            chunksize=chunksize,
            **kwargs,
        )
    finally:
        # Shut the workers down before unlinking the segments so that no
        # task is still reading storage that has been released. This also
        # runs when a task failed or the user interrupted, after which the
        # segments are reclaimed and subsequent computations are unaffected.
        try:
            if cleanup:
                pool.shutdown()
        finally:
            if shared_inputs is not None:
                shared_inputs.close()
    return result


def default_initializer():
    # If Numpy is already imported, presumably its random state was
    # inherited from the parent => re-seed it.
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed()


def initialize_worker_process(user_initializer=None):
    """
    Initialize a worker process before running any tasks in it.
    """
    default_initializer()

    if user_initializer is not None:
        user_initializer()
