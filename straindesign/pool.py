"""Provide a process pool with enhanced performance on Windows.
Copied and slightly changed from cobra."""

from multiprocessing.pool import Pool
import os
import pickle
from os.path import isfile
from platform import system
from tempfile import mkstemp
from types import TracebackType
from typing import Any, Callable, Optional, Tuple, Type

RUN = 0
CLOSE = 1
TERMINATE = 2

# __all__ = ("Pool",)

def _init_win_worker(filename: str) -> None:
    """Retrieve worker initialization code from a pickle file and call it."""
    with open(filename, mode="rb") as handle:
        func, *args = pickle.load(handle)
    func(*args)


class Pool(Pool):
    """Define a process pool that handles the Windows platform specially."""

    def __init__(
        self,
        processes: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        maxtasksperchild: Optional[int] = None,
        context=None):
        """
        Initialize a process pool.

        Add a thin layer on top of the `multiprocessing.Pool` that, on Windows, passes
        initialization code to workers via a pickle file rather than directly. This is
        done to avoid a performance issue that exists on Windows. Please, also see the
        discussion [1_].

        References
        ----------
        .. [1] https://github.com/opencobra/cobrapy/issues/997

        """
        self._filename = None
        if initializer is not None and system() == "Windows":
            descriptor, self._filename = mkstemp(suffix=".pkl")
            # We use the file descriptor to the open file returned by `mkstemp` to
            # ensure that the resource is closed and can later be removed. Otherwise
            # Windows will cause a `PermissionError`.
            with os.fdopen(descriptor, mode="wb") as handle:
                pickle.dump((initializer,) + initargs, handle)
            initializer = _init_win_worker
            initargs = (self._filename,)
        super().__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=context,
        )

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Clean up resources when leaving a context."""
        result = super().__exit__(exc_type, exc_val, exc_tb)
        self._clean_up()
        return result

    def close(self) -> None:
        """
        Close the process pool.

        Prevent any more tasks from being submitted to the pool. Once all the tasks have
        been completed, the worker processes will exit.

        """
        try:
            super().close()
        finally:
            self._clean_up()

    def _clean_up(self) -> None:
        """Remove the dump file if it exists."""
        if self._filename is not None and isfile(self._filename):
            os.remove(self._filename)
