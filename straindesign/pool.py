"""Provide a process pool with enhanced performance on Windows.

Copied and slightly changed from cobra.
"""

from multiprocessing.pool import Pool
from multiprocessing import get_context
import os
import sys
import pickle
from os.path import isfile
from platform import system
from tempfile import mkstemp
from typing import Callable, Optional, Tuple, Type

RUN = 0
CLOSE = 1
TERMINATE = 2

# __all__ = ("Pool",)


def _init_win_worker(filename: str) -> None:
    """Retrieve worker initialization code from a pickle file and call it."""
    with open(filename, mode="rb") as handle:
        func, *args = pickle.load(handle)
    func(*args)


class SDPool(Pool):
    """Define a process pool that handles the Windows platform specially."""

    def __init__(self,
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
        # Store and remove main.spec and main.file. Multiprocessing reads
        # these parameters to identify a python file for initialization.
        # The idea is to avoid that the workers call the main file.
        spec = None
        file = None
        if context is None:
            context = get_context('spawn')  # If not declared otherwise,
            # 'spawn' new threads. Experience has shown
            # that forking is unreliable.
            if sys.modules['__main__'].__spec__ is not None:
                spec = sys.modules['__main__'].__spec__
                sys.modules['__main__'].__spec__ = None
            if sys.modules['__main__'].__file__ is not None:
                file = sys.modules['__main__'].__file__
                sys.modules['__main__'].__file__ = None
        super().__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
            context=context,
        )
        # Restore backups
        if spec:
            sys.modules['__main__'].__spec__ = spec
        if file:
            sys.modules['__main__'].__file__ = file

    def __exit__(self, *args, **kwargs):
        """Clean up resources when leaving a context."""
        self._clean_up()
        super().__exit__(*args, **kwargs)

    def close(self):
        self._clean_up()
        super().close()

    def _clean_up(self):
        """Remove the dump file if it exists."""
        if self._filename is not None and isfile(self._filename):
            os.remove(self._filename)
