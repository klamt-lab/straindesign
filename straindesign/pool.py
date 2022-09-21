#!/usr/bin/env python3
#
# Copyright 2022 Max Planck Insitute Magdeburg
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
"""Provide a process pool with enhanced performance on Windows, copied and slightly adapted from cobra."""

from multiprocessing.pool import Pool
from multiprocessing import get_context
import os
import sys
import pickle
from os.path import isfile
from platform import system
from tempfile import mkstemp
from typing import Callable, Optional, Tuple

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
    """Multiprocessing process pool with enhanced Windows compatibility
    
    Initialize a process pool.

    Add a thin layer on top of the `multiprocessing.Pool` that, on Windows, passes
    initialization code to workers via a pickle file rather than directly. This is
    done to avoid a performance issue that exists on Windows. Please, also see the
    discussion [1_].

    References
    ----------
    .. [1] https://github.com/opencobra/cobrapy/issues/997

    """

    def __init__(self,
                 processes: Optional[int] = None,
                 initializer: Optional[Callable] = None,
                 initargs: Tuple = (),
                 maxtasksperchild: Optional[int] = None,
                 context=None):
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
            if hasattr(sys.modules['__main__'], '__spec__'):
                if sys.modules['__main__'].__spec__:
                    spec = sys.modules['__main__'].__spec__
                    sys.modules['__main__'].__spec__ = None
            if hasattr(sys.modules['__main__'], '__file__'):
                if sys.modules['__main__'].__file__:
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
        """Clean up resources when leaving a context"""
        self._clean_up()
        super().__exit__(*args, **kwargs)

    def close(self):
        """Call cleanup function and close"""
        self._clean_up()
        super().close()

    def _clean_up(self):
        """Remove the dump file if it exists"""
        if self._filename is not None and isfile(self._filename):
            os.remove(self._filename)
