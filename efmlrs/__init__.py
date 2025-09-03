# -*- coding: utf-8 -*-
from __future__ import print_function
from ._version import get_versions

__author__ = 'Bianca Buchner'
__email__ = 'bianca.buchner@gmail.com'
__version__ = get_versions()['version']
del get_versions

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
