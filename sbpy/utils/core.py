# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy utils module

created on March 12, 2019

"""

__all__ = ['time_input']


from typing import Callable
from astropy.time import Time


def time_input(func: Callable, **kwargs):
    """Decorator that validates time inputs.


    Examples
    --------

    >>> from sbpy.core import time_input
    >>> @time_input(t=Time)
    >>> def myfunction(t):
    >>>     return t.mjd

    >>> from sbpy.core import time_input
    >>> @time_input(t=Time)
    >>> def myfunction(t):
    >>>     return t.mjd


    """
