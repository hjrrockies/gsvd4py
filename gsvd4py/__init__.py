"""gsvd4py — Generalized SVD via LAPACK ?ggsvd3."""

from ._gsvd import gsvd, gsvdvals
from ._lapack import lapack_info

__version__ = '0.3.0'

__all__ = ['gsvd', 'gsvdvals', 'lapack_info', '__version__']
