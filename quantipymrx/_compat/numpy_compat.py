"""
NumPy Compatibility Layer

Provides compatibility between numpy 1.14.x (used by original Quantipy3)
and numpy 1.24+ / 2.x (modern).

Key changes handled:
- np.object, np.int, np.float, np.bool deprecated in 1.20, removed in 1.24
- np.str deprecated
- Various dtype handling changes

Usage:
    from quantipymrx._compat.numpy_compat import ensure_numpy_compat
    ensure_numpy_compat()  # Call once at startup
"""

from __future__ import annotations

import numpy as np

# Version detection
NUMPY_VERSION = tuple(int(x) for x in np.__version__.split('.')[:2])
NUMPY_1_24_PLUS = NUMPY_VERSION >= (1, 24)
NUMPY_2_PLUS = NUMPY_VERSION >= (2, 0)


def ensure_numpy_compat():
    """
    Ensure numpy compatibility with legacy code.

    Adds back deprecated type aliases that were removed in numpy 1.24:
    - np.object -> object
    - np.int -> int
    - np.float -> float
    - np.bool -> bool
    - np.str -> str
    - np.complex -> complex

    Call this once at module initialization.
    """
    import warnings

    # These were deprecated in numpy 1.20 and removed in 1.24
    # Legacy Quantipy3 code uses them
    # Suppress FutureWarnings during this setup
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)

        # Check using getattr with default to avoid triggering warnings
        if getattr(np, 'object', None) is None:
            np.object = object

        if getattr(np, 'int', None) is None:
            np.int = int

        if getattr(np, 'float', None) is None:
            np.float = float

        if getattr(np, 'bool', None) is None:
            np.bool = bool

        if getattr(np, 'str', None) is None:
            np.str = str

        if getattr(np, 'complex', None) is None:
            np.complex = complex


def is_object_dtype(dtype) -> bool:
    """
    Check if dtype is object type.

    Compatible replacement for checking np.object dtype.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to check

    Returns
    -------
    bool
        True if object dtype

    Example
    -------
    # Instead of:
    # if isinstance(df[col].dtype, np.object):

    # Use:
    if is_object_dtype(df[col].dtype):
        ...
    """
    return dtype == object or str(dtype) == 'object'


def is_numeric_dtype(dtype) -> bool:
    """
    Check if dtype is numeric.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to check

    Returns
    -------
    bool
    """
    return np.issubdtype(dtype, np.number)


def is_integer_dtype(dtype) -> bool:
    """
    Check if dtype is integer.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to check

    Returns
    -------
    bool
    """
    return np.issubdtype(dtype, np.integer)


def is_float_dtype(dtype) -> bool:
    """
    Check if dtype is float.

    Parameters
    ----------
    dtype : numpy dtype
        The dtype to check

    Returns
    -------
    bool
    """
    return np.issubdtype(dtype, np.floating)


def safe_astype(arr, dtype, **kwargs):
    """
    Safely convert array dtype with backward compatibility.

    Parameters
    ----------
    arr : array-like
        Array to convert
    dtype : dtype
        Target dtype
    **kwargs
        Additional arguments to astype

    Returns
    -------
    np.ndarray
    """
    # Handle legacy dtype strings
    dtype_map = {
        'object': object,
        'int': np.int64,
        'float': np.float64,
        'bool': np.bool_,
        'str': str,
    }

    if isinstance(dtype, str) and dtype in dtype_map:
        dtype = dtype_map[dtype]

    return np.asarray(arr).astype(dtype, **kwargs)


# Ensure compatibility on import
ensure_numpy_compat()
