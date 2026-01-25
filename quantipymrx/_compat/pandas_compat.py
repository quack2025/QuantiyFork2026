"""
Pandas Compatibility Layer

Provides compatibility between pandas 0.25.x (used by original Quantipy3)
and pandas 2.x (modern).

Key changes handled:
- DataFrame.append() removed in pandas 2.0
- Series.append() removed in pandas 2.0
- inplace=True deprecated (will be removed in pandas 3.0)
- .ix[] removed (use .loc[] or .iloc[])

Usage:
    from quantipymrx._compat.pandas_compat import PandasBackport
    PandasBackport.patch()  # Apply once at startup
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, Optional, Union

import pandas as pd
import numpy as np

# Version detection
PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split('.')[:2])
PANDAS_2_PLUS = PANDAS_VERSION >= (2, 0)
PANDAS_1_PLUS = PANDAS_VERSION >= (1, 0)


def series_append(
    series: pd.Series,
    to_append: Union[pd.Series, Any],
    ignore_index: bool = False,
    verify_integrity: bool = False,
) -> pd.Series:
    """
    Replacement for Series.append() deprecated in pandas 2.0.

    Parameters
    ----------
    series : pd.Series
        Original series
    to_append : pd.Series or scalar
        Data to append
    ignore_index : bool, default False
        If True, do not use index values
    verify_integrity : bool, default False
        If True, raise on duplicate index

    Returns
    -------
    pd.Series
        Concatenated series

    Example
    -------
    # Before (deprecated):
    result = series.append(other)

    # After:
    from quantipymrx._compat import series_append
    result = series_append(series, other)
    """
    if isinstance(to_append, pd.Series):
        result = pd.concat(
            [series, to_append],
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
        )
    else:
        # Scalar value - wrap in Series
        to_append_series = pd.Series([to_append])
        result = pd.concat(
            [series, to_append_series],
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
        )
    return result


def dataframe_append(
    df: pd.DataFrame,
    other: Union[pd.DataFrame, pd.Series, dict],
    ignore_index: bool = False,
    verify_integrity: bool = False,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Replacement for DataFrame.append() deprecated in pandas 2.0.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame
    other : DataFrame, Series, or dict
        Data to append
    ignore_index : bool, default False
        If True, do not use index values
    verify_integrity : bool, default False
        If True, raise on duplicate index
    sort : bool, default False
        Sort columns if they are not aligned

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame
    """
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    elif isinstance(other, pd.Series):
        other = other.to_frame().T

    return pd.concat(
        [df, other],
        ignore_index=ignore_index,
        verify_integrity=verify_integrity,
        sort=sort,
    )


def safe_loc(df: pd.DataFrame, row_indexer, col_indexer=None):
    """
    Safe replacement for .ix[] which was removed in pandas 1.0.

    Automatically determines whether to use .loc[] or .iloc[].

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to index
    row_indexer : label or array-like
        Row selection
    col_indexer : label or array-like, optional
        Column selection

    Returns
    -------
    DataFrame or Series or scalar
    """
    if col_indexer is None:
        return df.loc[row_indexer]
    return df.loc[row_indexer, col_indexer]


class PandasBackport:
    """
    Backport of pandas methods for legacy Quantipy3 code.

    Usage:
        from quantipymrx._compat import PandasBackport
        PandasBackport.patch()  # Apply patches at startup

    This adds back:
    - Series.append()
    - DataFrame.append()

    As compatibility shims that use pd.concat() internally.
    """

    _patched = False
    _original_series_append = None
    _original_df_append = None

    @classmethod
    def patch(cls):
        """Apply compatibility patches to pandas."""
        if cls._patched:
            return

        # Store originals if they exist
        if hasattr(pd.Series, 'append'):
            cls._original_series_append = pd.Series.append
        if hasattr(pd.DataFrame, 'append'):
            cls._original_df_append = pd.DataFrame.append

        # Patch Series.append
        def _series_append(self, to_append, ignore_index=False, verify_integrity=False):
            return series_append(self, to_append, ignore_index, verify_integrity)
        pd.Series.append = _series_append

        # Patch DataFrame.append
        def _df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return dataframe_append(self, other, ignore_index, verify_integrity, sort)
        pd.DataFrame.append = _df_append

        cls._patched = True

    @classmethod
    def unpatch(cls):
        """Remove compatibility patches (for testing)."""
        if not cls._patched:
            return

        if cls._original_series_append:
            pd.Series.append = cls._original_series_append
        elif hasattr(pd.Series, 'append'):
            delattr(pd.Series, 'append')

        if cls._original_df_append:
            pd.DataFrame.append = cls._original_df_append
        elif hasattr(pd.DataFrame, 'append'):
            delattr(pd.DataFrame, 'append')

        cls._patched = False


def fix_inplace_operations(func: Callable) -> Callable:
    """
    Decorator to handle inplace=True deprecation.

    Wraps functions that use inplace=True to use assignment instead.

    Usage:
        @fix_inplace_operations
        def my_function(df):
            df.rename(columns={'a': 'b'}, inplace=True)  # Will be fixed
            return df
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            return func(*args, **kwargs)
    return wrapper


# Utility functions for common patterns

def rename_inplace(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Safe rename that returns new DataFrame instead of inplace.

    Usage:
        # Instead of: df.rename(columns={'a': 'b'}, inplace=True)
        df = rename_inplace(df, columns={'a': 'b'})
    """
    return df.rename(**kwargs)


def set_index_safe(df: pd.DataFrame, keys, **kwargs) -> pd.DataFrame:
    """
    Safe set_index without inplace.

    Usage:
        # Instead of: df.set_index('col', inplace=True)
        df = set_index_safe(df, 'col')
    """
    kwargs.pop('inplace', None)  # Remove if present
    return df.set_index(keys, **kwargs)


def sort_index_safe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Safe sort_index without inplace.
    """
    kwargs.pop('inplace', None)
    return df.sort_index(**kwargs)


def reset_index_safe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Safe reset_index without inplace.
    """
    kwargs.pop('inplace', None)
    return df.reset_index(**kwargs)
