"""
SciPy Compatibility Layer

Provides compatibility between scipy 0.x/1.x (used by original Quantipy3)
and modern scipy versions.

Key changes handled:
- scipy.stats.stats._ttest_finish was a private function that Quantipy3 used
  directly. This is not available in modern scipy.
- Various internal API changes

Usage:
    # Replace the old import:
    # from scipy.stats.stats import _ttest_finish as get_pval

    # With:
    from quantipymrx._compat.scipy_compat import get_pval
"""

from __future__ import annotations

from typing import Literal, Tuple, Union
import numpy as np
from scipy import stats


def ttest_pvalue(
    t_stat: float,
    df: float,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
) -> float:
    """
    Calculate p-value for t-test statistic.

    This is a public replacement for scipy.stats.stats._ttest_finish
    which was a private function used by original Quantipy3.

    Parameters
    ----------
    t_stat : float
        The calculated t-statistic
    df : float
        Degrees of freedom
    alternative : str, default 'two-sided'
        - 'two-sided': two-tailed test
        - 'less': one-tailed test (left)
        - 'greater': one-tailed test (right)

    Returns
    -------
    float
        The p-value

    Example
    -------
    >>> t_stat = 2.5
    >>> df = 30
    >>> p = ttest_pvalue(t_stat, df, 'two-sided')
    >>> print(f"p-value: {p:.4f}")
    """
    if alternative == 'two-sided':
        # Two-tailed: probability of |t| >= |t_stat|
        return 2 * stats.t.sf(np.abs(t_stat), df)
    elif alternative == 'less':
        # One-tailed left: probability of t <= t_stat
        return stats.t.cdf(t_stat, df)
    elif alternative == 'greater':
        # One-tailed right: probability of t >= t_stat
        return stats.t.sf(t_stat, df)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', "
            f"not '{alternative}'"
        )


def get_pval(t_stat: float, df: float) -> Tuple[float, float]:
    """
    Compatibility function that mimics scipy.stats.stats._ttest_finish.

    The original function returned (t_stat, p_value) as a tuple.
    This maintains that interface for backward compatibility.

    Parameters
    ----------
    t_stat : float
        The calculated t-statistic
    df : float
        Degrees of freedom

    Returns
    -------
    tuple of (float, float)
        (t_statistic, p_value)

    Example
    -------
    # Original Quantipy3 code:
    # from scipy.stats.stats import _ttest_finish as get_pval
    # t, p = get_pval(t_stat, df)

    # Now use:
    from quantipymrx._compat.scipy_compat import get_pval
    t, p = get_pval(t_stat, df)
    """
    p = ttest_pvalue(t_stat, df, 'two-sided')
    return (t_stat, p)


def chi2_pvalue(chi2_stat: float, df: int) -> float:
    """
    Calculate p-value for chi-square statistic.

    Parameters
    ----------
    chi2_stat : float
        The chi-square statistic
    df : int
        Degrees of freedom

    Returns
    -------
    float
        The p-value (survival function value)
    """
    return stats.chi2.sf(chi2_stat, df)


def f_pvalue(f_stat: float, dfn: int, dfd: int) -> float:
    """
    Calculate p-value for F-statistic (ANOVA).

    Parameters
    ----------
    f_stat : float
        The F-statistic
    dfn : int
        Degrees of freedom numerator
    dfd : int
        Degrees of freedom denominator

    Returns
    -------
    float
        The p-value
    """
    return stats.f.sf(f_stat, dfn, dfd)


def z_pvalue(
    z_stat: float,
    alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided',
) -> float:
    """
    Calculate p-value for z-statistic (normal distribution).

    Parameters
    ----------
    z_stat : float
        The z-statistic
    alternative : str, default 'two-sided'
        Type of test

    Returns
    -------
    float
        The p-value
    """
    if alternative == 'two-sided':
        return 2 * stats.norm.sf(np.abs(z_stat))
    elif alternative == 'less':
        return stats.norm.cdf(z_stat)
    elif alternative == 'greater':
        return stats.norm.sf(z_stat)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")


# Convenience imports for common scipy.stats functions
# These haven't changed API but we provide them here for consistency

def ttest_ind(a, b, **kwargs):
    """Independent samples t-test."""
    return stats.ttest_ind(a, b, **kwargs)


def ttest_rel(a, b, **kwargs):
    """Paired samples t-test."""
    return stats.ttest_rel(a, b, **kwargs)


def chi2_contingency(observed, **kwargs):
    """Chi-square test of independence."""
    return stats.chi2_contingency(observed, **kwargs)


def pearsonr(x, y):
    """Pearson correlation coefficient."""
    return stats.pearsonr(x, y)


def spearmanr(x, y, **kwargs):
    """Spearman rank correlation."""
    return stats.spearmanr(x, y, **kwargs)


def f_oneway(*args, **kwargs):
    """One-way ANOVA."""
    return stats.f_oneway(*args, **kwargs)
