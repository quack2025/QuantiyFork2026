"""
QuantipyMRX Compatibility Layer

Provides compatibility between legacy Quantipy3 code and modern
pandas 2.x, scipy 1.x, and numpy 1.24+ APIs.

Usage:
    # Option 1: Apply compatibility patches only
    from quantipymrx._compat import ensure_compatibility
    ensure_compatibility()

    # Option 2: Install full legacy shim (allows `import quantipy` to work)
    from quantipymrx._compat import install_legacy_shim
    install_legacy_shim()

    # Import specific utilities
    from quantipymrx._compat.pandas_compat import series_append
    from quantipymrx._compat.scipy_compat import get_pval
"""

from quantipymrx._compat.pandas_compat import PandasBackport
from quantipymrx._compat.scipy_compat import get_pval, ttest_pvalue
from quantipymrx._compat.numpy_compat import ensure_numpy_compat
from quantipymrx._compat.i18n import (
    fix_encoding,
    fix_dataframe_encoding,
    fix_meta_encoding,
    normalize_text,
    detect_encoding,
    get_text_key_info,
    suggest_text_key,
    SUPPORTED_TEXT_KEYS,
)

__all__ = [
    'ensure_compatibility',
    'install_legacy_shim',
    'PandasBackport',
    'get_pval',
    'ttest_pvalue',
    # i18n
    'fix_encoding',
    'fix_dataframe_encoding',
    'fix_meta_encoding',
    'normalize_text',
    'detect_encoding',
    'get_text_key_info',
    'suggest_text_key',
    'SUPPORTED_TEXT_KEYS',
]


def ensure_compatibility():
    """
    Apply all compatibility patches.

    Call this once at module initialization to ensure
    legacy code works with modern dependencies.
    """
    # Pandas patches
    PandasBackport.patch()

    # NumPy patches
    ensure_numpy_compat()


def install_legacy_shim():
    """
    Install the legacy shim that allows `import quantipy` to work.

    This makes legacy code using `import quantipy` work by
    redirecting to `quantipymrx._legacy`.
    """
    from quantipymrx._compat.legacy_shim import install
    install()
