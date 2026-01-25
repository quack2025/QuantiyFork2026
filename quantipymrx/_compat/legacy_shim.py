"""
Legacy Shim Module

This module creates a compatibility shim that allows legacy code
using `import quantipy` to work with the new structure.

It registers `quantipy` as an alias to `quantipymrx._legacy` in sys.modules.

Usage:
    # Import this ONCE at application startup
    from quantipymrx._compat import legacy_shim
    legacy_shim.install()

    # Now legacy code works:
    import quantipy  # -> actually imports quantipymrx._legacy
"""

import sys
import importlib
from types import ModuleType


_installed = False


def install():
    """
    Install the legacy shim.

    This makes `import quantipy` work by redirecting to `quantipymrx._legacy`.
    Should be called once at application startup.
    """
    global _installed

    if _installed:
        return

    # Apply compatibility patches first
    from quantipymrx._compat import ensure_compatibility
    ensure_compatibility()

    # Create module alias
    # When someone does `import quantipy`, they get `quantipymrx._legacy`
    try:
        import quantipymrx._legacy as legacy_module
        sys.modules['quantipy'] = legacy_module

        # Also create sub-module aliases for common patterns
        _create_submodule_aliases(legacy_module)

        _installed = True

    except ImportError as e:
        raise ImportError(
            f"Failed to install legacy shim. "
            f"Make sure quantipymrx._legacy is properly set up. "
            f"Original error: {e}"
        )


def _create_submodule_aliases(legacy_module):
    """Create aliases for common sub-module imports."""

    submodules = [
        'core',
        'core.dataset',
        'core.view',
        'core.link',
        'core.chain',
        'core.stack',
        'core.cluster',
        'core.batch',
        'core.options',
        'core.cache',
        'core.rules',
        'core.quantify',
        'core.quantify.engine',
        'core.weights',
        'core.weights.rim',
        'core.weights.weight_engine',
        'core.tools',
        'core.tools.dp',
        'core.tools.dp.io',
        'core.tools.dp.prep',
        'core.tools.dp.query',
        'core.tools.dp.spss',
        'core.tools.dp.spss.reader',
        'core.tools.dp.spss.writer',
        'core.tools.view',
        'core.tools.view.agg',
        'core.tools.view.logic',
        'core.tools.view.struct',
        'core.tools.view.query',
        'core.helpers',
        'core.helpers.functions',
        'core.helpers.constants',
        'core.view_generators',
        'core.view_generators.view_mapper',
        'core.view_generators.view_maps',
        'core.view_generators.view_specs',
        'core.builds',
        'core.builds.excel',
        'core.builds.excel.excel_painter',
        'core.builds.powerpoint',
        'core.builds.powerpoint.pptx_painter',
    ]

    for submod in submodules:
        full_name = f'quantipy.{submod}'
        legacy_name = f'quantipymrx._legacy.{submod}'

        try:
            # Import the legacy submodule
            module = importlib.import_module(legacy_name)
            # Register it under the quantipy name
            sys.modules[full_name] = module
        except ImportError:
            # Submodule might not exist, skip silently
            pass


def uninstall():
    """
    Remove the legacy shim.

    Useful for testing.
    """
    global _installed

    if not _installed:
        return

    # Remove quantipy from sys.modules
    modules_to_remove = [k for k in sys.modules if k.startswith('quantipy')]
    for mod in modules_to_remove:
        del sys.modules[mod]

    _installed = False


def is_installed() -> bool:
    """Check if the shim is installed."""
    return _installed
