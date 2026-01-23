"""QuantipyMRX Export Module."""

from quantipymrx.export.json_export import (
    to_json,
    save_json,
    frequency_to_json,
    crosstab_to_json,
    analysis_to_json,
    dataset_to_json,
    api_response,
    paginated_response,
    QuantipyJSONEncoder,
)

# Conditional imports for optional dependencies
try:
    from quantipymrx.export.excel import (
        export_frequency_to_excel,
        export_crosstab_to_excel,
        export_analysis_to_excel,
        ExcelStyle,
    )
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

try:
    from quantipymrx.export.powerpoint import (
        export_frequency_to_pptx,
        export_crosstab_to_pptx,
        export_analysis_to_pptx,
        PPTXStyle,
    )
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

__all__ = [
    # JSON (always available)
    "to_json",
    "save_json",
    "frequency_to_json",
    "crosstab_to_json",
    "analysis_to_json",
    "dataset_to_json",
    "api_response",
    "paginated_response",
    "QuantipyJSONEncoder",
    # Availability flags
    "EXCEL_AVAILABLE",
    "PPTX_AVAILABLE",
]

# Add Excel exports if available
if EXCEL_AVAILABLE:
    __all__.extend([
        "export_frequency_to_excel",
        "export_crosstab_to_excel",
        "export_analysis_to_excel",
        "ExcelStyle",
    ])

# Add PowerPoint exports if available
if PPTX_AVAILABLE:
    __all__.extend([
        "export_frequency_to_pptx",
        "export_crosstab_to_pptx",
        "export_analysis_to_pptx",
        "PPTXStyle",
    ])
