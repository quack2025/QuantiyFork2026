"""QuantipyMRX I/O Module.

Provides data import/export functionality:
- SPSS (.sav) files
- Excel (.xlsx) with professional formatting
- Google Sheets (via API)
"""

from quantipymrx.io.spss import (
    read_spss,
    write_spss,
    get_spss_info,
)

# Excel export (requires openpyxl)
try:
    from quantipymrx.io.excel_export import (
        ExcelExporter,
        export_frequency_to_excel,
        export_crosstab_to_excel,
        export_processing_results,
    )
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

# Google Sheets export (requires gspread)
try:
    from quantipymrx.io.gsheets_export import (
        GoogleSheetsExporter,
        get_gspread_client,
        export_to_google_sheets,
    )
    HAS_GSHEETS = True
except ImportError:
    HAS_GSHEETS = False

__all__ = [
    # SPSS
    "read_spss",
    "write_spss",
    "get_spss_info",
    # Excel
    "ExcelExporter",
    "export_frequency_to_excel",
    "export_crosstab_to_excel",
    "export_processing_results",
    # Google Sheets
    "GoogleSheetsExporter",
    "get_gspread_client",
    "export_to_google_sheets",
    # Feature flags
    "HAS_EXCEL",
    "HAS_GSHEETS",
]
