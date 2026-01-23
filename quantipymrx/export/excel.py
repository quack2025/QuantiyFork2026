"""
QuantipyMRX - Excel Export

Professional Excel export with formatting, multiple sheets,
and market research styling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import pandas as pd

try:
    from openpyxl import Workbook
    from openpyxl.styles import (
        Font, Fill, PatternFill, Border, Side, Alignment,
        NamedStyle, numbers
    )
    from openpyxl.utils.dataframe import dataframe_to_rows
    from openpyxl.utils import get_column_letter
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult
    from quantipymrx.analysis.auto import AutoAnalysisResult


# =============================================================================
# STYLE DEFINITIONS
# =============================================================================

@dataclass
class ExcelStyle:
    """Excel styling configuration."""
    # Colors
    header_bg: str = "1F4E79"  # Dark blue
    header_font: str = "FFFFFF"  # White
    subheader_bg: str = "2E75B6"  # Medium blue
    alt_row_bg: str = "D6DCE4"  # Light gray
    highlight_bg: str = "FFC000"  # Yellow/Gold
    significance_bg: str = "C6EFCE"  # Light green

    # Fonts
    header_font_size: int = 11
    body_font_size: int = 10
    font_name: str = "Calibri"

    # Borders
    border_color: str = "B4B4B4"


DEFAULT_STYLE = ExcelStyle()


def _create_styles(wb: "Workbook", style: ExcelStyle = DEFAULT_STYLE):
    """Create named styles for the workbook."""
    # Header style
    header_style = NamedStyle(name="header")
    header_style.font = Font(
        name=style.font_name,
        size=style.header_font_size,
        bold=True,
        color=style.header_font
    )
    header_style.fill = PatternFill(
        start_color=style.header_bg,
        end_color=style.header_bg,
        fill_type="solid"
    )
    header_style.alignment = Alignment(horizontal="center", vertical="center")

    # Subheader style
    subheader_style = NamedStyle(name="subheader")
    subheader_style.font = Font(
        name=style.font_name,
        size=style.body_font_size,
        bold=True
    )
    subheader_style.fill = PatternFill(
        start_color=style.subheader_bg,
        end_color=style.subheader_bg,
        fill_type="solid"
    )
    subheader_style.font = Font(color="FFFFFF", bold=True)

    # Percentage style
    pct_style = NamedStyle(name="percentage")
    pct_style.number_format = '0.0%'
    pct_style.alignment = Alignment(horizontal="right")

    # Number style
    num_style = NamedStyle(name="number")
    num_style.number_format = '#,##0'
    num_style.alignment = Alignment(horizontal="right")

    # Register styles
    for s in [header_style, subheader_style, pct_style, num_style]:
        try:
            wb.add_named_style(s)
        except ValueError:
            pass  # Style already exists


def _apply_header_style(ws, row: int, style: ExcelStyle = DEFAULT_STYLE):
    """Apply header styling to a row."""
    header_fill = PatternFill(
        start_color=style.header_bg,
        end_color=style.header_bg,
        fill_type="solid"
    )
    header_font = Font(
        name=style.font_name,
        size=style.header_font_size,
        bold=True,
        color=style.header_font
    )

    for cell in ws[row]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _auto_column_width(ws, min_width: int = 10, max_width: int = 50):
    """Auto-adjust column widths."""
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)

        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass

        adjusted_width = min(max(max_length + 2, min_width), max_width)
        ws.column_dimensions[column_letter].width = adjusted_width


# =============================================================================
# FREQUENCY EXPORT
# =============================================================================

def export_frequency_to_excel(
    result: "FrequencyResult",
    filepath: Union[str, Path],
    style: ExcelStyle = DEFAULT_STYLE,
) -> None:
    """
    Export a frequency result to Excel.

    Parameters
    ----------
    result : FrequencyResult
        Frequency result to export.
    filepath : str or Path
        Output file path.
    style : ExcelStyle
        Styling configuration.
    """
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is required for Excel export")

    wb = Workbook()
    ws = wb.active
    ws.title = "Frequency"

    _create_styles(wb, style)

    # Title row
    ws['A1'] = result.variable
    ws['A1'].font = Font(size=14, bold=True)
    ws['A2'] = result.label
    ws['A2'].font = Font(size=12, italic=True)

    # Base info
    ws['A4'] = f"Base: {result.base:,}"
    if result.weighted_base:
        ws['B4'] = f"(Weighted: {result.weighted_base:,.1f})"

    # Table headers
    start_row = 6
    headers = list(result.table.columns)
    ws.cell(row=start_row, column=1, value="Response")

    for col_idx, header in enumerate(headers, start=2):
        ws.cell(row=start_row, column=col_idx, value=header)

    _apply_header_style(ws, start_row, style)

    # Data rows
    for row_idx, (idx, row) in enumerate(result.table.iterrows(), start=start_row + 1):
        ws.cell(row=row_idx, column=1, value=str(idx))

        for col_idx, header in enumerate(headers, start=2):
            value = row[header]
            cell = ws.cell(row=row_idx, column=col_idx, value=value)

            # Format percentages
            if 'Percent' in header or '%' in header:
                cell.number_format = '0.0'

    # Statistics
    stat_row = start_row + len(result.table) + 2
    if result.mean is not None:
        ws.cell(row=stat_row, column=1, value="Mean")
        ws.cell(row=stat_row, column=2, value=result.mean)
        ws.cell(row=stat_row, column=2).number_format = '0.00'

    if result.std is not None:
        ws.cell(row=stat_row + 1, column=1, value="Std Dev")
        ws.cell(row=stat_row + 1, column=2, value=result.std)
        ws.cell(row=stat_row + 1, column=2).number_format = '0.00'

    _auto_column_width(ws)
    wb.save(filepath)


# =============================================================================
# CROSSTAB EXPORT
# =============================================================================

def export_crosstab_to_excel(
    result: "CrosstabResult",
    filepath: Union[str, Path],
    include_counts: bool = True,
    include_significance: bool = True,
    style: ExcelStyle = DEFAULT_STYLE,
) -> None:
    """
    Export a crosstab result to Excel.

    Parameters
    ----------
    result : CrosstabResult
        Crosstab result to export.
    filepath : str or Path
        Output file path.
    include_counts : bool
        Include counts sheet.
    include_significance : bool
        Include significance letters.
    style : ExcelStyle
        Styling configuration.
    """
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is required for Excel export")

    wb = Workbook()
    _create_styles(wb, style)

    # Column % sheet (main view)
    ws_pct = wb.active
    ws_pct.title = "Column %"
    _write_crosstab_sheet(ws_pct, result, result.col_pct, "Column Percentages",
                          include_significance, style)

    # Counts sheet
    if include_counts:
        ws_counts = wb.create_sheet("Counts")
        _write_crosstab_sheet(ws_counts, result, result.counts, "Counts",
                             False, style)

    # Statistics sheet
    ws_stats = wb.create_sheet("Statistics")
    _write_stats_sheet(ws_stats, result, style)

    wb.save(filepath)


def _write_crosstab_sheet(
    ws,
    result: "CrosstabResult",
    data: pd.DataFrame,
    title: str,
    include_sig: bool,
    style: ExcelStyle,
):
    """Write crosstab data to a worksheet."""
    # Title
    ws['A1'] = f"{result.x_label} by {result.y_label}"
    ws['A1'].font = Font(size=14, bold=True)
    ws['A2'] = title
    ws['A2'].font = Font(size=12, italic=True)

    # Base row
    ws['A4'] = "Base"
    for col_idx, col in enumerate(data.columns, start=2):
        base_val = result.base.get(col, 0) if isinstance(result.base, dict) else (
            result.base[col] if col in result.base.index else 0
        )
        ws.cell(row=4, column=col_idx, value=base_val)

    # Headers
    start_row = 6
    ws.cell(row=start_row, column=1, value=result.x_label)
    for col_idx, col in enumerate(data.columns, start=2):
        ws.cell(row=start_row, column=col_idx, value=str(col))

    _apply_header_style(ws, start_row, style)

    # Data rows
    for row_idx, (idx, row) in enumerate(data.iterrows(), start=start_row + 1):
        ws.cell(row=row_idx, column=1, value=str(idx))

        for col_idx, col in enumerate(data.columns, start=2):
            value = row[col]
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.number_format = '0.0'

            # Add significance letters
            if include_sig and result.significance is not None:
                try:
                    sig_letter = result.significance.loc[idx, col]
                    if sig_letter:
                        # Add as comment or append to cell
                        cell.value = f"{value:.1f} {sig_letter}"
                except:
                    pass

    _auto_column_width(ws)


def _write_stats_sheet(ws, result: "CrosstabResult", style: ExcelStyle):
    """Write statistics to a worksheet."""
    ws['A1'] = "Statistical Summary"
    ws['A1'].font = Font(size=14, bold=True)

    stats = [
        ("Row Variable", result.x),
        ("Row Label", result.x_label),
        ("Column Variable", result.y),
        ("Column Label", result.y_label),
        ("Total Base", result.total_base),
        ("Weighted", "Yes" if result.is_weighted else "No"),
        ("Chi-Square", result.chi2),
        ("P-Value", result.chi2_pvalue),
        ("Significant", "Yes" if result.chi2_pvalue and result.chi2_pvalue < 0.05 else "No"),
    ]

    for row_idx, (label, value) in enumerate(stats, start=3):
        ws.cell(row=row_idx, column=1, value=label)
        ws.cell(row=row_idx, column=2, value=value)

    _auto_column_width(ws)


# =============================================================================
# BATCH EXPORT
# =============================================================================

def export_analysis_to_excel(
    results: "AutoAnalysisResult",
    filepath: Union[str, Path],
    style: ExcelStyle = DEFAULT_STYLE,
) -> None:
    """
    Export complete analysis results to Excel.

    Parameters
    ----------
    results : AutoAnalysisResult
        Analysis results from auto_analyze().
    filepath : str or Path
        Output file path.
    style : ExcelStyle
        Styling configuration.
    """
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl is required for Excel export")

    wb = Workbook()
    _create_styles(wb, style)

    # Summary sheet
    ws_summary = wb.active
    ws_summary.title = "Summary"
    _write_summary_sheet(ws_summary, results, style)

    # Frequencies sheet
    if results.frequencies:
        ws_freq = wb.create_sheet("Frequencies")
        _write_frequencies_sheet(ws_freq, results.frequencies, style)

    # Crosstabs sheets (one per banner variable)
    for banner_var, xtabs in results.crosstabs.items():
        if xtabs:
            ws_xtab = wb.create_sheet(f"By {banner_var[:20]}")
            _write_crosstabs_sheet(ws_xtab, xtabs, banner_var, style)

    wb.save(filepath)


def _write_summary_sheet(ws, results: "AutoAnalysisResult", style: ExcelStyle):
    """Write summary sheet."""
    ws['A1'] = "Analysis Summary"
    ws['A1'].font = Font(size=16, bold=True)

    summary = results.dataset_summary
    row = 3

    ws.cell(row=row, column=1, value="Dataset")
    ws.cell(row=row, column=2, value=summary.get('name', 'Unknown'))
    row += 1

    ws.cell(row=row, column=1, value="Total Responses")
    ws.cell(row=row, column=2, value=summary.get('n_cases', 0))
    row += 1

    ws.cell(row=row, column=1, value="Total Variables")
    ws.cell(row=row, column=2, value=summary.get('n_variables', 0))
    row += 2

    # Variable types
    ws.cell(row=row, column=1, value="Variable Types")
    ws.cell(row=row, column=1).font = Font(bold=True)
    row += 1

    for var_type, count in summary.get('variable_types', {}).items():
        ws.cell(row=row, column=1, value=f"  {var_type}")
        ws.cell(row=row, column=2, value=count)
        row += 1

    row += 1

    # Key Insights
    ws.cell(row=row, column=1, value="Key Insights")
    ws.cell(row=row, column=1).font = Font(bold=True)
    row += 1

    for insight in results.key_insights:
        ws.cell(row=row, column=1, value=f"• {insight}")
        row += 1

    _auto_column_width(ws)


def _write_frequencies_sheet(
    ws,
    frequencies: Dict[str, "FrequencyResult"],
    style: ExcelStyle,
):
    """Write all frequencies to a single sheet."""
    ws['A1'] = "Frequency Tables"
    ws['A1'].font = Font(size=14, bold=True)

    row = 3
    for var_name, freq in frequencies.items():
        # Variable header
        ws.cell(row=row, column=1, value=freq.variable)
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=2, value=freq.label)
        ws.cell(row=row, column=3, value=f"Base: {freq.base}")
        row += 1

        # Table headers
        headers = ['Response', 'Count', '%']
        for col_idx, header in enumerate(headers, start=1):
            ws.cell(row=row, column=col_idx, value=header)
            ws.cell(row=row, column=col_idx).font = Font(bold=True)
        row += 1

        # Data
        for idx, data_row in freq.table.iterrows():
            ws.cell(row=row, column=1, value=str(idx))
            ws.cell(row=row, column=2, value=data_row.get('Count', 0))
            ws.cell(row=row, column=3, value=data_row.get('Percent', 0))
            ws.cell(row=row, column=3).number_format = '0.0'
            row += 1

        row += 2  # Space between tables

    _auto_column_width(ws)


def _write_crosstabs_sheet(
    ws,
    crosstabs: Dict[str, "CrosstabResult"],
    banner_var: str,
    style: ExcelStyle,
):
    """Write crosstabs for one banner to a sheet."""
    ws['A1'] = f"Cross-tabulations by {banner_var}"
    ws['A1'].font = Font(size=14, bold=True)

    row = 3
    for x_var, xtab in crosstabs.items():
        # Header
        ws.cell(row=row, column=1, value=f"{xtab.x_label}")
        ws.cell(row=row, column=1).font = Font(bold=True)
        row += 1

        # Column headers
        ws.cell(row=row, column=1, value="")
        for col_idx, col in enumerate(xtab.col_pct.columns, start=2):
            ws.cell(row=row, column=col_idx, value=str(col))
            ws.cell(row=row, column=col_idx).font = Font(bold=True)
        row += 1

        # Base row
        ws.cell(row=row, column=1, value="Base")
        for col_idx, col in enumerate(xtab.col_pct.columns, start=2):
            try:
                base_val = xtab.base[col] if col in xtab.base.index else 0
                ws.cell(row=row, column=col_idx, value=int(base_val))
            except:
                pass
        row += 1

        # Data rows
        for idx, data_row in xtab.col_pct.iterrows():
            ws.cell(row=row, column=1, value=str(idx))
            for col_idx, col in enumerate(xtab.col_pct.columns, start=2):
                value = data_row[col]
                cell = ws.cell(row=row, column=col_idx, value=value)
                cell.number_format = '0.0'
            row += 1

        # Significance
        if xtab.chi2_pvalue is not None:
            ws.cell(row=row, column=1, value=f"Chi-sq p-value: {xtab.chi2_pvalue:.4f}")
            if xtab.chi2_pvalue < 0.05:
                ws.cell(row=row, column=1).font = Font(bold=True, color="FF0000")
        row += 2

    _auto_column_width(ws)
