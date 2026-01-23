"""
QuantipyMRX - PowerPoint Export

Professional PowerPoint export with charts, tables,
and market research styling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import pandas as pd

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.chart import XL_CHART_TYPE
    from pptx.chart.data import CategoryChartData
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult
    from quantipymrx.analysis.auto import AutoAnalysisResult


# =============================================================================
# STYLE DEFINITIONS
# =============================================================================

@dataclass
class PPTXStyle:
    """PowerPoint styling configuration."""
    # Colors (RGB tuples)
    primary_color: Tuple[int, int, int] = (31, 78, 121)  # Dark blue
    secondary_color: Tuple[int, int, int] = (46, 117, 182)  # Medium blue
    accent_color: Tuple[int, int, int] = (255, 192, 0)  # Gold
    text_color: Tuple[int, int, int] = (51, 51, 51)  # Dark gray

    # Fonts
    title_font: str = "Calibri"
    title_size: int = 32
    subtitle_size: int = 24
    body_font: str = "Calibri"
    body_size: int = 14

    # Slide dimensions
    slide_width: float = 13.333  # inches (16:9)
    slide_height: float = 7.5


DEFAULT_PPTX_STYLE = PPTXStyle()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _rgb(color: Tuple[int, int, int]) -> RGBColor:
    """Convert RGB tuple to RGBColor."""
    return RGBColor(color[0], color[1], color[2])


def _add_title_slide(prs: "Presentation", title: str, subtitle: str = "", style: PPTXStyle = DEFAULT_PPTX_STYLE):
    """Add a title slide."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(2.5), Inches(12.33), Inches(1)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(style.title_size)
    title_para.font.bold = True
    title_para.font.color.rgb = _rgb(style.primary_color)
    title_para.alignment = PP_ALIGN.CENTER

    # Subtitle
    if subtitle:
        sub_box = slide.shapes.add_textbox(
            Inches(0.5), Inches(3.7), Inches(12.33), Inches(0.5)
        )
        sub_frame = sub_box.text_frame
        sub_para = sub_frame.paragraphs[0]
        sub_para.text = subtitle
        sub_para.font.size = Pt(style.subtitle_size)
        sub_para.font.color.rgb = _rgb(style.text_color)
        sub_para.alignment = PP_ALIGN.CENTER

    return slide


def _add_content_slide(
    prs: "Presentation",
    title: str,
    style: PPTXStyle = DEFAULT_PPTX_STYLE
) -> Tuple:
    """Add a content slide with title, return (slide, content_area)."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.6)
    )
    title_frame = title_box.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.text = title
    title_para.font.size = Pt(style.subtitle_size)
    title_para.font.bold = True
    title_para.font.color.rgb = _rgb(style.primary_color)

    # Content area coordinates
    content_area = {
        'left': Inches(0.5),
        'top': Inches(1.1),
        'width': Inches(12.33),
        'height': Inches(5.9),
    }

    return slide, content_area


# =============================================================================
# TABLE HELPERS
# =============================================================================

def _add_table(
    slide,
    df: pd.DataFrame,
    left: float,
    top: float,
    width: float,
    height: float,
    style: PPTXStyle = DEFAULT_PPTX_STYLE,
    max_rows: int = 15,
):
    """Add a DataFrame as a table to a slide."""
    # Limit rows
    if len(df) > max_rows:
        df = df.head(max_rows)

    rows = len(df) + 1  # +1 for header
    cols = len(df.columns) + 1  # +1 for index

    table = slide.shapes.add_table(rows, cols, left, top, width, height).table

    # Set column widths
    col_width = width // cols
    for i in range(cols):
        table.columns[i].width = col_width

    # Header row styling
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = table.cell(0, col_idx)
        cell.text = str(col_name)
        cell.fill.solid()
        cell.fill.fore_color.rgb = _rgb(style.primary_color)

        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(10)
        para.font.bold = True
        para.font.color.rgb = RGBColor(255, 255, 255)
        para.alignment = PP_ALIGN.CENTER

    # Index header
    cell = table.cell(0, 0)
    cell.text = ""
    cell.fill.solid()
    cell.fill.fore_color.rgb = _rgb(style.primary_color)

    # Data rows
    for row_idx, (idx, row) in enumerate(df.iterrows(), start=1):
        # Index column
        cell = table.cell(row_idx, 0)
        cell.text = str(idx)[:30]  # Truncate long labels
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(9)
        para.font.bold = True

        # Data columns
        for col_idx, col_name in enumerate(df.columns, start=1):
            cell = table.cell(row_idx, col_idx)
            value = row[col_name]

            if pd.isna(value):
                cell.text = "-"
            elif isinstance(value, float):
                cell.text = f"{value:.1f}"
            else:
                cell.text = str(value)

            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(9)
            para.alignment = PP_ALIGN.RIGHT

    return table


# =============================================================================
# CHART HELPERS
# =============================================================================

def _add_bar_chart(
    slide,
    df: pd.DataFrame,
    value_column: str,
    left: float,
    top: float,
    width: float,
    height: float,
    chart_title: str = "",
    horizontal: bool = True,
):
    """Add a bar chart to a slide."""
    chart_data = CategoryChartData()
    chart_data.categories = [str(idx) for idx in df.index]
    chart_data.add_series(value_column, df[value_column].values)

    chart_type = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED

    chart = slide.shapes.add_chart(
        chart_type, left, top, width, height, chart_data
    ).chart

    # Title
    if chart_title:
        chart.has_title = True
        chart.chart_title.text_frame.text = chart_title

    # Format
    chart.has_legend = False

    return chart


# =============================================================================
# FREQUENCY EXPORT
# =============================================================================

def export_frequency_to_pptx(
    result: "FrequencyResult",
    filepath: Union[str, Path],
    include_chart: bool = True,
    style: PPTXStyle = DEFAULT_PPTX_STYLE,
) -> None:
    """
    Export a frequency result to PowerPoint.

    Parameters
    ----------
    result : FrequencyResult
        Frequency result to export.
    filepath : str or Path
        Output file path.
    include_chart : bool
        Include bar chart visualization.
    style : PPTXStyle
        Styling configuration.
    """
    if not PPTX_AVAILABLE:
        raise ImportError("python-pptx is required for PowerPoint export")

    prs = Presentation()
    prs.slide_width = Inches(style.slide_width)
    prs.slide_height = Inches(style.slide_height)

    # Title slide
    _add_title_slide(
        prs,
        title=result.label,
        subtitle=f"Base: {result.base:,}",
        style=style
    )

    # Content slide
    slide, area = _add_content_slide(prs, result.label, style)

    if include_chart:
        # Chart on left, table on right
        chart_width = area['width'] * 0.55
        table_width = area['width'] * 0.4

        # Add chart
        _add_bar_chart(
            slide,
            result.table,
            'Percent',
            area['left'],
            area['top'],
            chart_width,
            area['height'],
            chart_title="",
        )

        # Add table
        _add_table(
            slide,
            result.table[['Count', 'Percent']],
            area['left'] + chart_width + Inches(0.2),
            area['top'],
            table_width,
            area['height'],
            style=style,
        )
    else:
        # Full table
        _add_table(
            slide,
            result.table,
            area['left'],
            area['top'],
            area['width'],
            area['height'],
            style=style,
        )

    prs.save(filepath)


# =============================================================================
# CROSSTAB EXPORT
# =============================================================================

def export_crosstab_to_pptx(
    result: "CrosstabResult",
    filepath: Union[str, Path],
    style: PPTXStyle = DEFAULT_PPTX_STYLE,
) -> None:
    """
    Export a crosstab result to PowerPoint.

    Parameters
    ----------
    result : CrosstabResult
        Crosstab result to export.
    filepath : str or Path
        Output file path.
    style : PPTXStyle
        Styling configuration.
    """
    if not PPTX_AVAILABLE:
        raise ImportError("python-pptx is required for PowerPoint export")

    prs = Presentation()
    prs.slide_width = Inches(style.slide_width)
    prs.slide_height = Inches(style.slide_height)

    # Title slide
    _add_title_slide(
        prs,
        title=f"{result.x_label} by {result.y_label}",
        subtitle=f"Base: {result.total_base:,}",
        style=style
    )

    # Table slide
    slide, area = _add_content_slide(
        prs,
        f"{result.x_label} by {result.y_label} (Column %)",
        style
    )

    _add_table(
        slide,
        result.col_pct,
        area['left'],
        area['top'],
        area['width'],
        area['height'],
        style=style,
    )

    # Add significance note if applicable
    if result.chi2_pvalue is not None:
        sig_text = f"Chi-square p-value: {result.chi2_pvalue:.4f}"
        if result.chi2_pvalue < 0.05:
            sig_text += " (Significant)"

        note_box = slide.shapes.add_textbox(
            area['left'],
            area['top'] + area['height'] + Inches(0.1),
            area['width'],
            Inches(0.3)
        )
        note_para = note_box.text_frame.paragraphs[0]
        note_para.text = sig_text
        note_para.font.size = Pt(10)
        note_para.font.italic = True

    prs.save(filepath)


# =============================================================================
# FULL ANALYSIS EXPORT
# =============================================================================

def export_analysis_to_pptx(
    results: "AutoAnalysisResult",
    filepath: Union[str, Path],
    max_slides: int = 50,
    style: PPTXStyle = DEFAULT_PPTX_STYLE,
) -> None:
    """
    Export complete analysis results to PowerPoint.

    Parameters
    ----------
    results : AutoAnalysisResult
        Analysis results from auto_analyze().
    filepath : str or Path
        Output file path.
    max_slides : int
        Maximum number of slides.
    style : PPTXStyle
        Styling configuration.
    """
    if not PPTX_AVAILABLE:
        raise ImportError("python-pptx is required for PowerPoint export")

    prs = Presentation()
    prs.slide_width = Inches(style.slide_width)
    prs.slide_height = Inches(style.slide_height)

    summary = results.dataset_summary
    slide_count = 0

    # Title slide
    _add_title_slide(
        prs,
        title=summary.get('name', 'Survey Analysis'),
        subtitle=f"{summary.get('n_cases', 0):,} Responses | {summary.get('n_variables', 0)} Variables",
        style=style
    )
    slide_count += 1

    # Summary slide
    slide, area = _add_content_slide(prs, "Executive Summary", style)

    # Add insights as bullet points
    content_box = slide.shapes.add_textbox(
        area['left'], area['top'], area['width'], area['height']
    )
    tf = content_box.text_frame
    tf.word_wrap = True

    for i, insight in enumerate(results.key_insights[:8]):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"• {insight}"
        p.font.size = Pt(14)
        p.space_after = Pt(12)

    slide_count += 1

    # Frequency slides (top 10)
    freq_items = list(results.frequencies.items())[:10]
    for var_name, freq in freq_items:
        if slide_count >= max_slides:
            break

        slide, area = _add_content_slide(prs, freq.label, style)

        # Chart
        _add_bar_chart(
            slide,
            freq.table.head(10),
            'Percent',
            area['left'],
            area['top'],
            area['width'] * 0.6,
            area['height'],
        )

        # Base info
        base_box = slide.shapes.add_textbox(
            area['left'] + area['width'] * 0.65,
            area['top'],
            area['width'] * 0.3,
            Inches(0.5)
        )
        base_para = base_box.text_frame.paragraphs[0]
        base_para.text = f"Base: {freq.base:,}"
        base_para.font.size = Pt(12)

        slide_count += 1

    # Crosstab slides (top 5 per banner)
    for banner_var, xtabs in results.crosstabs.items():
        xtab_items = list(xtabs.items())[:5]

        for x_var, xtab in xtab_items:
            if slide_count >= max_slides:
                break

            slide, area = _add_content_slide(
                prs,
                f"{xtab.x_label} by {xtab.y_label}",
                style
            )

            _add_table(
                slide,
                xtab.col_pct,
                area['left'],
                area['top'],
                area['width'],
                area['height'] * 0.85,
                style=style,
            )

            # Significance indicator
            if xtab.chi2_pvalue is not None and xtab.chi2_pvalue < 0.05:
                sig_box = slide.shapes.add_textbox(
                    area['left'],
                    area['top'] + area['height'] * 0.88,
                    area['width'],
                    Inches(0.3)
                )
                sig_para = sig_box.text_frame.paragraphs[0]
                sig_para.text = f"⚡ Statistically significant (p={xtab.chi2_pvalue:.4f})"
                sig_para.font.size = Pt(10)
                sig_para.font.color.rgb = RGBColor(192, 0, 0)

            slide_count += 1

    prs.save(filepath)
