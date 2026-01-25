# -*- coding: utf-8 -*-
"""
QuantipyMRX - Google Sheets Export Module

Export market research analysis results to Google Sheets.
Supports service account authentication (for server-side apps like talk2data)
and OAuth authentication (for local development).

Requirements:
    pip install gspread google-auth

For service account authentication:
    1. Create a service account in Google Cloud Console
    2. Download the JSON key file
    3. Share the target spreadsheet with the service account email

For OAuth authentication:
    1. Create OAuth credentials in Google Cloud Console
    2. Download the client secrets JSON file
    3. Run the authentication flow
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
import json

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import gspread
    from gspread.utils import rowcol_to_a1
    from google.oauth2.service_account import Credentials as ServiceCredentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False
    logger.warning("gspread not installed. Google Sheets export unavailable.")

if TYPE_CHECKING:
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult
    from quantipymrx.analysis.grouping import (
        AwarenessSummary, GridSummary, TopOfMindSummary, GroupedFrequency
    )

# Google Sheets API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file',
]

# Formatting colors (Google Sheets RGB format 0-1)
COLORS = {
    'header': {'red': 0.122, 'green': 0.306, 'blue': 0.475},  # Dark blue
    'subheader': {'red': 0.180, 'green': 0.459, 'blue': 0.714},  # Medium blue
    'highlight': {'red': 0.741, 'green': 0.843, 'blue': 0.933},  # Light blue
    'white': {'red': 1, 'green': 1, 'blue': 1},
    'light_gray': {'red': 0.95, 'green': 0.95, 'blue': 0.95},
}


def _ensure_gspread():
    """Raise error if gspread is not available."""
    if not HAS_GSPREAD:
        raise ImportError(
            "gspread and google-auth are required for Google Sheets export. "
            "Install with: pip install gspread google-auth"
        )


# =============================================================================
# Authentication
# =============================================================================

def get_gspread_client(
    credentials_path: Optional[Union[str, Path]] = None,
    credentials_dict: Optional[Dict[str, Any]] = None,
) -> "gspread.Client":
    """
    Get an authenticated gspread client.

    Parameters
    ----------
    credentials_path : str or Path, optional
        Path to service account JSON key file.
    credentials_dict : dict, optional
        Service account credentials as dictionary (for server environments).

    Returns
    -------
    gspread.Client
        Authenticated client.

    Examples
    --------
    # From file
    client = get_gspread_client("path/to/service-account.json")

    # From environment variable (common in cloud deployments)
    import os, json
    creds = json.loads(os.environ['GOOGLE_CREDENTIALS'])
    client = get_gspread_client(credentials_dict=creds)
    """
    _ensure_gspread()

    if credentials_dict:
        creds = ServiceCredentials.from_service_account_info(
            credentials_dict,
            scopes=SCOPES
        )
    elif credentials_path:
        credentials_path = Path(credentials_path)
        creds = ServiceCredentials.from_service_account_file(
            str(credentials_path),
            scopes=SCOPES
        )
    else:
        raise ValueError("Either credentials_path or credentials_dict is required")

    return gspread.authorize(creds)


# =============================================================================
# Google Sheets Exporter
# =============================================================================

class GoogleSheetsExporter:
    """
    Export analysis results to Google Sheets.

    Provides formatted export of:
    - Frequency tables
    - Crosstab tables
    - Awareness summaries
    - Grid/Scale summaries
    - Top of Mind summaries
    """

    def __init__(
        self,
        client: "gspread.Client",
        spreadsheet_name: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
        folder_id: Optional[str] = None,
    ):
        """
        Initialize the Google Sheets exporter.

        Parameters
        ----------
        client : gspread.Client
            Authenticated gspread client.
        spreadsheet_name : str, optional
            Name for a new spreadsheet (will be created).
        spreadsheet_id : str, optional
            ID of existing spreadsheet to use.
        folder_id : str, optional
            Google Drive folder ID for new spreadsheets.
        """
        _ensure_gspread()
        self.client = client
        self._sheet_count = 0

        if spreadsheet_id:
            # Open existing spreadsheet
            self.spreadsheet = client.open_by_key(spreadsheet_id)
            logger.info(f"Opened existing spreadsheet: {self.spreadsheet.title}")
        elif spreadsheet_name:
            # Create new spreadsheet
            self.spreadsheet = client.create(spreadsheet_name, folder_id=folder_id)
            logger.info(f"Created new spreadsheet: {spreadsheet_name}")
            logger.info(f"Spreadsheet URL: {self.spreadsheet.url}")
        else:
            raise ValueError("Either spreadsheet_name or spreadsheet_id required")

    @property
    def url(self) -> str:
        """Get the spreadsheet URL."""
        return self.spreadsheet.url

    @property
    def id(self) -> str:
        """Get the spreadsheet ID."""
        return self.spreadsheet.id

    def share(self, email: str, perm_type: str = 'user', role: str = 'writer'):
        """
        Share the spreadsheet with an email address.

        Parameters
        ----------
        email : str
            Email address to share with.
        perm_type : str
            Permission type: 'user', 'group', 'domain', or 'anyone'.
        role : str
            Role: 'reader', 'writer', or 'owner'.
        """
        self.spreadsheet.share(email, perm_type=perm_type, role=role)
        logger.info(f"Shared spreadsheet with {email} as {role}")

    def add_cover_sheet(
        self,
        dataset_name: str,
        case_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a cover sheet with export information.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        case_count : int
            Number of cases.
        metadata : dict, optional
            Additional metadata to display.
        """
        ws = self._get_or_create_worksheet("Cover")

        # Build cover content
        rows = [
            ["QuantipyMRX Analysis Export"],
            [""],
            ["Dataset:", dataset_name],
            ["Cases:", f"{case_count:,}"],
            ["Export Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["Generated by:", "QuantipyMRX"],
        ]

        if metadata:
            rows.append([""])
            for key, value in metadata.items():
                rows.append([f"{key}:", str(value)])

        ws.update('A1', rows)

        # Format title
        ws.format('A1', {
            'textFormat': {'bold': True, 'fontSize': 16},
        })

        # Format labels
        ws.format('A3:A10', {
            'textFormat': {'bold': True},
        })

    def add_frequency(
        self,
        result: "FrequencyResult",
        sheet_name: Optional[str] = None,
    ):
        """
        Add a frequency table worksheet.

        Parameters
        ----------
        result : FrequencyResult
            The frequency result to export.
        sheet_name : str, optional
            Name for the worksheet.
        """
        sheet_name = sheet_name or self._make_sheet_name(result.variable)
        ws = self._get_or_create_worksheet(sheet_name)

        # Build content
        rows = [
            [result.label or result.variable],
            [f"Base: {result.base:,}", f"Mean: {result.mean:.2f}" if result.mean else ""],
            [""],
            ["Code", "Label", "Count", "Percent"],
        ]

        # Add data rows
        for _, row in result.table.iterrows():
            rows.append([
                row.get('code', ''),
                row.get('label', ''),
                row.get('count', 0),
                f"{row.get('percent', 0):.1f}%",
            ])

        ws.update('A1', rows)

        # Format header row
        self._format_header_row(ws, 4, 4)

        # Format title
        ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})

    def add_crosstab(
        self,
        result: "CrosstabResult",
        sheet_name: Optional[str] = None,
    ):
        """
        Add a crosstab worksheet.

        Parameters
        ----------
        result : CrosstabResult
            The crosstab result to export.
        sheet_name : str, optional
            Name for the worksheet.
        """
        sheet_name = sheet_name or self._make_sheet_name(f"{result.x}_x_{result.y}")
        ws = self._get_or_create_worksheet(sheet_name)

        rows = [
            [f"{result.x_label} by {result.y_label}"],
            [f"Base: {result.total_base:,}",
             f"Chi-square: {result.chi2:.2f}" if result.chi2 else "",
             f"p-value: {result.chi2_pvalue:.4f}" if result.chi2_pvalue else ""],
            [""],
            ["Column Percentages"],
        ]

        # Add column percentages
        if result.col_pct is not None:
            # Header row
            headers = [""] + [str(c) for c in result.col_pct.columns]
            rows.append(headers)

            # Data rows
            for idx, row in result.col_pct.iterrows():
                data_row = [str(idx)] + [f"{v:.1f}%" if pd.notna(v) else "" for v in row]
                rows.append(data_row)

        ws.update('A1', rows)

        # Format
        ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
        ws.format('A4', {'textFormat': {'bold': True}})
        self._format_header_row(ws, 5, len(result.col_pct.columns) + 1)

    def add_awareness(
        self,
        result: "AwarenessSummary",
        sheet_name: Optional[str] = None,
    ):
        """
        Add an awareness summary worksheet.

        Parameters
        ----------
        result : AwarenessSummary
            The awareness summary to export.
        sheet_name : str, optional
            Name for the worksheet.
        """
        sheet_name = sheet_name or self._make_sheet_name(result.group_name)
        ws = self._get_or_create_worksheet(sheet_name)

        rows = [
            [result.display_name[:100] if result.display_name else result.group_name],
            [f"Base: {result.total_base:,}", f"Average Awareness: {result.average_awareness:.1f}%"],
            [""],
            ["Item", "Base", "Aware", "Awareness %"],
        ]

        # Add data rows
        for _, row in result.awareness_table.iterrows():
            rows.append([
                row.get('item_label', ''),
                row.get('base', 0),
                row.get('aware_count', 0),
                f"{row.get('awareness_pct', 0):.1f}%",
            ])

        ws.update('A1', rows)

        # Format
        ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
        self._format_header_row(ws, 4, 4)

    def add_grid_summary(
        self,
        result: "GridSummary",
        sheet_name: Optional[str] = None,
    ):
        """
        Add a grid/scale summary worksheet.

        Parameters
        ----------
        result : GridSummary
            The grid summary to export.
        sheet_name : str, optional
            Name for the worksheet.
        """
        sheet_name = sheet_name or self._make_sheet_name(result.group_name)
        ws = self._get_or_create_worksheet(sheet_name)

        rows = [
            [result.display_name[:100] if result.display_name else result.group_name],
            [f"Scale: {result.scale_min} to {result.scale_max}",
             f"Overall Mean: {result.overall_mean:.2f}" if result.overall_mean else ""],
            [""],
            ["Item", "Base", "Mean", "Std Dev", "Top 2 Box", "Bottom 2 Box"],
        ]

        # Check if NPS column exists
        has_nps = 'nps_score' in result.summary_table.columns
        if has_nps:
            rows[3].append("NPS Score")

        # Add data rows
        for _, row in result.summary_table.iterrows():
            data_row = [
                row.get('item_label', ''),
                row.get('base', 0),
                f"{row.get('mean', 0):.2f}" if pd.notna(row.get('mean')) else "",
                f"{row.get('std_dev', 0):.2f}" if pd.notna(row.get('std_dev')) else "",
                f"{row.get('top_2_box', 0):.1f}%" if pd.notna(row.get('top_2_box')) else "",
                f"{row.get('bottom_2_box', 0):.1f}%" if pd.notna(row.get('bottom_2_box')) else "",
            ]
            if has_nps:
                data_row.append(f"{row.get('nps_score', 0):.1f}" if pd.notna(row.get('nps_score')) else "")
            rows.append(data_row)

        ws.update('A1', rows)

        # Format
        ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
        self._format_header_row(ws, 4, 7 if has_nps else 6)

    def add_top_of_mind(
        self,
        result: "TopOfMindSummary",
        sheet_name: Optional[str] = None,
    ):
        """
        Add a Top of Mind summary worksheet.

        Parameters
        ----------
        result : TopOfMindSummary
            The TOM summary to export.
        sheet_name : str, optional
            Name for the worksheet.
        """
        sheet_name = sheet_name or self._make_sheet_name(result.group_name)
        ws = self._get_or_create_worksheet(sheet_name)

        rows = [
            [result.display_name[:100] if result.display_name else result.group_name],
            [f"Base: {result.total_base:,}", f"Mentioners: {result.total_mentioners:,}"],
            [""],
            ["First Mention (Top of Mind)"],
            ["Label", "Count", "Percent"],
        ]

        # First mention data
        for _, row in result.first_mention.iterrows():
            rows.append([
                row.get('label', ''),
                row.get('count', 0),
                f"{row.get('percent', 0):.1f}%",
            ])

        # Add any mention section
        rows.append([""])
        rows.append(["Any Mention"])
        rows.append(["Label", "Count", "Percent"])

        for _, row in result.any_mention.iterrows():
            rows.append([
                row.get('label', ''),
                row.get('count', 0),
                f"{row.get('percent', 0):.1f}%",
            ])

        ws.update('A1', rows)

        # Format
        ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
        ws.format('A4', {'textFormat': {'bold': True}})
        self._format_header_row(ws, 5, 3)

    def add_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        title: Optional[str] = None,
    ):
        """
        Add a generic DataFrame as a worksheet.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to export.
        sheet_name : str
            Name for the worksheet.
        title : str, optional
            Title to display above the data.
        """
        sheet_name = self._make_sheet_name(sheet_name)
        ws = self._get_or_create_worksheet(sheet_name)

        rows = []
        start_row = 1

        if title:
            rows.append([title])
            rows.append([""])
            start_row = 3

        # Headers
        headers = [""] + list(df.columns) if df.index.name else list(df.columns)
        rows.append(headers)

        # Data
        for idx, row in df.iterrows():
            data_row = [str(idx)] + [str(v) if pd.notna(v) else "" for v in row]
            rows.append(data_row)

        ws.update('A1', rows)

        if title:
            ws.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})
        self._format_header_row(ws, start_row, len(headers))

    def _get_or_create_worksheet(self, title: str) -> "gspread.Worksheet":
        """Get existing worksheet or create new one."""
        title = title[:100]  # Google Sheets max title length

        # Try to get existing
        try:
            return self.spreadsheet.worksheet(title)
        except gspread.WorksheetNotFound:
            pass

        # Create new
        try:
            return self.spreadsheet.add_worksheet(title=title, rows=1000, cols=26)
        except Exception:
            # If default Sheet1 exists and is empty, rename it
            try:
                ws = self.spreadsheet.sheet1
                if ws.title == "Sheet1":
                    ws.update_title(title)
                    return ws
            except Exception:
                pass
            raise

    def _make_sheet_name(self, name: str) -> str:
        """Create a valid worksheet name."""
        self._sheet_count += 1
        name = str(name)[:90]
        # Remove invalid characters
        for char in ['/', '\\', '?', '*', ':', '[', ']']:
            name = name.replace(char, '_')
        return f"{self._sheet_count:02d}_{name}"

    def _format_header_row(self, ws, row_num: int, num_cols: int):
        """Apply header formatting to a row."""
        end_col = chr(ord('A') + num_cols - 1)
        cell_range = f"A{row_num}:{end_col}{row_num}"

        ws.format(cell_range, {
            'backgroundColor': COLORS['header'],
            'textFormat': {
                'bold': True,
                'foregroundColor': COLORS['white'],
            },
            'horizontalAlignment': 'CENTER',
        })


# =============================================================================
# Convenience Functions
# =============================================================================

def export_to_google_sheets(
    grouped_results: Dict[str, Any],
    spreadsheet_name: str,
    credentials_path: Optional[Union[str, Path]] = None,
    credentials_dict: Optional[Dict[str, Any]] = None,
    dataset_name: str = "Survey",
    case_count: int = 0,
    share_with: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
) -> str:
    """
    Export all grouped results to a new Google Sheets spreadsheet.

    Parameters
    ----------
    grouped_results : dict
        Dictionary of group names to result objects.
    spreadsheet_name : str
        Name for the new spreadsheet.
    credentials_path : str or Path, optional
        Path to service account JSON key file.
    credentials_dict : dict, optional
        Service account credentials as dictionary.
    dataset_name : str
        Name of the dataset.
    case_count : int
        Number of cases.
    share_with : list of str, optional
        Email addresses to share the spreadsheet with.
    folder_id : str, optional
        Google Drive folder ID for the spreadsheet.

    Returns
    -------
    str
        URL of the created spreadsheet.
    """
    from quantipymrx.analysis.grouping import (
        AwarenessSummary, GridSummary, TopOfMindSummary, GroupedFrequency
    )

    client = get_gspread_client(credentials_path, credentials_dict)
    exporter = GoogleSheetsExporter(
        client,
        spreadsheet_name=spreadsheet_name,
        folder_id=folder_id,
    )

    # Add cover sheet
    exporter.add_cover_sheet(
        dataset_name=dataset_name,
        case_count=case_count,
        metadata={"Groups": len(grouped_results)}
    )

    # Add result sheets
    for name, result in grouped_results.items():
        try:
            if isinstance(result, AwarenessSummary):
                exporter.add_awareness(result)
            elif isinstance(result, GridSummary):
                exporter.add_grid_summary(result)
            elif isinstance(result, TopOfMindSummary):
                exporter.add_top_of_mind(result)
            elif isinstance(result, GroupedFrequency):
                exporter.add_dataframe(result.frequency_table, name, result.display_name)
        except Exception as e:
            logger.warning(f"Failed to export {name}: {e}")

    # Share if requested
    if share_with:
        for email in share_with:
            try:
                exporter.share(email)
            except Exception as e:
                logger.warning(f"Failed to share with {email}: {e}")

    return exporter.url


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_gspread_client",
    "GoogleSheetsExporter",
    "export_to_google_sheets",
]
