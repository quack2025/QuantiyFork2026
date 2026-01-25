"""
QuantipyMRX - SPSS Reader/Writer

Modern SPSS file handling using pyreadstat.
Extracts complete metadata including labels, missing values, and variable types.
Includes i18n support for proper character encoding in multiple languages.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import pyreadstat
except ImportError:
    raise ImportError(
        "pyreadstat is required for SPSS support. "
        "Install with: pip install pyreadstat"
    )

from quantipymrx.core.metadata import (
    DataSetMeta,
    DataSetInfo,
    VariableMeta,
    VariableType,
    ValueLabel,
    ArrayMeta,
)

# Import i18n utilities for encoding fix
try:
    from quantipymrx._compat.i18n import fix_encoding
    HAS_I18N = True
except ImportError:
    HAS_I18N = False
    def fix_encoding(text):
        return text


def read_spss(
    filepath: Union[str, Path],
    text_key: str = "en-GB",
    apply_value_formats: bool = False,
    dates_as_strings: bool = False,
    convert_categoricals: bool = True,
    encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, DataSetMeta]:
    """
    Read an SPSS .sav file and return data + metadata.

    Parameters
    ----------
    filepath : str or Path
        Path to the .sav file.
    text_key : str, default "en-GB"
        Language key for storing labels.
    apply_value_formats : bool, default False
        If True, applies value labels to data (converts to categorical).
    dates_as_strings : bool, default False
        If True, date variables are kept as strings.
    convert_categoricals : bool, default True
        If True, converts single-choice vars to pandas Categorical.
    encoding : str, optional
        File encoding. If None, pyreadstat auto-detects.

    Returns
    -------
    tuple[pd.DataFrame, DataSetMeta]
        Data and metadata objects.

    Example
    -------
    >>> data, meta = read_spss("survey.sav")
    >>> print(meta.info.case_count)
    1000
    >>> print(meta.single_variables)
    ['q1', 'q2', 'gender', 'age_group']
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SPSS file not found: {filepath}")

    logger.info(f"Reading SPSS file: {filepath}")

    # Read with pyreadstat
    read_kwargs: Dict[str, Any] = {
        "apply_value_formats": apply_value_formats,
        "dates_as_pandas_datetime": not dates_as_strings,
    }
    if encoding:
        read_kwargs["encoding"] = encoding

    df, spss_meta = pyreadstat.read_sav(str(filepath), **read_kwargs)
    logger.debug(f"Read {len(df)} rows, {len(df.columns)} columns")

    # Build our metadata structure
    meta = _build_metadata(df, spss_meta, text_key, filepath.name)

    # Post-process data
    df = _process_dataframe(df, meta, spss_meta, convert_categoricals)

    logger.info(f"SPSS file loaded: {meta.info.case_count} cases, {len(meta.columns)} variables")

    return df, meta


def _build_metadata(
    df: pd.DataFrame,
    spss_meta: Any,
    text_key: str,
    filename: str,
) -> DataSetMeta:
    """Build DataSetMeta from pyreadstat metadata."""

    # Dataset info
    info = DataSetInfo(
        name=filename.replace(".sav", ""),
        source_file=filename,
        source_format="spss",
        case_count=len(df),
    )

    # Build column metadata
    columns: Dict[str, VariableMeta] = {}

    for idx, col_name in enumerate(spss_meta.column_names):
        var_meta = _build_variable_meta(
            col_name=col_name,
            col_index=idx,
            df=df,
            spss_meta=spss_meta,
            text_key=text_key,
        )
        columns[col_name] = var_meta

    # Detect arrays/grids from multiple response sets
    masks = _detect_arrays(spss_meta, columns, text_key)

    # Build sets (variable groups)
    sets = _build_sets(spss_meta, columns)

    return DataSetMeta(
        info=info,
        columns=columns,
        masks=masks,
        sets=sets,
    )


def _build_variable_meta(
    col_name: str,
    col_index: int,
    df: pd.DataFrame,
    spss_meta: Any,
    text_key: str,
) -> VariableMeta:
    """Build metadata for a single variable."""

    # Get variable label and fix encoding
    label = ""
    if col_index < len(spss_meta.column_labels):
        label = spss_meta.column_labels[col_index] or ""
        if HAS_I18N and label:
            label = fix_encoding(label)

    # Determine variable type and get value labels
    var_type, values = _determine_type_and_values(
        col_name, df, spss_meta, text_key
    )

    # Get missing values
    missing_values = _get_missing_values(col_name, spss_meta)

    return VariableMeta(
        name=col_name,
        type=var_type,
        text={text_key: label} if label else {},
        values=values,
        missing_values=missing_values,
    )


def _determine_type_and_values(
    col_name: str,
    df: pd.DataFrame,
    spss_meta: Any,
    text_key: str,
) -> Tuple[VariableType, Optional[List[ValueLabel]]]:
    """Determine variable type and extract value labels."""

    values: Optional[List[ValueLabel]] = None

    # Check if variable has value labels -> categorical
    if col_name in spss_meta.variable_value_labels:
        value_labels = spss_meta.variable_value_labels[col_name]
        values = []
        for code, label in sorted(value_labels.items()):
            # Fix encoding for value labels (important for i18n support)
            label_text = str(label)
            if HAS_I18N:
                label_text = fix_encoding(label_text)
            values.append(ValueLabel(value=int(code), text={text_key: label_text}))

        # Check if it's part of a multiple response set -> delimited set
        if _is_multi_response(col_name, spss_meta, df):
            return VariableType.DELIMITED_SET, values

        return VariableType.SINGLE, values

    # Check original variable types for format info
    if col_name in spss_meta.original_variable_types:
        fmt = spss_meta.original_variable_types[col_name]

        if "DATETIME" in fmt or "DATE" in fmt or "TIME" in fmt:
            return VariableType.DATE, None
        elif fmt.startswith("A"):  # String format
            return VariableType.STRING, None
        elif "." in fmt:  # Has decimal places
            return VariableType.FLOAT, None
        else:
            return VariableType.INT, None

    # Infer from pandas dtype
    dtype = df[col_name].dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return VariableType.DATE, None
    elif pd.api.types.is_float_dtype(dtype):
        # Check if actually integers stored as float
        col_data = df[col_name].dropna()
        if len(col_data) > 0 and (col_data % 1 == 0).all():
            return VariableType.INT, None
        return VariableType.FLOAT, None
    elif pd.api.types.is_integer_dtype(dtype):
        return VariableType.INT, None
    elif pd.api.types.is_object_dtype(dtype):
        return VariableType.STRING, None

    return VariableType.STRING, None


def _is_multi_response(col_name: str, spss_meta: Any, df: pd.DataFrame) -> bool:
    """
    Check if variable is a multiple response (delimited set) variable.

    Detection methods:
    1. Check pyreadstat's multiple response sets metadata
    2. Detect by data format (values like "1;2;3;")
    3. Detect by variable name pattern (common MR suffixes)

    Parameters
    ----------
    col_name : str
        Variable name.
    spss_meta : Any
        pyreadstat metadata object.
    df : pd.DataFrame
        Data frame for content inspection.

    Returns
    -------
    bool
        True if variable appears to be multi-response.
    """
    # Method 1: Check pyreadstat's multiple response sets
    # pyreadstat >= 1.2.0 may have multiple_response_sets attribute
    if hasattr(spss_meta, 'multiple_response_sets'):
        mr_sets = spss_meta.multiple_response_sets or {}
        for set_name, set_info in mr_sets.items():
            if col_name in set_info.get('variables', []):
                logger.debug(f"'{col_name}' is part of MR set '{set_name}'")
                return True

    # Method 2: Check if data contains delimited values (e.g., "1;2;3;")
    if col_name in df.columns:
        col_data = df[col_name].dropna()
        if len(col_data) > 0:
            # Sample first 100 non-null values
            sample = col_data.head(100)

            # Check for common delimited set patterns
            delimited_pattern = re.compile(r'^\d+[;,|\s]\d+')  # e.g., "1;2" or "1,2"
            quantipy_pattern = re.compile(r'^\d+;$|^\d+;\d+')  # e.g., "1;" or "1;2;3"

            matches = 0
            for val in sample:
                val_str = str(val)
                if delimited_pattern.match(val_str) or quantipy_pattern.match(val_str):
                    matches += 1

            # If >10% of values match delimited pattern, likely multi-response
            if matches > len(sample) * 0.1:
                logger.debug(f"'{col_name}' detected as multi-response by data pattern")
                return True

    # Method 3: Check for common MR naming patterns
    # (usually set at the mask/array level, not individual variables)
    mr_patterns = [
        r'_mr$',           # ends with _mr
        r'_multi$',        # ends with _multi
        r'^mr_',           # starts with mr_
        r'_multiresp$',    # ends with _multiresp
    ]
    for pattern in mr_patterns:
        if re.search(pattern, col_name, re.IGNORECASE):
            logger.debug(f"'{col_name}' detected as multi-response by name pattern")
            return True

    return False


def _get_missing_values(col_name: str, spss_meta: Any) -> Optional[List[int]]:
    """Extract missing value codes for a variable."""
    if not hasattr(spss_meta, 'missing_ranges'):
        return None

    if col_name in spss_meta.missing_ranges:
        missing = spss_meta.missing_ranges[col_name]
        if missing:
            # Convert to list of int codes
            codes = []
            for item in missing:
                if isinstance(item, dict):
                    # Range: {'lo': x, 'hi': y}
                    if 'lo' in item and 'hi' in item:
                        codes.extend(range(int(item['lo']), int(item['hi']) + 1))
                else:
                    codes.append(int(item))
            return codes if codes else None
    return None


def _detect_arrays(
    spss_meta: Any,
    columns: Dict[str, VariableMeta],
    text_key: str,
) -> Dict[str, ArrayMeta]:
    """Detect array/grid questions from multiple response definitions."""
    masks: Dict[str, ArrayMeta] = {}

    # Check for multiple response sets
    if hasattr(spss_meta, 'variable_to_label'):
        # Try to detect grids by naming patterns
        # Common patterns: q1_1, q1_2, q1_3 or q1a, q1b, q1c
        pass  # Will be implemented with smarter detection

    return masks


def _build_sets(
    spss_meta: Any,
    columns: Dict[str, VariableMeta],
) -> Dict[str, List[str]]:
    """Build variable sets/groups."""
    sets: Dict[str, List[str]] = {
        "data file": list(columns.keys()),
    }
    return sets


def _process_dataframe(
    df: pd.DataFrame,
    meta: DataSetMeta,
    spss_meta: Any,
    convert_categoricals: bool,
) -> pd.DataFrame:
    """Post-process dataframe after reading."""
    df = df.copy()

    if convert_categoricals:
        for col_name, var_meta in meta.columns.items():
            if var_meta.type == VariableType.SINGLE and var_meta.values:
                # Convert to categorical with proper ordering
                categories = [v.value for v in var_meta.values]
                if col_name in df.columns:
                    # Handle NaN values
                    df[col_name] = pd.Categorical(
                        df[col_name],
                        categories=categories,
                        ordered=False
                    )

    return df


def write_spss(
    filepath: Union[str, Path],
    data: pd.DataFrame,
    meta: DataSetMeta,
    text_key: str = "en-GB",
) -> None:
    """
    Write data and metadata to an SPSS .sav file.

    Parameters
    ----------
    filepath : str or Path
        Output path for the .sav file.
    data : pd.DataFrame
        Data to write.
    meta : DataSetMeta
        Metadata with variable labels and value labels.
    text_key : str, default "en-GB"
        Language key to use for labels.

    Example
    -------
    >>> write_spss("output.sav", data, meta)
    """
    filepath = Path(filepath)

    # Prepare column labels
    column_labels = []
    for col in data.columns:
        if col in meta.columns:
            column_labels.append(meta.columns[col].get_label(text_key))
        else:
            column_labels.append("")

    # Prepare value labels
    variable_value_labels: Dict[str, Dict[int, str]] = {}
    for col_name, var_meta in meta.columns.items():
        if var_meta.values and col_name in data.columns:
            variable_value_labels[col_name] = {
                v.value: v.get_text(text_key)
                for v in var_meta.values
            }

    # Write with pyreadstat
    pyreadstat.write_sav(
        df=data,
        dst_path=str(filepath),
        column_labels=column_labels,
        variable_value_labels=variable_value_labels,
    )


def get_spss_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get quick info about an SPSS file without reading all data.

    Parameters
    ----------
    filepath : str or Path
        Path to the .sav file.

    Returns
    -------
    dict
        Dictionary with file information.

    Example
    -------
    >>> info = get_spss_info("survey.sav")
    >>> print(info['n_rows'], info['n_columns'])
    1000 50
    """
    filepath = Path(filepath)

    # Read metadata only
    _, spss_meta = pyreadstat.read_sav(
        str(filepath),
        metadataonly=True,
    )

    return {
        "n_rows": spss_meta.number_rows,
        "n_columns": spss_meta.number_columns,
        "column_names": spss_meta.column_names,
        "column_labels": dict(zip(
            spss_meta.column_names,
            spss_meta.column_labels
        )),
        "has_value_labels": bool(spss_meta.variable_value_labels),
        "variables_with_labels": list(spss_meta.variable_value_labels.keys()),
    }
