"""
QuantipyMRX - Frequency Analysis

Calculate frequency distributions with counts, percentages, and statistics.
Supports weighted analysis and multiple variable types.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet

logger = logging.getLogger(__name__)

# Supported delimited set separators
DELIMITED_SEPARATORS = [';', ',', '|', ' ']
DELIMITED_PATTERN = re.compile(r'[;,|\s]+')


@dataclass
class FrequencyResult:
    """
    Result of a frequency analysis.

    Attributes
    ----------
    variable : str
        Name of the analyzed variable.
    label : str
        Variable label.
    table : pd.DataFrame
        Frequency table with counts and percentages.
    base : int
        Total base (unweighted count).
    weighted_base : float
        Weighted base if weights applied.
    mean : float, optional
        Mean for numeric variables.
    std : float, optional
        Standard deviation for numeric variables.
    is_weighted : bool
        Whether analysis was weighted.
    """
    variable: str
    label: str
    table: pd.DataFrame
    base: int
    weighted_base: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    is_weighted: bool = False
    text_key: str = "en-GB"
    _value_labels: Dict[int, str] = field(default_factory=dict)

    @property
    def counts(self) -> pd.Series:
        """Get count column."""
        return self.table['Count']

    @property
    def percentages(self) -> pd.Series:
        """Get percentage column."""
        return self.table['Percent']

    @property
    def valid_percent(self) -> pd.Series:
        """Get valid percentage column (excluding missing)."""
        if 'Valid Percent' in self.table.columns:
            return self.table['Valid Percent']
        return self.percentages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'variable': self.variable,
            'label': self.label,
            'base': self.base,
            'weighted_base': self.weighted_base,
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'is_weighted': self.is_weighted,
            'data': self.table.to_dict(orient='records'),
        }

    def to_excel(
        self,
        filepath: Union[str, Path],
        include_stats: bool = True,
    ) -> None:
        """Export to Excel file."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main table
            self.table.to_excel(writer, sheet_name='Frequency')

            if include_stats:
                # Stats summary
                stats = pd.DataFrame([{
                    'Variable': self.variable,
                    'Label': self.label,
                    'Base': self.base,
                    'Weighted Base': self.weighted_base,
                    'Mean': self.mean,
                    'Std Dev': self.std,
                    'Median': self.median,
                }])
                stats.to_excel(writer, sheet_name='Statistics', index=False)

    def __repr__(self) -> str:
        weight_info = " (weighted)" if self.is_weighted else ""
        return f"FrequencyResult('{self.variable}', n={self.base}{weight_info})"


def frequency(
    dataset: "DataSet",
    variable: str,
    weight: Optional[str] = None,
    show_labels: bool = True,
    include_missing: bool = False,
    decimals: int = 1,
) -> FrequencyResult:
    """
    Calculate frequency distribution for a variable.

    Parameters
    ----------
    dataset : DataSet
        The dataset to analyze.
    variable : str
        Variable name.
    weight : str, optional
        Weight variable name.
    show_labels : bool, default True
        Show value labels instead of codes.
    include_missing : bool, default False
        Include missing values in output.
    decimals : int, default 1
        Decimal places for percentages.

    Returns
    -------
    FrequencyResult
        Frequency table with statistics.

    Example
    -------
    >>> from quantipymrx import DataSet
    >>> ds = DataSet.from_spss("survey.sav")
    >>> freq = frequency(ds, "q1")
    >>> print(freq.table)
    """
    if variable not in dataset:
        raise ValueError(f"Variable '{variable}' not found in dataset")

    data = dataset.data
    meta = dataset.get_variable_meta(variable)
    text_key = dataset.text_key

    logger.debug(f"Calculating frequency for '{variable}'")

    # Get the column data
    col_data = data[variable].copy()

    # Handle missing values
    missing_codes = meta.missing_values if meta else None
    if missing_codes and not include_missing:
        col_data = col_data.replace(missing_codes, np.nan)
        logger.debug(f"Excluded missing codes: {missing_codes}")

    # Calculate weights - FIXED: explicit check, not heuristic
    is_weighted = False
    if weight is not None:
        if weight not in data.columns:
            logger.warning(f"Weight variable '{weight}' not found, using unweighted")
            weights = pd.Series(1.0, index=data.index)
        else:
            weights = data[weight].fillna(0).astype(float)
            is_weighted = True
            logger.debug(f"Using weight variable '{weight}'")
    else:
        weights = pd.Series(1.0, index=data.index)

    # Get value labels
    value_labels = {}
    if meta and meta.values:
        value_labels = {v.value: v.get_text(text_key) for v in meta.values}

    # Calculate frequency based on variable type
    if meta and meta.is_multi:
        table = _frequency_delimited_set(
            col_data, weights, value_labels, show_labels, decimals, is_weighted
        )
    else:
        table = _frequency_single(
            col_data, weights, value_labels, show_labels, decimals, is_weighted
        )

    # Calculate base
    valid_mask = col_data.notna()
    base = int(valid_mask.sum())
    weighted_base = float(weights[valid_mask].sum()) if is_weighted else None

    # Calculate statistics for numeric/categorical variables
    mean_val = std_val = median_val = None
    if meta and (meta.is_numeric or (meta.is_categorical and meta.values)):
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        valid = numeric_data.notna()

        if valid.sum() > 0:
            if is_weighted:
                mean_val = _weighted_mean(numeric_data, weights)
                std_val = _weighted_std(numeric_data, weights)
                median_val = _weighted_median(numeric_data, weights)
            else:
                mean_val = float(numeric_data.mean())
                std_val = float(numeric_data.std())
                median_val = float(numeric_data.median())

    return FrequencyResult(
        variable=variable,
        label=meta.get_label(text_key) if meta else variable,
        table=table,
        base=base,
        weighted_base=weighted_base,
        mean=mean_val,
        std=std_val,
        median=median_val,
        is_weighted=is_weighted,
        text_key=text_key,
        _value_labels=value_labels,
    )


def _frequency_single(
    col_data: pd.Series,
    weights: pd.Series,
    value_labels: Dict[int, str],
    show_labels: bool,
    decimals: int,
    is_weighted: bool,
) -> pd.DataFrame:
    """Calculate frequency for single-choice variable."""
    # Always calculate weighted counts (weights=1 if unweighted)
    freq_data = col_data.dropna()
    weight_data = weights[col_data.notna()]

    counts = {}
    for val in freq_data.unique():
        mask = freq_data == val
        counts[val] = weight_data[mask].sum()

    if not counts:
        # Empty result
        return pd.DataFrame({
            'Value': [],
            'Label': [],
            'Count': [],
            'Percent': [],
        })

    counts_series = pd.Series(counts).sort_index()

    # Calculate percentages
    total = counts_series.sum()
    pct = (counts_series / total * 100).round(decimals) if total > 0 else counts_series * 0

    # Build table - safely handle index conversion
    value_list = counts_series.index.tolist()
    label_list = []
    for v in value_list:
        try:
            label_list.append(value_labels.get(int(v), '') if value_labels else '')
        except (ValueError, TypeError):
            label_list.append(str(v))

    table = pd.DataFrame({
        'Value': value_list,
        'Label': label_list,
        'Count': counts_series.values,
        'Percent': pct.values,
    })

    # Set proper index
    if show_labels and value_labels:
        index = []
        for v in value_list:
            try:
                index.append(value_labels.get(int(v), str(v)))
            except (ValueError, TypeError):
                index.append(str(v))
        table.index = index
        table.index.name = 'Response'
    else:
        table.index = pd.Index(value_list)
        table.index.name = 'Code'

    return table


def _parse_delimited_value(val: Any) -> Set[int]:
    """
    Parse a delimited set value into a set of integer codes.

    Supports multiple formats:
    - "1;2;3;" (semicolon, Quantipy standard)
    - "1,2,3" (comma)
    - "1|2|3" (pipe)
    - "{1,2,3}" (SPSS MR format)
    - "1 2 3" (space separated)

    Parameters
    ----------
    val : Any
        Value to parse.

    Returns
    -------
    set of int
        Parsed codes.
    """
    if pd.isna(val):
        return set()

    val_str = str(val).strip()

    # Handle empty
    if not val_str or val_str.lower() == 'nan':
        return set()

    # Remove SPSS MR brackets if present
    if val_str.startswith('{') and val_str.endswith('}'):
        val_str = val_str[1:-1]

    # Split by any common delimiter
    codes = set()
    parts = DELIMITED_PATTERN.split(val_str)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        try:
            # Handle both int and float representations
            code = int(float(part))
            codes.add(code)
        except (ValueError, TypeError):
            # Log but don't fail - might be label text
            logger.debug(f"Could not parse delimited code: '{part}'")
            continue

    return codes


def _frequency_delimited_set(
    col_data: pd.Series,
    weights: pd.Series,
    value_labels: Dict[int, str],
    show_labels: bool,
    decimals: int,
    is_weighted: bool,
) -> pd.DataFrame:
    """Calculate frequency for multiple-choice (delimited set) variable."""
    all_codes: Set[int] = set()
    parsed_data: List[Set[int]] = []

    # Parse all values
    for idx, val in col_data.items():
        codes = _parse_delimited_value(val)
        parsed_data.append(codes)
        all_codes.update(codes)

    if not all_codes:
        logger.warning("No valid codes found in delimited set variable")
        return pd.DataFrame({
            'Value': [],
            'Label': [],
            'Count': [],
            'Percent': [],
        })

    # Count each code with weights
    counts = {}
    for code in sorted(all_codes):
        count = 0.0
        for idx, codes in enumerate(parsed_data):
            if code in codes:
                count += weights.iloc[idx]
        counts[code] = count

    counts_series = pd.Series(counts).sort_index()

    # Calculate percentages (% of respondents who answered)
    # Base is respondents with at least one selection
    responding_mask = col_data.notna() & (col_data != '') & (col_data != '{}')
    base = weights[responding_mask].sum()
    pct = (counts_series / base * 100).round(decimals) if base > 0 else counts_series * 0

    # Build table
    value_list = counts_series.index.tolist()
    label_list = [value_labels.get(v, '') for v in value_list] if value_labels else [''] * len(value_list)

    table = pd.DataFrame({
        'Value': value_list,
        'Label': label_list,
        'Count': counts_series.values,
        'Percent': pct.values,
    })

    if show_labels and value_labels:
        table.index = [value_labels.get(v, str(v)) for v in value_list]
        table.index.name = 'Response'
    else:
        table.index = pd.Index(value_list)
        table.index.name = 'Code'

    return table


def _weighted_mean(data: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted mean."""
    valid = data.notna()
    if valid.sum() == 0:
        return np.nan
    return float(np.average(data[valid], weights=weights[valid]))


def _weighted_std(data: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted standard deviation."""
    valid = data.notna()
    if valid.sum() == 0:
        return np.nan

    d = data[valid].values
    w = weights[valid].values

    mean = np.average(d, weights=w)
    variance = np.average((d - mean) ** 2, weights=w)
    return float(np.sqrt(variance))


def _weighted_median(data: pd.Series, weights: pd.Series) -> float:
    """
    Calculate weighted median.

    Uses linear interpolation for weighted percentiles.

    Parameters
    ----------
    data : pd.Series
        Data values.
    weights : pd.Series
        Weight values.

    Returns
    -------
    float
        Weighted median.
    """
    valid = data.notna() & weights.notna() & (weights > 0)
    if valid.sum() == 0:
        return np.nan

    d = data[valid].values
    w = weights[valid].values

    # Sort by data values
    sort_idx = np.argsort(d)
    d_sorted = d[sort_idx]
    w_sorted = w[sort_idx]

    # Cumulative weights
    cumsum = np.cumsum(w_sorted)
    total = cumsum[-1]

    # Find median position (50th percentile)
    median_pos = total / 2

    # Find index where cumsum crosses median position
    idx = np.searchsorted(cumsum, median_pos)

    if idx == 0:
        return float(d_sorted[0])
    elif idx >= len(d_sorted):
        return float(d_sorted[-1])
    else:
        # Linear interpolation
        lower_cumsum = cumsum[idx - 1] if idx > 0 else 0
        upper_cumsum = cumsum[idx]
        lower_val = d_sorted[idx - 1] if idx > 0 else d_sorted[0]
        upper_val = d_sorted[idx]

        if upper_cumsum == lower_cumsum:
            return float(upper_val)

        # Interpolate
        frac = (median_pos - lower_cumsum) / (upper_cumsum - lower_cumsum)
        return float(lower_val + frac * (upper_val - lower_val))


def frequencies_batch(
    dataset: "DataSet",
    variables: Optional[List[str]] = None,
    weight: Optional[str] = None,
    **kwargs,
) -> Dict[str, FrequencyResult]:
    """
    Calculate frequencies for multiple variables.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variables : list of str, optional
        Variables to analyze. If None, uses all categorical.
    weight : str, optional
        Weight variable.
    **kwargs
        Additional arguments for frequency().

    Returns
    -------
    dict
        {variable_name: FrequencyResult}
    """
    if variables is None:
        variables = dataset.singles() + dataset.multis()

    logger.info(f"Calculating frequencies for {len(variables)} variables")

    results = {}
    for var in variables:
        if var in dataset:
            try:
                results[var] = frequency(dataset, var, weight=weight, **kwargs)
            except Exception as e:
                logger.error(f"Could not calculate frequency for {var}: {e}")

    return results
