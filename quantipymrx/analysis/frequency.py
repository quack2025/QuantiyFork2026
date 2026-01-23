"""
QuantipyMRX - Frequency Analysis

Calculate frequency distributions with counts, percentages, and statistics.
Supports weighted analysis and multiple variable types.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet


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

    # Get the column data
    col_data = data[variable].copy()

    # Handle missing values
    missing_codes = meta.missing_values if meta else None
    if missing_codes and not include_missing:
        col_data = col_data.replace(missing_codes, np.nan)

    # Calculate weights
    if weight and weight in data.columns:
        weights = data[weight].fillna(0)
        is_weighted = True
    else:
        weights = pd.Series(1, index=data.index)
        is_weighted = False

    # Get value labels
    value_labels = {}
    if meta and meta.values:
        value_labels = {v.value: v.get_text(text_key) for v in meta.values}

    # Calculate frequency based on variable type
    if meta and meta.is_multi:
        table = _frequency_delimited_set(
            col_data, weights, value_labels, show_labels, decimals
        )
    else:
        table = _frequency_single(
            col_data, weights, value_labels, show_labels, decimals
        )

    # Calculate base
    valid_mask = col_data.notna()
    base = int(valid_mask.sum())
    weighted_base = float(weights[valid_mask].sum()) if is_weighted else None

    # Calculate statistics for numeric variables
    mean_val = std_val = median_val = None
    if meta and meta.is_numeric:
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        if is_weighted:
            mean_val = _weighted_mean(numeric_data, weights)
            std_val = _weighted_std(numeric_data, weights)
        else:
            mean_val = numeric_data.mean()
            std_val = numeric_data.std()
        median_val = numeric_data.median()
    elif meta and meta.is_categorical and meta.values:
        # Calculate mean of codes for categorical
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        if is_weighted:
            mean_val = _weighted_mean(numeric_data, weights)
            std_val = _weighted_std(numeric_data, weights)
        else:
            mean_val = numeric_data.mean()
            std_val = numeric_data.std()
        median_val = numeric_data.median()

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
) -> pd.DataFrame:
    """Calculate frequency for single-choice variable."""
    # Get value counts
    if weights.sum() > len(weights):  # Weighted
        # Weighted frequency
        freq_data = col_data.dropna()
        weight_data = weights[col_data.notna()]

        counts = {}
        for val in freq_data.unique():
            mask = freq_data == val
            counts[val] = weight_data[mask].sum()

        counts_series = pd.Series(counts).sort_index()
    else:
        counts_series = col_data.value_counts().sort_index()

    # Calculate percentages
    total = counts_series.sum()
    pct = (counts_series / total * 100).round(decimals) if total > 0 else counts_series * 0

    # Build table
    if show_labels and value_labels:
        index = [value_labels.get(int(v), str(v)) for v in counts_series.index]
    else:
        index = counts_series.index.tolist()

    table = pd.DataFrame({
        'Value': counts_series.index.tolist(),
        'Label': [value_labels.get(int(v), '') for v in counts_series.index] if value_labels else [''] * len(counts_series),
        'Count': counts_series.values,
        'Percent': pct.values,
    })

    # Set proper index
    if show_labels and value_labels:
        table.index = index
        table.index.name = 'Response'
    else:
        table.index = counts_series.index
        table.index.name = 'Code'

    return table


def _frequency_delimited_set(
    col_data: pd.Series,
    weights: pd.Series,
    value_labels: Dict[int, str],
    show_labels: bool,
    decimals: int,
) -> pd.DataFrame:
    """Calculate frequency for multiple-choice (delimited set) variable."""
    # Parse delimited values (format: "1;2;3;")
    all_codes = set()
    parsed_data = []

    for idx, val in col_data.items():
        if pd.isna(val):
            parsed_data.append(set())
            continue

        # Parse codes from delimited string
        codes = set()
        for code_str in str(val).split(';'):
            code_str = code_str.strip()
            if code_str and code_str != 'nan':
                try:
                    codes.add(int(float(code_str)))
                except ValueError:
                    continue
        parsed_data.append(codes)
        all_codes.update(codes)

    # Count each code
    counts = {}
    for code in sorted(all_codes):
        count = 0
        for idx, codes in enumerate(parsed_data):
            if code in codes:
                count += weights.iloc[idx]
        counts[code] = count

    counts_series = pd.Series(counts).sort_index()

    # Calculate percentages (% of respondents)
    base = (col_data.notna() * weights).sum()
    pct = (counts_series / base * 100).round(decimals) if base > 0 else counts_series * 0

    # Build table
    table = pd.DataFrame({
        'Value': counts_series.index.tolist(),
        'Label': [value_labels.get(v, '') for v in counts_series.index] if value_labels else [''] * len(counts_series),
        'Count': counts_series.values,
        'Percent': pct.values,
    })

    if show_labels and value_labels:
        table.index = [value_labels.get(v, str(v)) for v in counts_series.index]
        table.index.name = 'Response'
    else:
        table.index = counts_series.index
        table.index.name = 'Code'

    return table


def _weighted_mean(data: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted mean."""
    valid = data.notna()
    if valid.sum() == 0:
        return np.nan
    return np.average(data[valid], weights=weights[valid])


def _weighted_std(data: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted standard deviation."""
    valid = data.notna()
    if valid.sum() == 0:
        return np.nan

    d = data[valid].values
    w = weights[valid].values

    mean = np.average(d, weights=w)
    variance = np.average((d - mean) ** 2, weights=w)
    return np.sqrt(variance)


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

    results = {}
    for var in variables:
        if var in dataset:
            try:
                results[var] = frequency(dataset, var, weight=weight, **kwargs)
            except Exception as e:
                print(f"Warning: Could not calculate frequency for {var}: {e}")

    return results
