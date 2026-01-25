"""
QuantipyMRX - Crosstab Analysis

Cross-tabulation between two variables with counts, percentages,
and statistical significance testing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet

logger = logging.getLogger(__name__)


StatType = Literal['counts', 'col_pct', 'row_pct', 'total_pct', 'mean', 'stddev']


@dataclass
class CrosstabResult:
    """
    Result of a crosstab analysis.

    Attributes
    ----------
    x : str
        Row variable name.
    y : str
        Column variable name.
    x_label : str
        Row variable label.
    y_label : str
        Column variable label.
    counts : pd.DataFrame
        Count table.
    col_pct : pd.DataFrame
        Column percentages.
    row_pct : pd.DataFrame
        Row percentages.
    total_pct : pd.DataFrame
        Total percentages.
    base : pd.Series
        Column bases.
    total_base : int
        Total base.
    chi2 : float, optional
        Chi-square statistic.
    chi2_pvalue : float, optional
        Chi-square p-value.
    significance : pd.DataFrame, optional
        Significance letters matrix.
    is_weighted : bool
        Whether analysis was weighted.
    """
    x: str
    y: str
    x_label: str
    y_label: str
    counts: pd.DataFrame
    col_pct: pd.DataFrame
    row_pct: Optional[pd.DataFrame] = None
    total_pct: Optional[pd.DataFrame] = None
    base: pd.Series = field(default_factory=pd.Series)
    weighted_base: Optional[pd.Series] = None
    total_base: int = 0
    chi2: Optional[float] = None
    chi2_pvalue: Optional[float] = None
    chi2_warning: Optional[str] = None  # Warning if expected cells < 5
    significance: Optional[pd.DataFrame] = None
    is_weighted: bool = False
    text_key: str = "en-GB"

    @property
    def table(self) -> pd.DataFrame:
        """Return the column percentage table (most common view)."""
        return self.col_pct

    def get_table(self, stat: StatType = 'col_pct') -> pd.DataFrame:
        """Get specific statistic table."""
        tables = {
            'counts': self.counts,
            'col_pct': self.col_pct,
            'row_pct': self.row_pct,
            'total_pct': self.total_pct,
        }
        return tables.get(stat, self.col_pct)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': self.x,
            'y': self.y,
            'x_label': self.x_label,
            'y_label': self.y_label,
            'counts': self.counts.to_dict(),
            'col_pct': self.col_pct.to_dict(),
            'base': self.base.to_dict() if self.base is not None else None,
            'total_base': self.total_base,
            'chi2': self.chi2,
            'chi2_pvalue': self.chi2_pvalue,
            'chi2_warning': self.chi2_warning,
            'is_weighted': self.is_weighted,
        }

    def to_excel(
        self,
        filepath: Union[str, Path],
        include_counts: bool = True,
        include_significance: bool = True,
    ) -> None:
        """Export to Excel file."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Column percentages (main view)
            self.col_pct.to_excel(writer, sheet_name='Column %')

            if include_counts:
                self.counts.to_excel(writer, sheet_name='Counts')

            if self.row_pct is not None:
                self.row_pct.to_excel(writer, sheet_name='Row %')

            if include_significance and self.significance is not None:
                self.significance.to_excel(writer, sheet_name='Significance')

            # Stats summary
            stats_df = pd.DataFrame([{
                'X Variable': self.x,
                'X Label': self.x_label,
                'Y Variable': self.y,
                'Y Label': self.y_label,
                'Total Base': self.total_base,
                'Chi-Square': self.chi2,
                'P-Value': self.chi2_pvalue,
                'Weighted': self.is_weighted,
            }])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

    def __repr__(self) -> str:
        sig = f", chi2 p={self.chi2_pvalue:.4f}" if self.chi2_pvalue else ""
        return f"CrosstabResult('{self.x}' x '{self.y}', n={self.total_base}{sig})"


def crosstab(
    dataset: "DataSet",
    x: str,
    y: str,
    weight: Optional[str] = None,
    show_labels: bool = True,
    decimals: int = 1,
    test_significance: bool = True,
    sig_level: float = 0.05,
    include_totals: bool = True,
) -> CrosstabResult:
    """
    Calculate crosstab between two variables.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    x : str
        Row variable name.
    y : str
        Column variable name.
    weight : str, optional
        Weight variable.
    show_labels : bool, default True
        Use value labels for index/columns.
    decimals : int, default 1
        Decimal places for percentages.
    test_significance : bool, default True
        Perform chi-square test.
    sig_level : float, default 0.05
        Significance level for column comparison.
    include_totals : bool, default True
        Include total column.

    Returns
    -------
    CrosstabResult
        Cross-tabulation results.

    Example
    -------
    >>> xtab = crosstab(ds, "q1", "gender")
    >>> print(xtab.col_pct)
    >>> print(f"Chi-square p-value: {xtab.chi2_pvalue}")
    """
    if x not in dataset:
        raise ValueError(f"Variable '{x}' not found in dataset")
    if y not in dataset:
        raise ValueError(f"Variable '{y}' not found in dataset")

    data = dataset.data
    x_meta = dataset.get_variable_meta(x)
    y_meta = dataset.get_variable_meta(y)
    text_key = dataset.text_key

    # Get data columns
    x_data = data[x].copy()
    y_data = data[y].copy()

    # Handle weights
    if weight and weight in data.columns:
        weights = data[weight].fillna(0)
        is_weighted = True
    else:
        weights = pd.Series(1, index=data.index)
        is_weighted = False

    # Get value labels
    x_labels = {}
    y_labels = {}
    if x_meta and x_meta.values:
        x_labels = {v.value: v.get_text(text_key) for v in x_meta.values}
    if y_meta and y_meta.values:
        y_labels = {v.value: v.get_text(text_key) for v in y_meta.values}

    # Calculate crosstab
    if x_meta and x_meta.is_multi:
        counts = _crosstab_multi_x(x_data, y_data, weights, x_labels)
    else:
        counts = _crosstab_single(x_data, y_data, weights)

    # Add totals column
    if include_totals:
        counts['Total'] = counts.sum(axis=1)

    # Calculate percentages
    col_totals = counts.sum(axis=0)
    col_pct = (counts.div(col_totals, axis=1) * 100).round(decimals)

    row_totals = counts.sum(axis=1)
    row_pct = (counts.div(row_totals, axis=0) * 100).round(decimals)

    grand_total = counts.values.sum()
    total_pct = (counts / grand_total * 100).round(decimals) if grand_total > 0 else counts * 0

    # Apply labels to index and columns
    if show_labels:
        if x_labels:
            new_index = [x_labels.get(int(v), str(v)) if pd.notna(v) else str(v)
                        for v in counts.index]
            counts.index = new_index
            col_pct.index = new_index
            row_pct.index = new_index
            total_pct.index = new_index

        if y_labels:
            new_cols = []
            for c in counts.columns:
                if c == 'Total':
                    new_cols.append('Total')
                else:
                    try:
                        new_cols.append(y_labels.get(int(c), str(c)))
                    except (ValueError, TypeError):
                        new_cols.append(str(c))
            counts.columns = new_cols
            col_pct.columns = new_cols
            row_pct.columns = new_cols
            total_pct.columns = new_cols

    # Calculate bases
    base = col_totals.copy()
    weighted_base = base.copy() if is_weighted else None
    total_base = int(col_totals.get('Total', col_totals.sum()))

    # Significance testing
    chi2 = chi2_pvalue = chi2_warning = None
    significance = None

    if test_significance and len(counts.columns) > 1:
        # Chi-square test (exclude Total column)
        test_counts = counts.drop('Total', axis=1, errors='ignore')
        if test_counts.shape[0] > 1 and test_counts.shape[1] > 1:
            chi2, chi2_pvalue, chi2_warning = _chi_square_test(test_counts)

            # Column proportion tests
            significance = _column_proportion_tests(
                test_counts, col_pct.drop('Total', axis=1, errors='ignore'),
                sig_level
            )

    logger.debug(f"Crosstab '{x}' x '{y}': base={total_base}, chi2_p={chi2_pvalue}")

    return CrosstabResult(
        x=x,
        y=y,
        x_label=x_meta.get_label(text_key) if x_meta else x,
        y_label=y_meta.get_label(text_key) if y_meta else y,
        counts=counts,
        col_pct=col_pct,
        row_pct=row_pct,
        total_pct=total_pct,
        base=base,
        weighted_base=weighted_base,
        total_base=total_base,
        chi2=chi2,
        chi2_pvalue=chi2_pvalue,
        chi2_warning=chi2_warning,
        significance=significance,
        is_weighted=is_weighted,
        text_key=text_key,
    )


def _crosstab_single(
    x_data: pd.Series,
    y_data: pd.Series,
    weights: pd.Series,
) -> pd.DataFrame:
    """Crosstab for single-choice x variable."""
    # Create weighted crosstab
    df = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'w': weights,
    })

    # Remove rows with missing values
    df = df.dropna(subset=['x', 'y'])

    # Pivot with weights
    result = df.pivot_table(
        values='w',
        index='x',
        columns='y',
        aggfunc='sum',
        fill_value=0,
        observed=False,  # Keep all categories, not just observed ones
    )

    return result


def _crosstab_multi_x(
    x_data: pd.Series,
    y_data: pd.Series,
    weights: pd.Series,
    x_labels: Dict[int, str],
) -> pd.DataFrame:
    """Crosstab for multiple-choice x variable (delimited set)."""
    # Parse x codes
    all_x_codes = set()
    parsed_x = []

    for val in x_data:
        if pd.isna(val):
            parsed_x.append(set())
            continue

        codes = set()
        for code_str in str(val).split(';'):
            code_str = code_str.strip()
            if code_str and code_str != 'nan':
                try:
                    codes.add(int(float(code_str)))
                except ValueError:
                    continue
        parsed_x.append(codes)
        all_x_codes.update(codes)

    # Get unique y values
    y_unique = sorted(y_data.dropna().unique())

    # Build crosstab
    result_data = {}
    for y_val in y_unique:
        y_mask = y_data == y_val
        col_counts = {}

        for x_code in sorted(all_x_codes):
            count = 0
            for idx, (x_codes, is_y) in enumerate(zip(parsed_x, y_mask)):
                if is_y and x_code in x_codes:
                    count += weights.iloc[idx]
            col_counts[x_code] = count

        result_data[y_val] = col_counts

    result = pd.DataFrame(result_data)
    result.index.name = 'x'

    return result


def _chi_square_test(counts: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Perform chi-square test of independence with cell size validation.

    Returns
    -------
    tuple
        (chi2 statistic, p-value, warning message or None)

    Notes
    -----
    The chi-square test assumption requires expected cell frequencies >= 5.
    When more than 20% of cells have expected count < 5, results may be unreliable.
    """
    try:
        chi2, pvalue, dof, expected = stats.chi2_contingency(counts.values)

        # Check for small expected cell counts
        warning = None
        small_cells = (expected < 5).sum()
        total_cells = expected.size
        pct_small = small_cells / total_cells * 100

        if pct_small > 20:
            warning = (
                f"Chi-square assumption violated: {small_cells} of {total_cells} "
                f"cells ({pct_small:.0f}%) have expected count < 5. "
                "Consider using Fisher's exact test or combining categories."
            )
            logger.warning(warning)
        elif small_cells > 0:
            warning = (
                f"Note: {small_cells} of {total_cells} cells have expected count < 5. "
                "Results should be interpreted with caution."
            )
            logger.debug(warning)

        # Also warn if any expected cell is < 1
        if (expected < 1).any():
            warning = (
                "Chi-square test unreliable: some expected cell counts < 1. "
                "Consider combining categories."
            )
            logger.warning(warning)

        return float(chi2), float(pvalue), warning
    except Exception as e:
        logger.error(f"Chi-square test failed: {e}")
        return None, None, None


def _column_proportion_tests(
    counts: pd.DataFrame,
    col_pct: pd.DataFrame,
    sig_level: float = 0.05,
) -> pd.DataFrame:
    """
    Perform column proportion z-tests.

    Returns DataFrame with significance letters (a, b, c, etc.)
    indicating which columns are significantly different.
    """
    n_cols = len(counts.columns)
    if n_cols < 2:
        return None

    # Assign letters to columns
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cols]
    col_letters = dict(zip(counts.columns, letters))

    # Column bases
    bases = counts.sum(axis=0)

    # Create significance matrix
    sig_matrix = pd.DataFrame(
        index=counts.index,
        columns=counts.columns,
        dtype=str,
    )
    sig_matrix[:] = ''

    # Test each row
    for row_idx in counts.index:
        row_counts = counts.loc[row_idx]
        row_pcts = col_pct.loc[row_idx] / 100  # Convert to proportion

        # Compare each pair of columns
        for i, col1 in enumerate(counts.columns):
            sig_letters = []

            for j, col2 in enumerate(counts.columns):
                if i == j:
                    continue

                # Z-test for proportions
                n1 = bases[col1]
                n2 = bases[col2]
                p1 = row_pcts[col1]
                p2 = row_pcts[col2]

                if n1 < 5 or n2 < 5:
                    continue

                # Pooled proportion
                p_pool = (row_counts[col1] + row_counts[col2]) / (n1 + n2)

                if p_pool <= 0 or p_pool >= 1:
                    continue

                # Standard error
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

                if se == 0:
                    continue

                # Z-statistic
                z = (p1 - p2) / se

                # Two-tailed p-value
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))

                # If col1 is significantly higher than col2
                if p_value < sig_level and p1 > p2:
                    sig_letters.append(col_letters[col2])

            sig_matrix.loc[row_idx, col1] = ''.join(sorted(sig_letters))

    return sig_matrix


def crosstab_batch(
    dataset: "DataSet",
    x_vars: List[str],
    y_var: str,
    weight: Optional[str] = None,
    **kwargs,
) -> Dict[str, CrosstabResult]:
    """
    Calculate crosstabs for multiple x variables against one y variable.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    x_vars : list of str
        Row variables.
    y_var : str
        Column variable (banner).
    weight : str, optional
        Weight variable.
    **kwargs
        Additional arguments for crosstab().

    Returns
    -------
    dict
        {x_variable: CrosstabResult}
    """
    results = {}
    for x_var in x_vars:
        if x_var in dataset and x_var != y_var:
            try:
                results[x_var] = crosstab(
                    dataset, x=x_var, y=y_var, weight=weight, **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not calculate crosstab for {x_var}: {e}")

    return results
