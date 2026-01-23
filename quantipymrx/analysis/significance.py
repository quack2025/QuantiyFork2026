"""
QuantipyMRX - Statistical Significance Tests

Comprehensive statistical testing for market research:
- Z-test for column proportions
- T-test for column means
- Chi-square test of independence
- ANOVA for multiple group comparison
- Correlation (Pearson/Spearman)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet


class SignificanceLevel(float, Enum):
    """Common significance levels."""
    P_10 = 0.10   # 90% confidence
    P_05 = 0.05   # 95% confidence
    P_01 = 0.01   # 99% confidence
    P_001 = 0.001 # 99.9% confidence


@dataclass
class TestResult:
    """Base result for statistical tests."""
    test_name: str
    statistic: float
    p_value: float
    df: Optional[float] = None  # degrees of freedom
    significant: bool = False
    sig_level: float = 0.05
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.significant = self.p_value < self.sig_level

    @property
    def significance_stars(self) -> str:
        """Return significance stars (* ** ***)."""
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        elif self.p_value < 0.10:
            return "†"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'df': self.df,
            'significant': self.significant,
            'sig_level': self.sig_level,
            'stars': self.significance_stars,
            **self.details,
        }


@dataclass
class ProportionTestResult(TestResult):
    """Result for proportion comparison tests."""
    prop1: float = 0.0
    prop2: float = 0.0
    n1: int = 0
    n2: int = 0
    diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.diff = self.prop1 - self.prop2


@dataclass
class MeansTestResult(TestResult):
    """Result for means comparison tests."""
    mean1: float = 0.0
    mean2: float = 0.0
    n1: int = 0
    n2: int = 0
    std1: float = 0.0
    std2: float = 0.0
    diff: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.diff = self.mean1 - self.mean2


@dataclass
class ANOVAResult(TestResult):
    """Result for ANOVA test."""
    group_means: Dict[Any, float] = field(default_factory=dict)
    group_stds: Dict[Any, float] = field(default_factory=dict)
    group_ns: Dict[Any, int] = field(default_factory=dict)
    post_hoc: Optional[pd.DataFrame] = None  # Tukey HSD results


@dataclass
class CorrelationResult(TestResult):
    """Result for correlation test."""
    method: str = "pearson"
    r: float = 0.0  # correlation coefficient
    r_squared: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.r = self.statistic
        self.r_squared = self.r ** 2


@dataclass
class SignificanceMatrix:
    """Matrix of significance letters for crosstab columns."""
    letters: pd.DataFrame
    comparisons: List[Dict[str, Any]]
    sig_level: float = 0.05

    def __repr__(self) -> str:
        return f"SignificanceMatrix(columns={list(self.letters.columns)}, level={self.sig_level})"


# =============================================================================
# Z-TEST FOR PROPORTIONS
# =============================================================================

def z_test_proportions(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    sig_level: float = 0.05,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
) -> ProportionTestResult:
    """
    Two-proportion z-test.

    Tests whether two proportions are significantly different.

    Parameters
    ----------
    count1, count2 : int
        Number of successes in each group.
    n1, n2 : int
        Sample sizes.
    sig_level : float
        Significance level.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    ProportionTestResult
        Test results including z-statistic and p-value.

    Example
    -------
    >>> result = z_test_proportions(80, 200, 60, 200)
    >>> print(f"p-value: {result.p_value:.4f}")
    """
    p1 = count1 / n1 if n1 > 0 else 0
    p2 = count2 / n2 if n2 > 0 else 0

    # Pooled proportion
    p_pool = (count1 + count2) / (n1 + n2) if (n1 + n2) > 0 else 0

    # Standard error
    if p_pool <= 0 or p_pool >= 1 or n1 < 1 or n2 < 1:
        return ProportionTestResult(
            test_name="z-test proportions",
            statistic=0.0,
            p_value=1.0,
            prop1=p1,
            prop2=p2,
            n1=n1,
            n2=n2,
            sig_level=sig_level,
        )

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return ProportionTestResult(
            test_name="z-test proportions",
            statistic=0.0,
            p_value=1.0,
            prop1=p1,
            prop2=p2,
            n1=n1,
            n2=n2,
            sig_level=sig_level,
        )

    # Z-statistic
    z = (p1 - p2) / se

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z)
    else:  # less
        p_value = stats.norm.cdf(z)

    # Confidence interval for difference
    z_crit = stats.norm.ppf(1 - sig_level/2)
    se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    ci_lower = (p1 - p2) - z_crit * se_diff
    ci_upper = (p1 - p2) + z_crit * se_diff

    return ProportionTestResult(
        test_name="z-test proportions",
        statistic=float(z),
        p_value=float(p_value),
        prop1=p1,
        prop2=p2,
        n1=n1,
        n2=n2,
        sig_level=sig_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


# =============================================================================
# T-TEST FOR MEANS
# =============================================================================

def t_test_means(
    data1: Union[pd.Series, np.ndarray, List[float]],
    data2: Union[pd.Series, np.ndarray, List[float]],
    sig_level: float = 0.05,
    equal_var: bool = False,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
) -> MeansTestResult:
    """
    Independent samples t-test.

    Tests whether means of two groups are significantly different.

    Parameters
    ----------
    data1, data2 : array-like
        Sample data for each group.
    sig_level : float
        Significance level.
    equal_var : bool
        If True, performs standard t-test assuming equal variances.
        If False (default), performs Welch's t-test.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    MeansTestResult
        Test results.

    Example
    -------
    >>> group1 = [4.2, 3.8, 4.5, 4.1, 3.9]
    >>> group2 = [3.5, 3.2, 3.8, 3.4, 3.6]
    >>> result = t_test_means(group1, group2)
    >>> print(f"t={result.statistic:.2f}, p={result.p_value:.4f}")
    """
    # Convert to arrays and remove NaN
    arr1 = np.array(data1, dtype=float)
    arr2 = np.array(data2, dtype=float)
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    if len(arr1) < 2 or len(arr2) < 2:
        return MeansTestResult(
            test_name="t-test means",
            statistic=0.0,
            p_value=1.0,
            mean1=np.mean(arr1) if len(arr1) > 0 else 0,
            mean2=np.mean(arr2) if len(arr2) > 0 else 0,
            n1=len(arr1),
            n2=len(arr2),
            sig_level=sig_level,
        )

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(
        arr1, arr2,
        equal_var=equal_var,
        alternative=alternative,
    )

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(arr1)-1)*np.var(arr1, ddof=1) + (len(arr2)-1)*np.var(arr2, ddof=1)) /
        (len(arr1) + len(arr2) - 2)
    )
    cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std if pooled_std > 0 else 0

    # Degrees of freedom (Welch-Satterthwaite for unequal variances)
    if equal_var:
        df = len(arr1) + len(arr2) - 2
    else:
        v1, v2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
        n1, n2 = len(arr1), len(arr2)
        df = ((v1/n1 + v2/n2)**2) / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    # Confidence interval
    t_crit = stats.t.ppf(1 - sig_level/2, df)
    se_diff = np.sqrt(np.var(arr1, ddof=1)/len(arr1) + np.var(arr2, ddof=1)/len(arr2))
    diff = np.mean(arr1) - np.mean(arr2)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    return MeansTestResult(
        test_name="t-test means",
        statistic=float(t_stat),
        p_value=float(p_value),
        df=float(df),
        mean1=float(np.mean(arr1)),
        mean2=float(np.mean(arr2)),
        n1=len(arr1),
        n2=len(arr2),
        std1=float(np.std(arr1, ddof=1)),
        std2=float(np.std(arr2, ddof=1)),
        sig_level=sig_level,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        details={'cohens_d': cohens_d},
    )


def t_test_paired(
    data1: Union[pd.Series, np.ndarray, List[float]],
    data2: Union[pd.Series, np.ndarray, List[float]],
    sig_level: float = 0.05,
    alternative: Literal['two-sided', 'greater', 'less'] = 'two-sided',
) -> MeansTestResult:
    """
    Paired samples t-test.

    For comparing means from the same subjects (e.g., pre/post).

    Parameters
    ----------
    data1, data2 : array-like
        Paired sample data.
    sig_level : float
        Significance level.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    MeansTestResult
    """
    arr1 = np.array(data1, dtype=float)
    arr2 = np.array(data2, dtype=float)

    # Remove pairs with NaN
    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    arr1 = arr1[valid]
    arr2 = arr2[valid]

    if len(arr1) < 2:
        return MeansTestResult(
            test_name="paired t-test",
            statistic=0.0,
            p_value=1.0,
            n1=len(arr1),
            n2=len(arr2),
            sig_level=sig_level,
        )

    t_stat, p_value = stats.ttest_rel(arr1, arr2, alternative=alternative)
    df = len(arr1) - 1

    return MeansTestResult(
        test_name="paired t-test",
        statistic=float(t_stat),
        p_value=float(p_value),
        df=float(df),
        mean1=float(np.mean(arr1)),
        mean2=float(np.mean(arr2)),
        n1=len(arr1),
        n2=len(arr2),
        std1=float(np.std(arr1, ddof=1)),
        std2=float(np.std(arr2, ddof=1)),
        sig_level=sig_level,
    )


# =============================================================================
# CHI-SQUARE TEST
# =============================================================================

def chi_square_test(
    observed: Union[pd.DataFrame, np.ndarray],
    sig_level: float = 0.05,
) -> TestResult:
    """
    Chi-square test of independence.

    Tests whether two categorical variables are independent.

    Parameters
    ----------
    observed : DataFrame or 2D array
        Contingency table of observed frequencies.
    sig_level : float
        Significance level.

    Returns
    -------
    TestResult
        Including chi2 statistic, p-value, and expected frequencies.

    Example
    -------
    >>> table = pd.DataFrame({
    ...     'Male': [45, 55],
    ...     'Female': [35, 65]
    ... }, index=['Yes', 'No'])
    >>> result = chi_square_test(table)
    >>> print(f"χ²={result.statistic:.2f}, p={result.p_value:.4f}")
    """
    if isinstance(observed, pd.DataFrame):
        obs_array = observed.values
    else:
        obs_array = np.array(observed)

    # Require at least 2x2
    if obs_array.shape[0] < 2 or obs_array.shape[1] < 2:
        return TestResult(
            test_name="chi-square",
            statistic=0.0,
            p_value=1.0,
            sig_level=sig_level,
        )

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(obs_array)

        # Cramér's V for effect size
        n = obs_array.sum()
        min_dim = min(obs_array.shape[0] - 1, obs_array.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if n * min_dim > 0 else 0

        return TestResult(
            test_name="chi-square",
            statistic=float(chi2),
            p_value=float(p_value),
            df=float(dof),
            sig_level=sig_level,
            details={
                'expected': expected.tolist(),
                'cramers_v': cramers_v,
            },
        )
    except Exception:
        return TestResult(
            test_name="chi-square",
            statistic=0.0,
            p_value=1.0,
            sig_level=sig_level,
        )


# =============================================================================
# ANOVA
# =============================================================================

def anova_oneway(
    *groups: Union[pd.Series, np.ndarray, List[float]],
    sig_level: float = 0.05,
    post_hoc: bool = True,
) -> ANOVAResult:
    """
    One-way ANOVA.

    Tests whether means of 3+ groups are significantly different.

    Parameters
    ----------
    *groups : array-like
        Data for each group.
    sig_level : float
        Significance level.
    post_hoc : bool
        If True, performs Tukey HSD post-hoc test.

    Returns
    -------
    ANOVAResult
        Including F-statistic, p-value, and post-hoc comparisons.

    Example
    -------
    >>> group1 = [4.2, 3.8, 4.5, 4.1]
    >>> group2 = [3.5, 3.2, 3.8, 3.4]
    >>> group3 = [2.9, 3.1, 2.8, 3.0]
    >>> result = anova_oneway(group1, group2, group3)
    >>> print(f"F={result.statistic:.2f}, p={result.p_value:.4f}")
    """
    # Convert and clean data
    clean_groups = []
    for g in groups:
        arr = np.array(g, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            clean_groups.append(arr)

    if len(clean_groups) < 2:
        return ANOVAResult(
            test_name="one-way ANOVA",
            statistic=0.0,
            p_value=1.0,
            sig_level=sig_level,
        )

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*clean_groups)

    # Calculate group statistics
    group_means = {i: float(np.mean(g)) for i, g in enumerate(clean_groups)}
    group_stds = {i: float(np.std(g, ddof=1)) for i, g in enumerate(clean_groups)}
    group_ns = {i: len(g) for i, g in enumerate(clean_groups)}

    # Degrees of freedom
    k = len(clean_groups)
    n_total = sum(len(g) for g in clean_groups)
    df_between = k - 1
    df_within = n_total - k

    # Effect size (eta-squared)
    grand_mean = np.mean(np.concatenate(clean_groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in clean_groups)
    ss_total = sum(np.sum((g - grand_mean)**2) for g in clean_groups)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Post-hoc tests (Tukey HSD)
    post_hoc_df = None
    if post_hoc and len(clean_groups) >= 3:
        post_hoc_df = _tukey_hsd(clean_groups, sig_level)

    return ANOVAResult(
        test_name="one-way ANOVA",
        statistic=float(f_stat),
        p_value=float(p_value),
        df=float(df_between),
        sig_level=sig_level,
        group_means=group_means,
        group_stds=group_stds,
        group_ns=group_ns,
        post_hoc=post_hoc_df,
        details={
            'df_within': df_within,
            'eta_squared': eta_squared,
        },
    )


def _tukey_hsd(groups: List[np.ndarray], sig_level: float) -> pd.DataFrame:
    """Perform Tukey HSD post-hoc test."""
    from scipy.stats import studentized_range

    k = len(groups)
    n_total = sum(len(g) for g in groups)
    df_within = n_total - k

    # MSE (mean square error within)
    ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
    mse = ss_within / df_within if df_within > 0 else 1

    results = []
    for i in range(k):
        for j in range(i + 1, k):
            mean_diff = np.mean(groups[i]) - np.mean(groups[j])
            se = np.sqrt(mse * (1/len(groups[i]) + 1/len(groups[j])) / 2)

            if se > 0:
                q = abs(mean_diff) / se
                # P-value from studentized range distribution
                p_value = 1 - studentized_range.cdf(q * np.sqrt(2), k, df_within)
            else:
                q = 0
                p_value = 1

            results.append({
                'Group 1': i,
                'Group 2': j,
                'Mean Diff': mean_diff,
                'q': q,
                'p-value': p_value,
                'Significant': p_value < sig_level,
            })

    return pd.DataFrame(results)


def anova_from_dataset(
    dataset: "DataSet",
    variable: str,
    group_var: str,
    weight: Optional[str] = None,
    sig_level: float = 0.05,
) -> ANOVAResult:
    """
    Perform ANOVA using dataset variables.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variable : str
        Numeric variable to compare.
    group_var : str
        Categorical variable defining groups.
    weight : str, optional
        Weight variable (not yet implemented for ANOVA).
    sig_level : float
        Significance level.

    Returns
    -------
    ANOVAResult
    """
    data = dataset.data
    groups = []
    group_labels = {}

    for code in data[group_var].dropna().unique():
        mask = data[group_var] == code
        group_data = data.loc[mask, variable].dropna().values
        if len(group_data) > 0:
            groups.append(group_data)
            group_labels[len(groups) - 1] = code

    result = anova_oneway(*groups, sig_level=sig_level)

    # Replace numeric indices with actual codes
    result.group_means = {group_labels.get(k, k): v for k, v in result.group_means.items()}
    result.group_stds = {group_labels.get(k, k): v for k, v in result.group_stds.items()}
    result.group_ns = {group_labels.get(k, k): v for k, v in result.group_ns.items()}

    return result


# =============================================================================
# CORRELATION
# =============================================================================

def correlation(
    x: Union[pd.Series, np.ndarray, List[float]],
    y: Union[pd.Series, np.ndarray, List[float]],
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
    sig_level: float = 0.05,
) -> CorrelationResult:
    """
    Calculate correlation between two variables.

    Parameters
    ----------
    x, y : array-like
        Variables to correlate.
    method : str
        'pearson', 'spearman', or 'kendall'.
    sig_level : float
        Significance level.

    Returns
    -------
    CorrelationResult
        Including correlation coefficient and p-value.

    Example
    -------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 5, 4, 5]
    >>> result = correlation(x, y)
    >>> print(f"r={result.r:.3f}, p={result.p_value:.4f}")
    """
    # Convert and align
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Remove pairs with NaN
    valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    if len(x_arr) < 3:
        return CorrelationResult(
            test_name=f"{method} correlation",
            statistic=0.0,
            p_value=1.0,
            method=method,
            sig_level=sig_level,
        )

    # Calculate correlation
    if method == 'pearson':
        r, p_value = stats.pearsonr(x_arr, y_arr)
    elif method == 'spearman':
        r, p_value = stats.spearmanr(x_arr, y_arr)
    else:  # kendall
        r, p_value = stats.kendalltau(x_arr, y_arr)

    return CorrelationResult(
        test_name=f"{method} correlation",
        statistic=float(r),
        p_value=float(p_value),
        df=float(len(x_arr) - 2),
        method=method,
        sig_level=sig_level,
    )


def correlation_matrix(
    dataset: "DataSet",
    variables: Optional[List[str]] = None,
    method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix for multiple variables.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variables : list of str, optional
        Variables to include. If None, uses all numeric.
    method : str
        Correlation method.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Correlation coefficients and p-values matrices.
    """
    if variables is None:
        variables = dataset.numerics()

    data = dataset.data[variables].dropna()

    # Correlation matrix
    if method == 'pearson':
        corr = data.corr(method='pearson')
    elif method == 'spearman':
        corr = data.corr(method='spearman')
    else:
        corr = data.corr(method='kendall')

    # P-values matrix
    n = len(variables)
    p_values = pd.DataFrame(np.ones((n, n)), index=variables, columns=variables)

    for i, v1 in enumerate(variables):
        for j, v2 in enumerate(variables):
            if i != j:
                result = correlation(data[v1], data[v2], method=method)
                p_values.iloc[i, j] = result.p_value

    return corr, p_values


# =============================================================================
# COLUMN SIGNIFICANCE (FOR CROSSTABS)
# =============================================================================

def column_significance_test(
    counts: pd.DataFrame,
    percentages: pd.DataFrame,
    test_type: Literal['proportions', 'means'] = 'proportions',
    sig_level: float = 0.05,
) -> SignificanceMatrix:
    """
    Perform significance tests between all column pairs.

    Used for adding significance letters (A, B, C...) to crosstabs.

    Parameters
    ----------
    counts : pd.DataFrame
        Count table from crosstab.
    percentages : pd.DataFrame
        Percentage table (column %).
    test_type : str
        'proportions' for z-test, 'means' for t-test.
    sig_level : float
        Significance level.

    Returns
    -------
    SignificanceMatrix
        Matrix of significance letters.
    """
    # Exclude Total column if present
    if 'Total' in counts.columns:
        counts = counts.drop('Total', axis=1)
        percentages = percentages.drop('Total', axis=1)

    n_cols = len(counts.columns)
    if n_cols < 2:
        return SignificanceMatrix(
            letters=pd.DataFrame(index=counts.index, columns=counts.columns, data=''),
            comparisons=[],
            sig_level=sig_level,
        )

    # Assign letters to columns
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_cols]
    col_to_letter = dict(zip(counts.columns, letters))

    # Column bases
    bases = counts.sum(axis=0)

    # Initialize results
    sig_letters = pd.DataFrame(
        index=counts.index,
        columns=counts.columns,
        data='',
    )
    comparisons = []

    # Test each row
    for row_idx in counts.index:
        row_counts = counts.loc[row_idx]
        row_pcts = percentages.loc[row_idx] / 100  # Convert to proportion

        # Compare each pair
        for i, col1 in enumerate(counts.columns):
            letters_for_col = []

            for j, col2 in enumerate(counts.columns):
                if i == j:
                    continue

                n1, n2 = bases[col1], bases[col2]
                c1, c2 = row_counts[col1], row_counts[col2]
                p1, p2 = row_pcts[col1], row_pcts[col2]

                if n1 < 5 or n2 < 5:
                    continue

                # Z-test for proportions
                result = z_test_proportions(
                    int(c1), int(n1),
                    int(c2), int(n2),
                    sig_level=sig_level,
                )

                if result.significant and p1 > p2:
                    letters_for_col.append(col_to_letter[col2])
                    comparisons.append({
                        'row': row_idx,
                        'col_higher': col1,
                        'col_lower': col2,
                        'z': result.statistic,
                        'p_value': result.p_value,
                    })

            sig_letters.loc[row_idx, col1] = ''.join(sorted(letters_for_col))

    return SignificanceMatrix(
        letters=sig_letters,
        comparisons=comparisons,
        sig_level=sig_level,
    )
