"""
QuantipyMRX - Market Research Calculations

Specialized calculations for market research:
- NPS (Net Promoter Score)
- Top Box / Top 2 Box
- Bottom Box / Bottom 2 Box
- Nets (category groupings)
- Index calculations
- Gap Analysis
- Satisfaction metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet


# =============================================================================
# NET PROMOTER SCORE (NPS)
# =============================================================================

@dataclass
class NPSResult:
    """Result of NPS calculation."""
    nps: float
    promoters_pct: float
    passives_pct: float
    detractors_pct: float
    promoters_count: int
    passives_count: int
    detractors_count: int
    base: int
    weighted_base: Optional[float] = None
    is_weighted: bool = False

    @property
    def nps_category(self) -> str:
        """Categorize NPS score."""
        if self.nps >= 70:
            return "Excellent"
        elif self.nps >= 50:
            return "Great"
        elif self.nps >= 30:
            return "Good"
        elif self.nps >= 0:
            return "Needs Improvement"
        else:
            return "Critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nps': self.nps,
            'nps_category': self.nps_category,
            'promoters_pct': self.promoters_pct,
            'passives_pct': self.passives_pct,
            'detractors_pct': self.detractors_pct,
            'promoters_count': self.promoters_count,
            'passives_count': self.passives_count,
            'detractors_count': self.detractors_count,
            'base': self.base,
            'weighted_base': self.weighted_base,
        }

    def __repr__(self) -> str:
        return f"NPSResult(nps={self.nps:.1f}, category='{self.nps_category}')"


def calculate_nps(
    data: Union[pd.Series, np.ndarray, List[int]],
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
    scale: Literal['0-10', '1-10', '1-5'] = '0-10',
    promoter_codes: Optional[List[int]] = None,
    detractor_codes: Optional[List[int]] = None,
) -> NPSResult:
    """
    Calculate Net Promoter Score.

    NPS = % Promoters - % Detractors

    Parameters
    ----------
    data : array-like
        Response data (typically 0-10 scale).
    weights : array-like, optional
        Weight values.
    scale : str
        Scale type: '0-10' (default), '1-10', or '1-5'.
    promoter_codes : list, optional
        Custom codes for promoters. Default based on scale.
    detractor_codes : list, optional
        Custom codes for detractors.

    Returns
    -------
    NPSResult
        NPS calculation results.

    Example
    -------
    >>> scores = [9, 10, 8, 7, 6, 9, 10, 5, 8, 9]
    >>> result = calculate_nps(scores)
    >>> print(f"NPS: {result.nps:.1f}")
    """
    arr = np.array(data, dtype=float)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid_mask]
    else:
        w = np.ones(len(arr))

    if len(arr) == 0:
        return NPSResult(
            nps=0, promoters_pct=0, passives_pct=0, detractors_pct=0,
            promoters_count=0, passives_count=0, detractors_count=0,
            base=0,
        )

    # Define codes based on scale
    if promoter_codes is None or detractor_codes is None:
        if scale == '0-10':
            promoter_codes = [9, 10]
            detractor_codes = [0, 1, 2, 3, 4, 5, 6]
        elif scale == '1-10':
            promoter_codes = [9, 10]
            detractor_codes = [1, 2, 3, 4, 5, 6]
        elif scale == '1-5':
            promoter_codes = [5]
            detractor_codes = [1, 2]
        else:
            promoter_codes = [9, 10]
            detractor_codes = [0, 1, 2, 3, 4, 5, 6]

    # Calculate counts
    promoters_mask = np.isin(arr, promoter_codes)
    detractors_mask = np.isin(arr, detractor_codes)
    passives_mask = ~promoters_mask & ~detractors_mask

    promoters_count = int(promoters_mask.sum())
    detractors_count = int(detractors_mask.sum())
    passives_count = int(passives_mask.sum())

    # Calculate weighted percentages
    total_weight = w.sum()
    promoters_pct = (w[promoters_mask].sum() / total_weight * 100) if total_weight > 0 else 0
    detractors_pct = (w[detractors_mask].sum() / total_weight * 100) if total_weight > 0 else 0
    passives_pct = (w[passives_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    # NPS = Promoters% - Detractors%
    nps = promoters_pct - detractors_pct

    is_weighted = weights is not None
    weighted_base = float(total_weight) if is_weighted else None

    return NPSResult(
        nps=float(nps),
        promoters_pct=float(promoters_pct),
        passives_pct=float(passives_pct),
        detractors_pct=float(detractors_pct),
        promoters_count=promoters_count,
        passives_count=passives_count,
        detractors_count=detractors_count,
        base=len(arr),
        weighted_base=weighted_base,
        is_weighted=is_weighted,
    )


def nps_from_dataset(
    dataset: "DataSet",
    variable: str,
    weight: Optional[str] = None,
    **kwargs,
) -> NPSResult:
    """Calculate NPS from a dataset variable."""
    data = dataset.data[variable]
    weights = dataset.data[weight] if weight else None
    return calculate_nps(data, weights, **kwargs)


# =============================================================================
# TOP BOX / BOTTOM BOX
# =============================================================================

@dataclass
class BoxResult:
    """Result of Top Box / Bottom Box calculation."""
    name: str
    percentage: float
    count: int
    base: int
    codes: List[int]
    weighted_base: Optional[float] = None
    is_weighted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'percentage': self.percentage,
            'count': self.count,
            'base': self.base,
            'codes': self.codes,
        }

    def __repr__(self) -> str:
        return f"BoxResult('{self.name}': {self.percentage:.1f}%)"


def top_box(
    data: Union[pd.Series, np.ndarray],
    n_boxes: int = 1,
    scale_max: Optional[int] = None,
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> BoxResult:
    """
    Calculate Top Box percentage.

    Parameters
    ----------
    data : array-like
        Response data.
    n_boxes : int
        Number of top categories (1 = Top Box, 2 = Top 2 Box).
    scale_max : int, optional
        Maximum value of scale. If None, auto-detected.
    weights : array-like, optional
        Weight values.

    Returns
    -------
    BoxResult

    Example
    -------
    >>> scores = [1, 2, 3, 4, 5, 5, 4, 5, 3, 5]
    >>> result = top_box(scores, n_boxes=2)  # Top 2 Box (4 & 5)
    >>> print(f"Top 2 Box: {result.percentage:.1f}%")
    """
    arr = np.array(data, dtype=float)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid_mask]
    else:
        w = np.ones(len(arr))

    if len(arr) == 0:
        return BoxResult(
            name=f"Top {n_boxes} Box",
            percentage=0,
            count=0,
            base=0,
            codes=[],
        )

    # Determine top codes
    if scale_max is None:
        scale_max = int(arr.max())

    top_codes = list(range(scale_max - n_boxes + 1, scale_max + 1))

    # Calculate
    top_mask = np.isin(arr, top_codes)
    count = int(top_mask.sum())
    total_weight = w.sum()
    percentage = (w[top_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    is_weighted = weights is not None
    weighted_base = float(total_weight) if is_weighted else None

    name = "Top Box" if n_boxes == 1 else f"Top {n_boxes} Box"

    return BoxResult(
        name=name,
        percentage=float(percentage),
        count=count,
        base=len(arr),
        codes=top_codes,
        weighted_base=weighted_base,
        is_weighted=is_weighted,
    )


def bottom_box(
    data: Union[pd.Series, np.ndarray],
    n_boxes: int = 1,
    scale_min: Optional[int] = None,
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> BoxResult:
    """
    Calculate Bottom Box percentage.

    Parameters
    ----------
    data : array-like
        Response data.
    n_boxes : int
        Number of bottom categories.
    scale_min : int, optional
        Minimum value of scale. If None, auto-detected.
    weights : array-like, optional
        Weight values.

    Returns
    -------
    BoxResult
    """
    arr = np.array(data, dtype=float)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid_mask]
    else:
        w = np.ones(len(arr))

    if len(arr) == 0:
        return BoxResult(
            name=f"Bottom {n_boxes} Box",
            percentage=0,
            count=0,
            base=0,
            codes=[],
        )

    # Determine bottom codes
    if scale_min is None:
        scale_min = int(arr.min())

    bottom_codes = list(range(scale_min, scale_min + n_boxes))

    # Calculate
    bottom_mask = np.isin(arr, bottom_codes)
    count = int(bottom_mask.sum())
    total_weight = w.sum()
    percentage = (w[bottom_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    is_weighted = weights is not None
    weighted_base = float(total_weight) if is_weighted else None

    name = "Bottom Box" if n_boxes == 1 else f"Bottom {n_boxes} Box"

    return BoxResult(
        name=name,
        percentage=float(percentage),
        count=count,
        base=len(arr),
        codes=bottom_codes,
        weighted_base=weighted_base,
        is_weighted=is_weighted,
    )


def satisfaction_summary(
    data: Union[pd.Series, np.ndarray],
    scale: Literal['1-5', '1-7', '1-10'] = '1-5',
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculate common satisfaction metrics.

    Returns Top 2 Box, Bottom 2 Box, Mean, and distribution.

    Parameters
    ----------
    data : array-like
        Satisfaction scores.
    scale : str
        Scale type.
    weights : array-like, optional
        Weight values.

    Returns
    -------
    dict
        Comprehensive satisfaction metrics.
    """
    arr = np.array(data, dtype=float)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid_mask]
    else:
        w = np.ones(len(arr))

    # Scale parameters
    scale_params = {
        '1-5': {'min': 1, 'max': 5},
        '1-7': {'min': 1, 'max': 7},
        '1-10': {'min': 1, 'max': 10},
    }
    params = scale_params.get(scale, {'min': 1, 'max': 5})

    t2b = top_box(arr, n_boxes=2, scale_max=params['max'], weights=w if weights is not None else None)
    b2b = bottom_box(arr, n_boxes=2, scale_min=params['min'], weights=w if weights is not None else None)

    # Weighted mean
    mean = np.average(arr, weights=w) if len(arr) > 0 else 0

    return {
        'mean': float(mean),
        'top_2_box': t2b.percentage,
        'bottom_2_box': b2b.percentage,
        'base': len(arr),
        'scale': scale,
    }


# =============================================================================
# NETS (CATEGORY GROUPINGS)
# =============================================================================

@dataclass
class Net:
    """Definition of a net (category grouping)."""
    name: str
    codes: List[int]
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.name


@dataclass
class NetResult:
    """Result of net calculation."""
    name: str
    label: str
    percentage: float
    count: int
    codes: List[int]
    base: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'label': self.label,
            'percentage': self.percentage,
            'count': self.count,
            'codes': self.codes,
            'base': self.base,
        }


def calculate_net(
    data: Union[pd.Series, np.ndarray],
    codes: List[int],
    name: str = "Net",
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> NetResult:
    """
    Calculate a net (grouped categories) percentage.

    Parameters
    ----------
    data : array-like
        Response data.
    codes : list of int
        Codes to include in the net.
    name : str
        Net name.
    weights : array-like, optional
        Weight values.

    Returns
    -------
    NetResult

    Example
    -------
    >>> # Net of "Agree" responses (codes 4 and 5)
    >>> data = [1, 2, 3, 4, 5, 4, 5, 3, 4, 5]
    >>> result = calculate_net(data, codes=[4, 5], name="Net Agree")
    >>> print(f"{result.name}: {result.percentage:.1f}%")
    """
    arr = np.array(data, dtype=float)
    valid_mask = ~np.isnan(arr)
    arr = arr[valid_mask]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid_mask]
    else:
        w = np.ones(len(arr))

    if len(arr) == 0:
        return NetResult(
            name=name,
            label=name,
            percentage=0,
            count=0,
            codes=codes,
            base=0,
        )

    net_mask = np.isin(arr, codes)
    count = int(net_mask.sum())
    total_weight = w.sum()
    percentage = (w[net_mask].sum() / total_weight * 100) if total_weight > 0 else 0

    return NetResult(
        name=name,
        label=name,
        percentage=float(percentage),
        count=count,
        codes=codes,
        base=len(arr),
    )


def calculate_nets(
    data: Union[pd.Series, np.ndarray],
    nets: List[Net],
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> List[NetResult]:
    """
    Calculate multiple nets.

    Parameters
    ----------
    data : array-like
        Response data.
    nets : list of Net
        Net definitions.
    weights : array-like, optional
        Weight values.

    Returns
    -------
    list of NetResult
    """
    return [
        calculate_net(data, net.codes, net.name, weights)
        for net in nets
    ]


# Common net presets
LIKERT_5_NETS = [
    Net("Top 2 Box", [4, 5], "Agree/Strongly Agree"),
    Net("Bottom 2 Box", [1, 2], "Disagree/Strongly Disagree"),
    Net("Neutral", [3], "Neither Agree nor Disagree"),
]

LIKERT_7_NETS = [
    Net("Top 2 Box", [6, 7], "Agree/Strongly Agree"),
    Net("Top 3 Box", [5, 6, 7], "Somewhat Agree+"),
    Net("Bottom 2 Box", [1, 2], "Disagree/Strongly Disagree"),
    Net("Bottom 3 Box", [1, 2, 3], "Somewhat Disagree+"),
]

SATISFACTION_5_NETS = [
    Net("Satisfied", [4, 5], "Satisfied/Very Satisfied"),
    Net("Dissatisfied", [1, 2], "Dissatisfied/Very Dissatisfied"),
]


# =============================================================================
# INDEX CALCULATIONS
# =============================================================================

@dataclass
class IndexResult:
    """Result of index calculation."""
    value: float
    actual: float
    benchmark: float
    name: str = "Index"

    @property
    def direction(self) -> str:
        """Index direction."""
        if self.value > 105:
            return "above"
        elif self.value < 95:
            return "below"
        return "average"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'index': self.value,
            'actual': self.actual,
            'benchmark': self.benchmark,
            'direction': self.direction,
        }

    def __repr__(self) -> str:
        return f"IndexResult({self.name}: {self.value:.0f})"


def calculate_index(
    actual: float,
    benchmark: float,
    name: str = "Index",
) -> IndexResult:
    """
    Calculate index value (actual / benchmark * 100).

    Parameters
    ----------
    actual : float
        Actual value.
    benchmark : float
        Benchmark/reference value.
    name : str
        Index name.

    Returns
    -------
    IndexResult

    Example
    -------
    >>> result = calculate_index(actual=45.5, benchmark=40.0)
    >>> print(f"Index: {result.value:.0f}")  # 114
    """
    if benchmark == 0:
        index_value = 0
    else:
        index_value = (actual / benchmark) * 100

    return IndexResult(
        value=float(index_value),
        actual=float(actual),
        benchmark=float(benchmark),
        name=name,
    )


def index_vs_total(
    data: Union[pd.Series, np.ndarray],
    segment_mask: Union[pd.Series, np.ndarray],
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
    name: str = "Index vs Total",
) -> IndexResult:
    """
    Calculate index of segment mean vs total mean.

    Parameters
    ----------
    data : array-like
        Numeric data.
    segment_mask : array-like
        Boolean mask for segment.
    weights : array-like, optional
        Weight values.
    name : str
        Index name.

    Returns
    -------
    IndexResult
    """
    arr = np.array(data, dtype=float)
    mask = np.array(segment_mask, dtype=bool)

    if weights is not None:
        w = np.array(weights, dtype=float)
    else:
        w = np.ones(len(arr))

    # Remove NaN
    valid = ~np.isnan(arr)
    arr_valid = arr[valid]
    w_valid = w[valid]
    mask_valid = mask[valid]

    # Total mean
    total_mean = np.average(arr_valid, weights=w_valid) if len(arr_valid) > 0 else 0

    # Segment mean
    segment_data = arr_valid[mask_valid]
    segment_weights = w_valid[mask_valid]
    segment_mean = np.average(segment_data, weights=segment_weights) if len(segment_data) > 0 else 0

    return calculate_index(segment_mean, total_mean, name)


# =============================================================================
# GAP ANALYSIS
# =============================================================================

@dataclass
class GapResult:
    """Result of gap analysis."""
    item: str
    importance: float
    performance: float
    gap: float
    priority: str

    @property
    def quadrant(self) -> str:
        """Determine priority quadrant."""
        if self.importance >= 50 and self.performance < 50:
            return "Concentrate Here"
        elif self.importance >= 50 and self.performance >= 50:
            return "Keep Up Good Work"
        elif self.importance < 50 and self.performance < 50:
            return "Low Priority"
        else:
            return "Possible Overkill"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'item': self.item,
            'importance': self.importance,
            'performance': self.performance,
            'gap': self.gap,
            'priority': self.priority,
            'quadrant': self.quadrant,
        }


def gap_analysis(
    importance_scores: Dict[str, float],
    performance_scores: Dict[str, float],
) -> List[GapResult]:
    """
    Perform Importance-Performance Gap Analysis.

    Parameters
    ----------
    importance_scores : dict
        {item_name: importance_score}
    performance_scores : dict
        {item_name: performance_score}

    Returns
    -------
    list of GapResult
        Sorted by gap size (largest first).

    Example
    -------
    >>> importance = {'Price': 85, 'Quality': 90, 'Service': 75}
    >>> performance = {'Price': 60, 'Quality': 85, 'Service': 80}
    >>> gaps = gap_analysis(importance, performance)
    >>> for g in gaps:
    ...     print(f"{g.item}: Gap={g.gap:.1f}")
    """
    results = []

    for item in importance_scores:
        if item in performance_scores:
            imp = importance_scores[item]
            perf = performance_scores[item]
            gap = imp - perf

            # Priority based on gap and importance
            if gap > 15 and imp > 70:
                priority = "High"
            elif gap > 10 or imp > 80:
                priority = "Medium"
            else:
                priority = "Low"

            results.append(GapResult(
                item=item,
                importance=float(imp),
                performance=float(perf),
                gap=float(gap),
                priority=priority,
            ))

    # Sort by gap (largest first)
    results.sort(key=lambda x: x.gap, reverse=True)
    return results


# =============================================================================
# DATASET INTEGRATION
# =============================================================================

def mrx_summary(
    dataset: "DataSet",
    variable: str,
    weight: Optional[str] = None,
    scale: Literal['1-5', '1-7', '1-10', '0-10'] = '1-5',
) -> Dict[str, Any]:
    """
    Calculate comprehensive MRX metrics for a variable.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variable : str
        Variable to analyze.
    weight : str, optional
        Weight variable.
    scale : str
        Scale type.

    Returns
    -------
    dict
        Comprehensive metrics including mean, T2B, B2B, NPS (if applicable).
    """
    data = dataset.data[variable]
    weights = dataset.data[weight] if weight else None

    arr = np.array(data, dtype=float)
    valid = ~np.isnan(arr)
    arr = arr[valid]

    if weights is not None:
        w = np.array(weights, dtype=float)[valid]
    else:
        w = np.ones(len(arr))

    # Basic stats
    mean = np.average(arr, weights=w) if len(arr) > 0 else 0
    std = np.sqrt(np.average((arr - mean)**2, weights=w)) if len(arr) > 0 else 0

    # Scale parameters
    scale_params = {
        '1-5': {'min': 1, 'max': 5},
        '1-7': {'min': 1, 'max': 7},
        '1-10': {'min': 1, 'max': 10},
        '0-10': {'min': 0, 'max': 10},
    }
    params = scale_params.get(scale, {'min': 1, 'max': 5})

    result = {
        'variable': variable,
        'label': dataset.label(variable),
        'base': len(arr),
        'mean': float(mean),
        'std': float(std),
        'scale': scale,
    }

    # Top/Bottom Box
    t2b = top_box(arr, n_boxes=2, scale_max=params['max'], weights=w if weights is not None else None)
    b2b = bottom_box(arr, n_boxes=2, scale_min=params['min'], weights=w if weights is not None else None)
    result['top_2_box'] = t2b.percentage
    result['bottom_2_box'] = b2b.percentage

    # NPS for 0-10 or 1-10 scales
    if scale in ['0-10', '1-10']:
        nps_result = calculate_nps(arr, w if weights is not None else None, scale=scale)
        result['nps'] = nps_result.nps
        result['promoters_pct'] = nps_result.promoters_pct
        result['detractors_pct'] = nps_result.detractors_pct

    return result
