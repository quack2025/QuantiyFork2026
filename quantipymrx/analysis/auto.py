"""
QuantipyMRX - Automatic Analysis

Automated analysis features for market research:
- Auto-detect variable types and demographics
- Auto-generate frequencies and crosstabs
- Auto-summary with key insights
- Smart banner detection
- Battery/grid grouping
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, TYPE_CHECKING
import re
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult


# =============================================================================
# VARIABLE CLASSIFICATION
# =============================================================================

@dataclass
class VariableClassification:
    """Classification of a variable."""
    name: str
    category: str  # 'demographic', 'behavioral', 'attitudinal', 'other'
    subcategory: Optional[str] = None  # 'gender', 'age', 'region', etc.
    is_scale: bool = False
    scale_type: Optional[str] = None  # 'likert', 'nps', 'satisfaction'
    is_banner_candidate: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'category': self.category,
            'subcategory': self.subcategory,
            'is_scale': self.is_scale,
            'scale_type': self.scale_type,
            'is_banner_candidate': self.is_banner_candidate,
            'confidence': self.confidence,
        }


# Patterns for demographic detection
DEMOGRAPHIC_PATTERNS = {
    'gender': [
        r'\bgender\b', r'\bsex\b', r'\bgenero\b', r'\bsexo\b',
        r'\bmale\b', r'\bfemale\b', r'\bhombre\b', r'\bmujer\b',
    ],
    'age': [
        r'\bage\b', r'\bedad\b', r'\bage.?group\b', r'\bgrupo.?edad\b',
        r'\byears?\s*old\b', r'\banos\b',
    ],
    'region': [
        r'\bregion\b', r'\barea\b', r'\bzone\b', r'\bzona\b',
        r'\bgeograph', r'\blocation\b', r'\bubicacion\b',
        r'\bstate\b', r'\bestado\b', r'\bcity\b', r'\bciudad\b',
    ],
    'income': [
        r'\bincome\b', r'\bingresos?\b', r'\bsalary\b', r'\bsalario\b',
        r'\bearnings?\b', r'\bsueldo\b',
    ],
    'education': [
        r'\beducation\b', r'\beducacion\b', r'\bschool\b', r'\bescuela\b',
        r'\bdegree\b', r'\btitulo\b', r'\buniversity\b', r'\buniversidad\b',
    ],
    'occupation': [
        r'\boccupation\b', r'\bocupacion\b', r'\bjob\b', r'\btrabajo\b',
        r'\bemployment\b', r'\bempleo\b', r'\bprofession\b',
    ],
    'marital': [
        r'\bmarital\b', r'\bcivil\b', r'\bmarried\b', r'\bcasado\b',
        r'\bsingle\b', r'\bsoltero\b',
    ],
    'household': [
        r'\bhousehold\b', r'\bhogar\b', r'\bfamily\b', r'\bfamilia\b',
        r'\bchildren\b', r'\bhijos\b', r'\bkids\b',
    ],
}

# Patterns for scale detection
SCALE_PATTERNS = {
    'likert': [
        r'\bagree\b', r'\bdisagree\b', r'\bsatisf', r'\blikely\b',
        r'\bimportant\b', r'\bfrequen', r'\bacuerdo\b', r'\bdesacuerdo\b',
    ],
    'nps': [
        r'\brecommend\b', r'\brecomendar\b', r'\bpromoter\b',
        r'\blikelihood.+recommend\b',
    ],
    'satisfaction': [
        r'\bsatisf', r'\bhappy\b', r'\bpleased\b', r'\bcontent\b',
    ],
}


def classify_variable(
    dataset: "DataSet",
    variable: str,
) -> VariableClassification:
    """
    Automatically classify a variable.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variable : str
        Variable name.

    Returns
    -------
    VariableClassification
    """
    meta = dataset.get_variable_meta(variable)
    if meta is None:
        return VariableClassification(name=variable, category='other')

    label = meta.get_label(dataset.text_key).lower()
    name_lower = variable.lower()
    search_text = f"{name_lower} {label}"

    category = 'other'
    subcategory = None
    is_banner_candidate = False
    confidence = 0.0

    # Check demographic patterns
    for demo_type, patterns in DEMOGRAPHIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, search_text, re.IGNORECASE):
                category = 'demographic'
                subcategory = demo_type
                is_banner_candidate = True
                confidence = 0.8
                break
        if category == 'demographic':
            break

    # Check if it's a scale
    is_scale = False
    scale_type = None

    if meta.is_categorical and meta.values:
        n_codes = len(meta.codes)

        # Check scale patterns in labels
        for scale_name, patterns in SCALE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    is_scale = True
                    scale_type = scale_name
                    if category == 'other':
                        category = 'attitudinal'
                    break
            if is_scale:
                break

        # Heuristic: 5 or 7 point scales are likely Likert
        if not is_scale and n_codes in [5, 7, 10, 11]:
            codes = sorted(meta.codes)
            if codes == list(range(1, n_codes + 1)) or codes == list(range(0, n_codes)):
                is_scale = True
                scale_type = 'likert' if n_codes in [5, 7] else 'rating'
                if category == 'other':
                    category = 'attitudinal'
                confidence = 0.6

        # Banner candidate: few categories, no scale
        if not is_scale and n_codes <= 8:
            is_banner_candidate = True
            confidence = max(confidence, 0.5)

    return VariableClassification(
        name=variable,
        category=category,
        subcategory=subcategory,
        is_scale=is_scale,
        scale_type=scale_type,
        is_banner_candidate=is_banner_candidate,
        confidence=confidence,
    )


def classify_all_variables(dataset: "DataSet") -> Dict[str, VariableClassification]:
    """Classify all variables in the dataset."""
    return {
        var: classify_variable(dataset, var)
        for var in dataset.variables
    }


# =============================================================================
# AUTO-DETECT BANNERS (DEMOGRAPHICS)
# =============================================================================

@dataclass
class BannerSpec:
    """Specification for a banner (crosstab columns)."""
    name: str
    variables: List[str]
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'variables': self.variables,
            'description': self.description,
        }


def detect_banners(
    dataset: "DataSet",
    max_categories: int = 10,
    min_cell_size: int = 30,
) -> BannerSpec:
    """
    Auto-detect demographic variables suitable for banners.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    max_categories : int
        Maximum categories for a banner variable.
    min_cell_size : int
        Minimum expected cell size.

    Returns
    -------
    BannerSpec
        Recommended banner specification.
    """
    classifications = classify_all_variables(dataset)
    banner_vars = []

    for var, classification in classifications.items():
        if not classification.is_banner_candidate:
            continue

        meta = dataset.get_variable_meta(var)
        if meta is None or not meta.is_categorical:
            continue

        n_codes = len(meta.codes)

        # Check category count
        if n_codes > max_categories or n_codes < 2:
            continue

        # Check minimum cell sizes
        value_counts = dataset.data[var].value_counts()
        if value_counts.min() < min_cell_size:
            continue

        # Prioritize by category type
        priority = 0
        if classification.subcategory == 'gender':
            priority = 10
        elif classification.subcategory == 'age':
            priority = 9
        elif classification.subcategory == 'region':
            priority = 8
        elif classification.subcategory in ['income', 'education']:
            priority = 7
        elif classification.category == 'demographic':
            priority = 5
        else:
            priority = 1

        banner_vars.append((var, priority, classification))

    # Sort by priority and take top variables
    banner_vars.sort(key=lambda x: x[1], reverse=True)
    selected = [v[0] for v in banner_vars[:5]]  # Max 5 banner variables

    return BannerSpec(
        name="Auto-detected Demographics",
        variables=selected,
        description="Automatically detected demographic variables for cross-tabulation",
    )


# =============================================================================
# AUTO-GENERATE ANALYSIS
# =============================================================================

@dataclass
class AutoAnalysisResult:
    """Result of automatic analysis."""
    dataset_summary: Dict[str, Any]
    variable_classifications: Dict[str, VariableClassification]
    banner: BannerSpec
    frequencies: Dict[str, "FrequencyResult"]
    crosstabs: Dict[str, Dict[str, "CrosstabResult"]]
    key_insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_summary': self.dataset_summary,
            'variable_classifications': {
                k: v.to_dict() for k, v in self.variable_classifications.items()
            },
            'banner': self.banner.to_dict(),
            'key_insights': self.key_insights,
        }


def auto_analyze(
    dataset: "DataSet",
    weight: Optional[str] = None,
    include_crosstabs: bool = True,
    max_crosstab_vars: int = 20,
    progress_callback: Optional[callable] = None,
) -> AutoAnalysisResult:
    """
    Perform automatic analysis of the dataset.

    Parameters
    ----------
    dataset : DataSet
        The dataset to analyze.
    weight : str, optional
        Weight variable.
    include_crosstabs : bool
        Whether to generate crosstabs.
    max_crosstab_vars : int
        Maximum variables for crosstabs.
    progress_callback : callable, optional
        Function to call with progress updates.

    Returns
    -------
    AutoAnalysisResult
        Comprehensive analysis results.

    Example
    -------
    >>> ds = DataSet.from_spss("survey.sav")
    >>> results = auto_analyze(ds)
    >>> print(results.key_insights)
    """
    from quantipymrx.analysis.frequency import frequency
    from quantipymrx.analysis.crosstab import crosstab

    def update_progress(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Step 1: Dataset summary
    update_progress("Generating dataset summary...")
    summary = generate_dataset_summary(dataset)

    # Step 2: Classify variables
    update_progress("Classifying variables...")
    classifications = classify_all_variables(dataset)

    # Step 3: Detect banners
    update_progress("Detecting banner variables...")
    banner = detect_banners(dataset)

    # Step 4: Generate frequencies
    update_progress("Calculating frequencies...")
    freq_results = {}
    categorical_vars = dataset.singles() + dataset.multis()

    for var in categorical_vars:
        try:
            freq_results[var] = frequency(dataset, var, weight=weight)
        except Exception:
            pass

    # Step 5: Generate crosstabs
    crosstab_results: Dict[str, Dict[str, "CrosstabResult"]] = {}

    if include_crosstabs and banner.variables:
        update_progress("Calculating crosstabs...")

        # Select variables for crosstabs (non-banner categoricals)
        crosstab_vars = [
            v for v in categorical_vars
            if v not in banner.variables
        ][:max_crosstab_vars]

        for y_var in banner.variables:
            crosstab_results[y_var] = {}
            for x_var in crosstab_vars:
                try:
                    crosstab_results[y_var][x_var] = crosstab(
                        dataset, x=x_var, y=y_var, weight=weight
                    )
                except Exception:
                    pass

    # Step 6: Generate insights
    update_progress("Generating insights...")
    insights = generate_insights(
        dataset, classifications, freq_results, crosstab_results
    )

    return AutoAnalysisResult(
        dataset_summary=summary,
        variable_classifications=classifications,
        banner=banner,
        frequencies=freq_results,
        crosstabs=crosstab_results,
        key_insights=insights,
    )


def generate_dataset_summary(dataset: "DataSet") -> Dict[str, Any]:
    """Generate summary statistics for the dataset."""
    classifications = classify_all_variables(dataset)

    # Count by category
    category_counts = {}
    for c in classifications.values():
        category_counts[c.category] = category_counts.get(c.category, 0) + 1

    # Count by type
    type_counts = {
        'single': len(dataset.singles()),
        'multi': len(dataset.multis()),
        'numeric': len(dataset.numerics()),
        'text': len(dataset.texts()),
    }

    # Demographics detected
    demographics = [
        c.name for c in classifications.values()
        if c.category == 'demographic'
    ]

    # Scales detected
    scales = [
        c.name for c in classifications.values()
        if c.is_scale
    ]

    return {
        'name': dataset.name,
        'n_cases': dataset.n_cases,
        'n_variables': dataset.n_variables,
        'variable_types': type_counts,
        'variable_categories': category_counts,
        'demographics_detected': demographics,
        'scales_detected': scales,
    }


def generate_insights(
    dataset: "DataSet",
    classifications: Dict[str, VariableClassification],
    frequencies: Dict[str, "FrequencyResult"],
    crosstabs: Dict[str, Dict[str, "CrosstabResult"]],
) -> List[str]:
    """Generate key insights from the analysis."""
    insights = []

    # Dataset size insight
    insights.append(
        f"Dataset contains {dataset.n_cases:,} responses across {dataset.n_variables} variables."
    )

    # Demographics insight
    demos = [c.name for c in classifications.values() if c.category == 'demographic']
    if demos:
        insights.append(
            f"Detected {len(demos)} demographic variables: {', '.join(demos[:5])}"
            + ("..." if len(demos) > 5 else ".")
        )

    # Top frequency insights
    for var, freq in list(frequencies.items())[:3]:
        if freq.table is not None and len(freq.table) > 0:
            top_response = freq.table.iloc[0]
            if 'Percent' in freq.table.columns:
                pct = freq.table['Percent'].iloc[0]
                insights.append(
                    f"Most common response for '{var}': {freq.table.index[0]} ({pct:.1f}%)"
                )

    # Significance insights
    significant_findings = []
    for banner_var, xtabs in crosstabs.items():
        for x_var, xtab in xtabs.items():
            if xtab.chi2_pvalue is not None and xtab.chi2_pvalue < 0.05:
                significant_findings.append((x_var, banner_var, xtab.chi2_pvalue))

    if significant_findings:
        # Sort by p-value
        significant_findings.sort(key=lambda x: x[2])
        top_sig = significant_findings[:3]
        for x, y, p in top_sig:
            insights.append(
                f"Significant relationship found between '{x}' and '{y}' (p={p:.4f})"
            )

    return insights


# =============================================================================
# BATTERY/GRID DETECTION
# =============================================================================

@dataclass
class BatteryGroup:
    """A group of related variables (battery/grid question)."""
    name: str
    variables: List[str]
    common_prefix: str
    n_items: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'variables': self.variables,
            'common_prefix': self.common_prefix,
            'n_items': self.n_items,
        }


def detect_batteries(
    dataset: "DataSet",
    min_items: int = 3,
) -> List[BatteryGroup]:
    """
    Detect battery/grid questions by naming patterns.

    Looks for patterns like:
    - q1_1, q1_2, q1_3 (underscore + number)
    - q1a, q1b, q1c (letter suffix)
    - q1_statement1, q1_statement2 (common prefix)

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    min_items : int
        Minimum items to be considered a battery.

    Returns
    -------
    list of BatteryGroup
    """
    variables = dataset.variables
    batteries = []

    # Pattern 1: prefix_number (q1_1, q1_2, ...)
    prefix_groups: Dict[str, List[str]] = {}
    pattern1 = re.compile(r'^(.+?)_(\d+)$')

    for var in variables:
        match = pattern1.match(var)
        if match:
            prefix = match.group(1)
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(var)

    for prefix, vars_list in prefix_groups.items():
        if len(vars_list) >= min_items:
            # Sort by number
            vars_list.sort(key=lambda v: int(pattern1.match(v).group(2)))
            batteries.append(BatteryGroup(
                name=prefix,
                variables=vars_list,
                common_prefix=prefix,
                n_items=len(vars_list),
            ))

    # Pattern 2: prefix + letter (q1a, q1b, ...)
    prefix_groups2: Dict[str, List[str]] = {}
    pattern2 = re.compile(r'^(.+?)([a-z])$', re.IGNORECASE)

    for var in variables:
        if any(var in b.variables for b in batteries):
            continue  # Skip if already in a battery

        match = pattern2.match(var)
        if match:
            prefix = match.group(1)
            if prefix not in prefix_groups2:
                prefix_groups2[prefix] = []
            prefix_groups2[prefix].append(var)

    for prefix, vars_list in prefix_groups2.items():
        if len(vars_list) >= min_items:
            # Sort alphabetically by suffix
            vars_list.sort()
            batteries.append(BatteryGroup(
                name=prefix,
                variables=vars_list,
                common_prefix=prefix,
                n_items=len(vars_list),
            ))

    return batteries


# =============================================================================
# QUICK ANALYSIS FUNCTIONS
# =============================================================================

def quick_summary(dataset: "DataSet") -> str:
    """
    Generate a quick text summary of the dataset.

    Returns
    -------
    str
        Human-readable summary.
    """
    summary = generate_dataset_summary(dataset)

    lines = [
        f"# Dataset Summary: {summary['name']}",
        "",
        f"**Total Responses:** {summary['n_cases']:,}",
        f"**Total Variables:** {summary['n_variables']}",
        "",
        "## Variable Types",
        f"- Single choice: {summary['variable_types']['single']}",
        f"- Multiple choice: {summary['variable_types']['multi']}",
        f"- Numeric: {summary['variable_types']['numeric']}",
        f"- Text/Open-ended: {summary['variable_types']['text']}",
        "",
    ]

    if summary['demographics_detected']:
        lines.extend([
            "## Demographics Detected",
            ", ".join(summary['demographics_detected']),
            "",
        ])

    if summary['scales_detected']:
        lines.extend([
            "## Scale Variables Detected",
            ", ".join(summary['scales_detected'][:10]),
            "",
        ])

    return "\n".join(lines)


def suggest_analyses(dataset: "DataSet") -> List[Dict[str, Any]]:
    """
    Suggest relevant analyses based on data structure.

    Returns
    -------
    list of dict
        Suggested analyses with descriptions.
    """
    classifications = classify_all_variables(dataset)
    suggestions = []

    # Demographic crosstabs
    demos = [c.name for c in classifications.values() if c.category == 'demographic']
    if demos:
        suggestions.append({
            'type': 'crosstab',
            'description': f"Cross-tabulate key questions by demographics ({', '.join(demos[:3])})",
            'priority': 'high',
        })

    # Scale analysis
    scales = [c.name for c in classifications.values() if c.is_scale]
    if scales:
        suggestions.append({
            'type': 'scale_analysis',
            'description': f"Calculate Top 2 Box / Bottom 2 Box for {len(scales)} scale variables",
            'variables': scales[:10],
            'priority': 'high',
        })

    # NPS
    nps_candidates = [
        c.name for c in classifications.values()
        if c.scale_type == 'nps'
    ]
    if nps_candidates:
        suggestions.append({
            'type': 'nps',
            'description': f"Calculate NPS for: {', '.join(nps_candidates)}",
            'variables': nps_candidates,
            'priority': 'high',
        })

    # Battery analysis
    batteries = detect_batteries(dataset)
    if batteries:
        suggestions.append({
            'type': 'battery_analysis',
            'description': f"Analyze {len(batteries)} battery/grid questions",
            'batteries': [b.name for b in batteries],
            'priority': 'medium',
        })

    # Correlation analysis for numeric
    numerics = dataset.numerics()
    if len(numerics) >= 2:
        suggestions.append({
            'type': 'correlation',
            'description': f"Correlation analysis for {len(numerics)} numeric variables",
            'priority': 'medium',
        })

    return suggestions
