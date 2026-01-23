"""
QuantipyMRX - Modern Survey Data Analysis Engine

A modern, type-safe library for survey data analysis and market research.
Built for speed, simplicity, and integration with AI-powered analytics.

Example
-------
>>> from quantipymrx import DataSet
>>>
>>> # Load SPSS file
>>> ds = DataSet.from_spss("survey.sav")
>>> print(ds.info())
>>>
>>> # Calculate frequency
>>> freq = ds.frequency("q1")
>>> print(freq.table)
>>>
>>> # Calculate crosstab
>>> xtab = ds.crosstab("satisfaction", "gender")
>>> print(xtab.col_pct)
>>> print(f"Chi-square p-value: {xtab.chi2_pvalue}")
>>>
>>> # Export to Excel
>>> xtab.to_excel("results.xlsx")
"""

__version__ = "0.1.0"

# Core classes
from quantipymrx.core.dataset import DataSet
from quantipymrx.core.metadata import (
    DataSetMeta,
    DataSetInfo,
    VariableMeta,
    VariableType,
    ValueLabel,
    ArrayMeta,
    TextKey,
)

# I/O functions
from quantipymrx.io.spss import (
    read_spss,
    write_spss,
    get_spss_info,
)

# Analysis functions
from quantipymrx.analysis.frequency import (
    frequency,
    frequencies_batch,
    FrequencyResult,
)
from quantipymrx.analysis.crosstab import (
    crosstab,
    crosstab_batch,
    CrosstabResult,
)
from quantipymrx.analysis.significance import (
    z_test_proportions,
    t_test_means,
    chi_square_test,
    anova_oneway,
    correlation,
    correlation_matrix,
)
from quantipymrx.analysis.mrx import (
    calculate_nps,
    top_box,
    bottom_box,
    calculate_net,
    mrx_summary,
    NPSResult,
    BoxResult,
)
from quantipymrx.analysis.auto import (
    auto_analyze,
    classify_variable,
    detect_banners,
    quick_summary,
    suggest_analyses,
    AutoAnalysisResult,
)

# Service layer
from quantipymrx.service.analysis_service import AnalysisService

__all__ = [
    # Version
    "__version__",
    # Core
    "DataSet",
    "DataSetMeta",
    "DataSetInfo",
    "VariableMeta",
    "VariableType",
    "ValueLabel",
    "ArrayMeta",
    "TextKey",
    # I/O
    "read_spss",
    "write_spss",
    "get_spss_info",
    # Analysis - Frequency/Crosstab
    "frequency",
    "frequencies_batch",
    "FrequencyResult",
    "crosstab",
    "crosstab_batch",
    "CrosstabResult",
    # Analysis - Significance
    "z_test_proportions",
    "t_test_means",
    "chi_square_test",
    "anova_oneway",
    "correlation",
    "correlation_matrix",
    # Analysis - MRX
    "calculate_nps",
    "top_box",
    "bottom_box",
    "calculate_net",
    "mrx_summary",
    "NPSResult",
    "BoxResult",
    # Analysis - Auto
    "auto_analyze",
    "classify_variable",
    "detect_banners",
    "quick_summary",
    "suggest_analyses",
    "AutoAnalysisResult",
    # Service
    "AnalysisService",
]
