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
    # Analysis
    "frequency",
    "frequencies_batch",
    "FrequencyResult",
    "crosstab",
    "crosstab_batch",
    "CrosstabResult",
]
