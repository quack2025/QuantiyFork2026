"""QuantipyMRX Analysis Module."""

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
    "frequency",
    "frequencies_batch",
    "FrequencyResult",
    "crosstab",
    "crosstab_batch",
    "CrosstabResult",
]
