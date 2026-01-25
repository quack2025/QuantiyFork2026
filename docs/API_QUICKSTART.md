# QuantipyMRX - Quick Start Guide for talk2data

## Overview

QuantipyMRX is a modern, type-safe library for survey data analysis. It provides a clean API for loading SPSS files, calculating frequencies and crosstabs, and exporting results for use in talk2data.

## Installation

```bash
pip install pandas numpy scipy pyreadstat pydantic ftfy
```

## Quick Start

### Loading Data

```python
from quantipymrx import DataSet

# Load SPSS file with Spanish labels
ds = DataSet.from_spss("survey.sav", text_key="es-ES")

# Get basic info
print(ds.info())
# DataSet: survey
# Cases: 1000
# Variables: 50
#   - Single choice: 35
#   - Multiple choice: 5
#   - Numeric: 8
#   - Text: 2
```

### Accessing Variables

```python
# List all variables
print(ds.variables)

# Get variable label
print(ds.label("q1"))  # "CLASIFICACION EDAD"

# Get value labels
print(ds.value_labels("q1"))
# {1: "Menor de 24 anos", 2: "Entre 24 y 35 anos", ...}

# Check variable type
print(ds.var_type("q1"))  # VariableType.SINGLE
print(ds.is_categorical("q1"))  # True
```

### Frequency Analysis

```python
# Calculate frequency
freq = ds.frequency("q1")

# Access results
print(freq.table)        # DataFrame with counts and percentages
print(freq.base)         # Total base
print(freq.mean)         # Mean (for numeric/categorical)
print(freq.percentages)  # Percentage series

# Export
freq.to_excel("q1_frequency.xlsx")

# JSON for API
freq_dict = freq.to_dict()
```

### Crosstab Analysis

```python
# Calculate crosstab
xtab = ds.crosstab("q1", "gender")

# Access results
print(xtab.col_pct)      # Column percentages
print(xtab.counts)       # Raw counts
print(xtab.chi2)         # Chi-square statistic
print(xtab.chi2_pvalue)  # P-value
print(xtab.significance) # Significance letters (A, B, C...)

# Export
xtab.to_excel("q1_by_gender.xlsx")

# JSON for API
xtab_dict = xtab.to_dict()
```

### Weighted Analysis

```python
# Set default weight
ds.weight = "weight_variable"

# Or specify per analysis
freq = ds.frequency("q1", weight="weight_variable")
xtab = ds.crosstab("q1", "gender", weight="weight_variable")
```

### Batch Operations

```python
# Frequencies for all categorical variables
all_freqs = ds.frequencies_all()

# Frequencies for specific variables
selected_freqs = ds.frequencies_all(variables=["q1", "q2", "q3"])

# Crosstabs against a banner
from quantipymrx import crosstab_batch
xtabs = crosstab_batch(ds, x_vars=["q1", "q2", "q3"], y_var="gender")
```

### Data Filtering

```python
# Filter by condition
filtered = ds.filter("age > 30")

# Filter by boolean mask
filtered = ds.filter(ds["gender"] == 1)

# Analysis on filtered data
freq = filtered.frequency("q1")
```

### Supported Languages (i18n)

QuantipyMRX supports 55+ languages/regions:

```python
# Spanish
ds = DataSet.from_spss("survey.sav", text_key="es-ES")

# Portuguese (Brazil)
ds = DataSet.from_spss("survey.sav", text_key="pt-BR")

# German
ds = DataSet.from_spss("survey.sav", text_key="de-DE")

# Chinese (Simplified)
ds = DataSet.from_spss("survey.sav", text_key="zh-CN")
```

### Available Text Keys

| Region | Languages |
|--------|-----------|
| Europe | en-GB, es-ES, pt-PT, de-DE, fr-FR, it-IT, nl-NL, pl-PL, ru-RU |
| North America | en-US, es-MX, fr-CA |
| South America | es-AR, es-CO, es-CL, es-PE, pt-BR |
| Asia | zh-CN, zh-TW, ja-JP, ko-KR, hi-IN, th-TH, vi-VN |
| Middle East | ar-SA, ar-AE, he-IL |

## API Reference

### DataSet

| Method | Description |
|--------|-------------|
| `from_spss(path, text_key)` | Load from SPSS file |
| `from_dataframe(df, name)` | Create from DataFrame |
| `frequency(var, weight)` | Calculate frequency |
| `crosstab(x, y, weight)` | Calculate crosstab |
| `frequencies_all(vars)` | Batch frequencies |
| `filter(condition)` | Filter data |
| `copy()` | Deep copy |
| `to_spss(path)` | Export to SPSS |
| `to_excel(path)` | Export to Excel |

### FrequencyResult

| Property | Description |
|----------|-------------|
| `table` | DataFrame with Value, Label, Count, Percent |
| `base` | Unweighted base |
| `weighted_base` | Weighted base (if weighted) |
| `mean` | Mean value |
| `std` | Standard deviation |
| `median` | Median value |
| `to_dict()` | Convert to dict for JSON |
| `to_excel(path)` | Export to Excel |

### CrosstabResult

| Property | Description |
|----------|-------------|
| `counts` | Count DataFrame |
| `col_pct` | Column percentages |
| `row_pct` | Row percentages |
| `total_pct` | Total percentages |
| `base` | Column bases |
| `chi2` | Chi-square statistic |
| `chi2_pvalue` | Chi-square p-value |
| `significance` | Significance letters |
| `to_dict()` | Convert to dict for JSON |
| `to_excel(path)` | Export to Excel |

## Integration with talk2data

For talk2data integration, use the `to_dict()` method to serialize results:

```python
import json
from quantipymrx import DataSet

ds = DataSet.from_spss("survey.sav", text_key="es-ES")

# Get frequency as JSON
freq = ds.frequency("q1")
json_result = json.dumps(freq.to_dict())

# Get crosstab as JSON
xtab = ds.crosstab("q1", "gender")
json_result = json.dumps(xtab.to_dict())
```

The `to_dict()` output includes all data needed for visualization and reporting in talk2data.
