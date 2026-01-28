# QuantipyMRX

Modern Python library for Market Research data analysis. A complete rewrite focused on Python 3.10+ compatibility, clean APIs, and international market research workflows.

## Features

- **SPSS Support**: Read/write `.sav` files using `pyreadstat` (no legacy dependencies)
- **Auto-Detection**: Intelligent detection of question types (awareness, scales, NPS, grids, demographics)
- **Multi-Language i18n**: Support for 55+ languages with automatic label detection
- **Statistical Analysis**: Frequencies, crosstabs, chi-square, t-tests, ANOVA, correlations
- **MRX Metrics**: NPS calculation, top/bottom box, nets, index scores, gap analysis
- **Excel Export**: Professional formatting with openpyxl
- **Google Sheets Export**: Direct API integration via gspread

## Installation

```bash
pip install -e .
```

### Required Dependencies
```bash
pip install pandas numpy scipy pyreadstat
```

### Optional Dependencies
```bash
pip install openpyxl  # For Excel export
pip install gspread   # For Google Sheets export
```

## Quick Start

### Load SPSS Data

```python
from quantipymrx import DataSet

# Load SPSS file with Spanish labels
ds = DataSet.from_spss("survey.sav", text_key='es-ES')

print(f"Cases: {ds.n_cases}")
print(f"Variables: {ds.n_variables}")
```

### Frequency Analysis

```python
# Single frequency
result = ds.frequency('Q1')
print(f"Base: {result.base}")
print(result.table)

# Batch frequencies
results = ds.frequencies(['Q1', 'Q2', 'Q3'])
```

### Crosstab with Chi-Square

```python
result = ds.crosstab('Q1', 'GENDER')
print(f"Chi-square: {result.chi2:.2f}, p={result.chi2_pvalue:.4f}")
print(result.col_pct)
```

### Auto-Detection

```python
from quantipymrx.analysis import auto_detect, process_all_groups

# Detect question patterns
spec = auto_detect(ds)
print(f"Banners: {len(spec.banners)}")
print(f"Variable Groups: {len(spec.variable_groups)}")

# Process all detected groups
results = process_all_groups(ds, spec)
```

### Export to Excel

```python
from quantipymrx.io import export_processing_results

export_processing_results(
    results,
    "output.xlsx",
    dataset_name=ds.name,
    case_count=ds.n_cases
)
```

### Export to Google Sheets

```python
from quantipymrx.io import GoogleSheetsExporter

exporter = GoogleSheetsExporter("credentials.json")
exporter.create_spreadsheet("Survey Results", results)
```

## Module Structure

```
quantipymrx/
├── __init__.py          # Main DataSet class
├── core/
│   ├── dataset.py       # DataSet implementation
│   └── metadata.py      # Metadata classes
├── analysis/
│   ├── frequency.py     # Frequency analysis
│   ├── crosstab.py      # Cross-tabulation
│   ├── significance.py  # Statistical tests
│   ├── mrx.py          # MRX metrics (NPS, nets, etc.)
│   ├── auto.py         # Basic auto-analysis
│   ├── auto_detect.py  # Pattern detection
│   └── grouping.py     # GroupingEngine
├── io/
│   ├── spss.py         # SPSS read/write
│   ├── excel_export.py # Excel export
│   └── gsheets_export.py # Google Sheets export
├── i18n/
│   └── languages.py    # 55+ language support
└── _compat/            # Compatibility layer
```

## Supported Question Types

The auto-detection system recognizes:

| Type | Description | Detection Pattern |
|------|-------------|-------------------|
| `AWARENESS` | Aided/unaided awareness | Binary Yes/No values |
| `SCALE` | Likert scales | Numeric 1-5, 1-7, 1-10 |
| `NPS` | Net Promoter Score | 0-10 scale |
| `GRID` | Matrix questions | Common stem with items |
| `RANKING` | Rank order | 1st, 2nd, 3rd mentions |
| `DEMOGRAPHIC` | Demographics | Age, gender, region patterns |
| `SINGLE` | Single choice | Categorical with labels |
| `MULTI` | Multiple choice | Dichotomous set |

## Statistical Functions

```python
from quantipymrx.analysis import (
    chi_square_test,
    z_test_proportions,
    t_test_means,
    anova_oneway,
    correlation_matrix
)

# Chi-square test
result = chi_square_test(contingency_table)
print(f"χ² = {result.statistic:.2f}, p = {result.p_value:.4f}")

# Z-test for proportions
result = z_test_proportions(0.45, 0.38, n1=500, n2=480)
print(f"Significant: {result.significant}")
```

## Python Compatibility

- **Minimum**: Python 3.10
- **Tested**: Python 3.10, 3.11, 3.12, 3.13
- **Uses**: Public scipy APIs only (no `_ttest_finish` or other private functions)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure code passes all tests:

```bash
python test_complete_verification.py
```
