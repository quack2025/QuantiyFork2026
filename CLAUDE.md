# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

**QuantipyMRX** is a modern Python library for Market Research data analysis. It was created as a clean rewrite of the legacy Quantipy3 project, focusing on Python 3.10+ compatibility and modern APIs.

## Architecture

### Main Module: `quantipymrx/`

```
quantipymrx/
├── __init__.py          # Exports DataSet class
├── core/
│   ├── dataset.py       # Main DataSet class
│   └── metadata.py      # VariableMeta, ValueMeta, DatasetMeta
├── analysis/
│   ├── frequency.py     # frequency(), FrequencyResult
│   ├── crosstab.py      # crosstab(), CrosstabResult
│   ├── significance.py  # chi_square_test, t_test, z_test, anova
│   ├── mrx.py          # NPS, top/bottom box, nets, gap analysis
│   ├── auto.py         # Basic auto-analysis utilities
│   ├── auto_detect.py  # PatternDetector, ProcessingSpec
│   └── grouping.py     # GroupingEngine, summaries
├── io/
│   ├── spss.py         # read_spss() using pyreadstat
│   ├── excel_export.py # ExcelExporter with openpyxl
│   └── gsheets_export.py # GoogleSheetsExporter with gspread
├── i18n/
│   └── languages.py    # 55+ language definitions
└── _compat/
    ├── __init__.py     # ensure_compatibility()
    ├── pandas_compat.py # Series.append, DataFrame.append backports
    ├── scipy_compat.py  # get_pval() without _ttest_finish
    └── numpy_compat.py  # NumPy 2.0 compatibility
```

### Key Design Decisions

1. **No Legacy Dependencies**: Uses `pyreadstat` for SPSS, not the broken legacy reader
2. **Public APIs Only**: Uses `scipy.stats.t.sf()` instead of `scipy._ttest_finish`
3. **i18n First**: Multi-language support built into core metadata
4. **Pattern Detection**: Auto-detects question types from variable names and value labels

## Common Tasks

### Running Tests

```bash
# Complete verification (10 tests)
python test_complete_verification.py

# Import verification
python test_imports.py
```

### Loading Data

```python
from quantipymrx import DataSet
ds = DataSet.from_spss("file.sav", text_key='es-ES')
```

### Analysis Flow

```python
from quantipymrx.analysis import auto_detect, process_all_groups

# 1. Auto-detect patterns
spec = auto_detect(ds)

# 2. Process all groups
results = process_all_groups(ds, spec)

# 3. Export
from quantipymrx.io import export_processing_results
export_processing_results(results, "output.xlsx")
```

## Important Patterns

### Question Type Detection

The `auto_detect.py` module classifies variables by:

1. **Name patterns**: Regex matching (P1_1, P1_2 → grid)
2. **Value labels**: Binary Yes/No → awareness
3. **Value ranges**: 0-10 → NPS, 1-5 → Likert scale

### Supported Question Types (QuestionType enum)

- `AWARENESS` - Binary Yes/No questions
- `SCALE` - Likert scales (1-5, 1-7, 1-10)
- `NPS` - Net Promoter Score (0-10)
- `GRID` - Matrix questions
- `RANKING` - Rank order mentions
- `DEMOGRAPHIC` - Demographics (age, gender, region)
- `SINGLE` - Single choice categorical
- `MULTI` - Multiple choice (dichotomous)

### Grouping Engine

Groups related variables and produces summaries:

- `AwarenessSummary` - Average awareness across brands
- `GridSummary` - Mean scores for grid items
- `TopOfMindSummary` - First mention percentages
- `GroupedFrequency` - Combined frequency table

## File Locations

| Purpose | File |
|---------|------|
| Main DataSet | `quantipymrx/core/dataset.py` |
| Pattern Detection | `quantipymrx/analysis/auto_detect.py` |
| Grouping Engine | `quantipymrx/analysis/grouping.py` |
| Excel Export | `quantipymrx/io/excel_export.py` |
| Statistical Tests | `quantipymrx/analysis/significance.py` |
| Language Support | `quantipymrx/i18n/languages.py` |

## Dependencies

### Required
- pandas >= 2.0
- numpy >= 1.24
- scipy >= 1.10
- pyreadstat >= 1.2

### Optional
- openpyxl (Excel export)
- gspread (Google Sheets export)

## Notes for Development

1. **All code is Python 3.10+** - Use modern syntax (type hints, walrus operator, etc.)
2. **No legacy quantipy code** - The old `quantipy/` folder was removed
3. **Tests use real SPSS data** - Located at `C:\Users\jorge\OneDrive\Desktop\spss.sav`
4. **Output directory** - Test exports go to `test_output/`

## Related Projects

- **surveycoder pro** - Open-ended response classification (separate project)
- **AI Followup Front/Back** - Survey analysis frontend/backend
