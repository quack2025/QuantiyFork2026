"""
QuantipyMRX - JSON Export

JSON export for frontend consumption and API responses.
Structured for easy parsing by JavaScript/TypeScript.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult
    from quantipymrx.analysis.auto import AutoAnalysisResult


class QuantipyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for QuantipyMRX objects."""

    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle pandas types
        if isinstance(obj, pd.DataFrame):
            return {
                'type': 'dataframe',
                'columns': list(obj.columns),
                'index': list(obj.index),
                'data': obj.to_dict(orient='records'),
            }
        if isinstance(obj, pd.Series):
            return {
                'type': 'series',
                'name': obj.name,
                'index': list(obj.index),
                'values': obj.tolist(),
            }

        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle dataclasses
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)

        # Handle Path
        if isinstance(obj, Path):
            return str(obj)

        return super().default(obj)


def to_json(obj: Any, indent: int = 2) -> str:
    """
    Convert any QuantipyMRX object to JSON string.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    indent : int
        JSON indentation.

    Returns
    -------
    str
        JSON string.
    """
    return json.dumps(obj, cls=QuantipyJSONEncoder, indent=indent)


def save_json(obj: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save object to JSON file.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    filepath : str or Path
        Output file path.
    indent : int
        JSON indentation.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, cls=QuantipyJSONEncoder, indent=indent, ensure_ascii=False)


# =============================================================================
# STRUCTURED EXPORTS
# =============================================================================

def frequency_to_json(result: "FrequencyResult") -> Dict[str, Any]:
    """
    Convert FrequencyResult to JSON-serializable dict.

    Structure optimized for frontend consumption.
    """
    return {
        'type': 'frequency',
        'variable': result.variable,
        'label': result.label,
        'base': result.base,
        'weighted_base': result.weighted_base,
        'is_weighted': result.is_weighted,
        'statistics': {
            'mean': result.mean,
            'std': result.std,
            'median': result.median,
        },
        'data': [
            {
                'code': int(row.get('Value', idx)) if 'Value' in result.table.columns else idx,
                'label': str(idx),
                'count': float(row.get('Count', 0)),
                'percent': float(row.get('Percent', 0)),
            }
            for idx, row in result.table.iterrows()
        ],
    }


def crosstab_to_json(result: "CrosstabResult") -> Dict[str, Any]:
    """
    Convert CrosstabResult to JSON-serializable dict.

    Structure optimized for frontend table rendering.
    """
    # Convert column percentages to nested structure
    data = []
    for idx, row in result.col_pct.iterrows():
        row_data = {
            'row_label': str(idx),
            'values': {}
        }
        for col in result.col_pct.columns:
            row_data['values'][str(col)] = {
                'percent': float(row[col]),
                'count': float(result.counts.loc[idx, col]) if col in result.counts.columns else 0,
            }

            # Add significance letter if available
            if result.significance is not None:
                try:
                    sig = result.significance.loc[idx, col]
                    if sig:
                        row_data['values'][str(col)]['significance'] = sig
                except:
                    pass

        data.append(row_data)

    return {
        'type': 'crosstab',
        'x': {
            'variable': result.x,
            'label': result.x_label,
        },
        'y': {
            'variable': result.y,
            'label': result.y_label,
        },
        'base': {
            'total': result.total_base,
            'by_column': {str(k): int(v) for k, v in result.base.items()},
        },
        'is_weighted': result.is_weighted,
        'statistics': {
            'chi_square': result.chi2,
            'p_value': result.chi2_pvalue,
            'significant': result.chi2_pvalue < 0.05 if result.chi2_pvalue else False,
        },
        'columns': [str(c) for c in result.col_pct.columns],
        'data': data,
    }


def analysis_to_json(results: "AutoAnalysisResult") -> Dict[str, Any]:
    """
    Convert AutoAnalysisResult to JSON-serializable dict.

    Comprehensive structure for full analysis export.
    """
    return {
        'type': 'analysis',
        'generated_at': datetime.now().isoformat(),
        'summary': results.dataset_summary,
        'insights': results.key_insights,
        'banner': results.banner.to_dict(),
        'variable_classifications': {
            k: v.to_dict() for k, v in results.variable_classifications.items()
        },
        'frequencies': {
            var: frequency_to_json(freq)
            for var, freq in results.frequencies.items()
        },
        'crosstabs': {
            banner_var: {
                x_var: crosstab_to_json(xtab)
                for x_var, xtab in xtabs.items()
            }
            for banner_var, xtabs in results.crosstabs.items()
        },
    }


def dataset_to_json(dataset: "DataSet", include_data: bool = False) -> Dict[str, Any]:
    """
    Convert DataSet to JSON-serializable dict.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    include_data : bool
        If True, includes raw data (can be large).

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    result = {
        'type': 'dataset',
        'name': dataset.name,
        'text_key': dataset.text_key,
        'n_cases': dataset.n_cases,
        'n_variables': dataset.n_variables,
        'weight': dataset.weight,
        'metadata': {
            'info': dataset.meta.info.model_dump() if hasattr(dataset.meta.info, 'model_dump') else {},
            'variables': {},
        },
    }

    # Add variable metadata
    for var_name in dataset.variables:
        meta = dataset.get_variable_meta(var_name)
        if meta:
            var_info = {
                'name': meta.name,
                'type': meta.type.value if hasattr(meta.type, 'value') else str(meta.type),
                'label': meta.get_label(dataset.text_key),
            }
            if meta.values:
                var_info['values'] = [
                    {'code': v.value, 'label': v.get_text(dataset.text_key)}
                    for v in meta.values
                ]
            result['metadata']['variables'][var_name] = var_info

    # Optionally include data
    if include_data:
        result['data'] = dataset.data.to_dict(orient='records')

    return result


# =============================================================================
# API RESPONSE FORMATS
# =============================================================================

def api_response(
    data: Any,
    success: bool = True,
    message: str = "",
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Wrap data in standard API response format.

    Parameters
    ----------
    data : Any
        Response data.
    success : bool
        Operation success status.
    message : str
        Optional message.
    errors : list of str
        Error messages if any.

    Returns
    -------
    dict
        Standard API response.
    """
    response = {
        'success': success,
        'data': data,
        'timestamp': datetime.now().isoformat(),
    }

    if message:
        response['message'] = message

    if errors:
        response['errors'] = errors

    return response


def paginated_response(
    items: List[Any],
    page: int = 1,
    per_page: int = 20,
    total: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create paginated API response.

    Parameters
    ----------
    items : list
        All items (will be sliced).
    page : int
        Current page (1-indexed).
    per_page : int
        Items per page.
    total : int, optional
        Total items (if different from len(items)).

    Returns
    -------
    dict
        Paginated response.
    """
    total_items = total if total is not None else len(items)
    total_pages = (total_items + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_items = items[start_idx:end_idx]

    return {
        'items': page_items,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1,
        },
    }
