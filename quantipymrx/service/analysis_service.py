"""
QuantipyMRX - Analysis Service

Main service layer for Talk2Data integration.
Provides high-level API for all analysis operations.
"""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from quantipymrx.core.dataset import DataSet
from quantipymrx.core.metadata import VariableType
from quantipymrx.analysis.frequency import frequency, frequencies_batch, FrequencyResult
from quantipymrx.analysis.crosstab import crosstab, crosstab_batch, CrosstabResult
from quantipymrx.analysis.significance import (
    chi_square_test, correlation, correlation_matrix, anova_from_dataset
)
from quantipymrx.analysis.mrx import (
    calculate_nps, top_box, bottom_box, mrx_summary, NPSResult, BoxResult
)
from quantipymrx.analysis.auto import (
    auto_analyze, classify_all_variables, detect_banners,
    quick_summary, suggest_analyses, AutoAnalysisResult
)
from quantipymrx.export.json_export import (
    frequency_to_json, crosstab_to_json, analysis_to_json, api_response
)
from quantipymrx.service.models import (
    AnalysisType, ExportFormat, AnalysisRequest, AnalysisResponse,
    FrequencyRequest, CrosstabRequest, BatchCrosstabRequest,
    NPSRequest, TopBoxRequest, AutoAnalysisRequest,
    ChatAnalysisRequest, ChatAnalysisResponse,
)


class AnalysisService:
    """
    Main service for survey data analysis.

    Provides a high-level API for Talk2Data integration, handling:
    - Dataset loading and caching
    - All analysis types (frequency, crosstab, NPS, etc.)
    - Export to various formats
    - AI-friendly responses

    Example
    -------
    >>> service = AnalysisService()
    >>> service.load_spss("survey.sav")
    >>> result = service.frequency("q1")
    >>> export_path = service.export_to_excel()
    """

    def __init__(self):
        self._dataset: Optional[DataSet] = None
        self._analysis_cache: Dict[str, Any] = {}
        self._auto_analysis: Optional[AutoAnalysisResult] = None
        self._export_dir: Path = Path("./exports")

    # =========================================================================
    # DATASET MANAGEMENT
    # =========================================================================

    def load_spss(
        self,
        filepath: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "en-GB",
    ) -> Dict[str, Any]:
        """
        Load an SPSS file.

        Parameters
        ----------
        filepath : str or Path
            Path to .sav file.
        name : str, optional
            Dataset name.
        text_key : str
            Language for labels.

        Returns
        -------
        dict
            Dataset info and summary.
        """
        self._dataset = DataSet.from_spss(filepath, name=name, text_key=text_key)
        self._analysis_cache.clear()
        self._auto_analysis = None

        return self.get_dataset_info()

    def load_dataframe(
        self,
        data,
        name: str = "dataset",
        text_key: str = "en-GB",
    ) -> Dict[str, Any]:
        """Load from pandas DataFrame."""
        self._dataset = DataSet.from_dataframe(data, name=name, text_key=text_key)
        self._analysis_cache.clear()
        self._auto_analysis = None

        return self.get_dataset_info()

    @property
    def dataset(self) -> DataSet:
        """Get current dataset."""
        if self._dataset is None:
            raise ValueError("No dataset loaded. Call load_spss() first.")
        return self._dataset

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset summary information."""
        ds = self.dataset
        classifications = classify_all_variables(ds)

        return {
            'name': ds.name,
            'n_cases': ds.n_cases,
            'n_variables': ds.n_variables,
            'text_key': ds.text_key,
            'variable_counts': {
                'single': len(ds.singles()),
                'multi': len(ds.multis()),
                'numeric': len(ds.numerics()),
                'text': len(ds.texts()),
            },
            'demographics': [
                c.name for c in classifications.values()
                if c.category == 'demographic'
            ],
            'scales': [
                c.name for c in classifications.values()
                if c.is_scale
            ],
            'variables': [
                {
                    'name': var,
                    'label': ds.label(var),
                    'type': ds.var_type(var).value if ds.var_type(var) else 'unknown',
                    'n_codes': len(ds.codes(var)) if ds.is_categorical(var) else None,
                }
                for var in ds.variables[:100]  # Limit for large datasets
            ],
        }

    def get_variable_info(self, variable: str) -> Dict[str, Any]:
        """Get detailed info for a variable."""
        ds = self.dataset
        meta = ds.get_variable_meta(variable)

        if meta is None:
            raise ValueError(f"Variable '{variable}' not found")

        info = {
            'name': variable,
            'label': meta.get_label(ds.text_key),
            'type': meta.type.value if hasattr(meta.type, 'value') else str(meta.type),
            'is_categorical': meta.is_categorical,
            'is_numeric': meta.is_numeric,
        }

        if meta.values:
            info['values'] = [
                {'code': v.value, 'label': v.get_text(ds.text_key)}
                for v in meta.values
            ]

        # Add basic statistics
        col_data = ds.data[variable]
        info['n_valid'] = int(col_data.notna().sum())
        info['n_missing'] = int(col_data.isna().sum())

        if meta.is_numeric or meta.is_categorical:
            numeric_data = col_data.dropna()
            if len(numeric_data) > 0:
                info['mean'] = float(numeric_data.mean())
                info['std'] = float(numeric_data.std())
                info['min'] = float(numeric_data.min())
                info['max'] = float(numeric_data.max())

        return info

    def set_weight(self, variable: Optional[str]) -> None:
        """Set default weight variable."""
        self.dataset.weight = variable

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def frequency(
        self,
        variable: str,
        weight: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate frequency for a variable.

        Returns JSON-serializable dict.
        """
        result = frequency(self.dataset, variable, weight=weight, **kwargs)

        # Cache result
        cache_key = self._cache_key('frequency', variable, weight)
        self._analysis_cache[cache_key] = result

        return frequency_to_json(result)

    def crosstab(
        self,
        x: str,
        y: str,
        weight: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate crosstab between two variables.

        Returns JSON-serializable dict.
        """
        result = crosstab(self.dataset, x=x, y=y, weight=weight, **kwargs)

        # Cache result
        cache_key = self._cache_key('crosstab', f"{x}_by_{y}", weight)
        self._analysis_cache[cache_key] = result

        return crosstab_to_json(result)

    def batch_crosstab(
        self,
        x_variables: List[str],
        y_variable: str,
        weight: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate crosstabs for multiple variables against a banner.

        Returns dict of {x_variable: crosstab_json}.
        """
        results = crosstab_batch(
            self.dataset,
            x_vars=x_variables,
            y_var=y_variable,
            weight=weight,
            **kwargs,
        )

        return {
            x_var: crosstab_to_json(result)
            for x_var, result in results.items()
        }

    def nps(
        self,
        variable: str,
        weight: Optional[str] = None,
        scale: str = '0-10',
    ) -> Dict[str, Any]:
        """Calculate NPS for a variable."""
        data = self.dataset.data[variable]
        weights = self.dataset.data[weight] if weight else None

        result = calculate_nps(data, weights, scale=scale)

        return result.to_dict()

    def top_bottom_box(
        self,
        variable: str,
        n_boxes: int = 2,
        weight: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate Top Box and Bottom Box."""
        data = self.dataset.data[variable]
        weights = self.dataset.data[weight] if weight else None

        t2b = top_box(data, n_boxes=n_boxes, weights=weights)
        b2b = bottom_box(data, n_boxes=n_boxes, weights=weights)

        return {
            'variable': variable,
            'top_box': t2b.to_dict(),
            'bottom_box': b2b.to_dict(),
        }

    def mrx_metrics(
        self,
        variable: str,
        weight: Optional[str] = None,
        scale: str = '1-5',
    ) -> Dict[str, Any]:
        """Calculate comprehensive MRX metrics."""
        return mrx_summary(self.dataset, variable, weight=weight, scale=scale)

    def correlation_analysis(
        self,
        variables: Optional[List[str]] = None,
        method: str = 'pearson',
    ) -> Dict[str, Any]:
        """Calculate correlation matrix."""
        corr_matrix, p_values = correlation_matrix(
            self.dataset, variables=variables, method=method
        )

        return {
            'method': method,
            'correlation': corr_matrix.to_dict(),
            'p_values': p_values.to_dict(),
        }

    def anova_analysis(
        self,
        variable: str,
        group_variable: str,
        weight: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform ANOVA analysis."""
        result = anova_from_dataset(
            self.dataset, variable, group_variable, weight=weight
        )

        return result.to_dict()

    # =========================================================================
    # AUTO ANALYSIS
    # =========================================================================

    def auto_analyze(
        self,
        weight: Optional[str] = None,
        include_crosstabs: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Perform automatic analysis of the dataset.

        Returns comprehensive analysis suitable for dashboard generation.
        """
        self._auto_analysis = auto_analyze(
            self.dataset,
            weight=weight,
            include_crosstabs=include_crosstabs,
            progress_callback=progress_callback,
        )

        return analysis_to_json(self._auto_analysis)

    def get_insights(self) -> List[str]:
        """Get key insights from auto-analysis."""
        if self._auto_analysis is None:
            self.auto_analyze()

        return self._auto_analysis.key_insights

    def get_suggested_analyses(self) -> List[Dict[str, Any]]:
        """Get suggested analyses based on data structure."""
        return suggest_analyses(self.dataset)

    def get_quick_summary(self) -> str:
        """Get text summary of the dataset."""
        return quick_summary(self.dataset)

    # =========================================================================
    # CHAT/AI INTEGRATION
    # =========================================================================

    def process_chat_request(
        self,
        request: ChatAnalysisRequest,
    ) -> ChatAnalysisResponse:
        """
        Process an analysis request from AI chat.

        Designed for Claude to call with structured requests.

        Parameters
        ----------
        request : ChatAnalysisRequest
            Structured request from AI.

        Returns
        -------
        ChatAnalysisResponse
            Response with data and natural language summary.
        """
        try:
            # Apply filters if provided
            ds = self.dataset
            if request.filters:
                # Build filter condition
                conditions = []
                for f in request.filters:
                    var = f.get('variable')
                    op = f.get('operator', '==')
                    val = f.get('value')
                    conditions.append(f"{var} {op} {val}")

                filter_str = ' & '.join(conditions) if request.options.get('filter_operator', 'and') == 'and' else ' | '.join(conditions)
                ds = ds.filter(filter_str)

            # Execute based on intent
            if request.intent == 'frequency':
                var = request.variables[0] if request.variables else None
                if not var:
                    raise ValueError("No variable specified for frequency")

                result = frequency(ds, var)
                data = frequency_to_json(result)

                # Generate summary
                top_response = result.table.index[0]
                top_pct = result.table['Percent'].iloc[0]
                summary = f"For '{result.label}', the most common response is '{top_response}' ({top_pct:.1f}%)."

                key_points = [
                    f"Base: {result.base:,} responses",
                    f"Most common: {top_response} ({top_pct:.1f}%)",
                ]
                if result.mean is not None:
                    key_points.append(f"Mean: {result.mean:.2f}")

                return ChatAnalysisResponse(
                    success=True,
                    analysis_type='frequency',
                    summary=summary,
                    key_points=key_points,
                    data=data,
                    visualization_hint='bar_chart',
                )

            elif request.intent == 'crosstab' or request.intent == 'compare':
                var = request.variables[0] if request.variables else None
                banner = request.banner

                if not var or not banner:
                    raise ValueError("Both variable and banner required for crosstab")

                result = crosstab(ds, x=var, y=banner)
                data = crosstab_to_json(result)

                # Generate summary
                sig_text = ""
                if result.chi2_pvalue is not None:
                    if result.chi2_pvalue < 0.05:
                        sig_text = " The difference is statistically significant."
                    else:
                        sig_text = " The difference is not statistically significant."

                summary = f"Cross-tabulation of '{result.x_label}' by '{result.y_label}' (Base: {result.total_base:,}).{sig_text}"

                key_points = [f"Total base: {result.total_base:,}"]
                if result.chi2_pvalue is not None:
                    key_points.append(f"Chi-square p-value: {result.chi2_pvalue:.4f}")

                return ChatAnalysisResponse(
                    success=True,
                    analysis_type='crosstab',
                    summary=summary,
                    key_points=key_points,
                    data=data,
                    visualization_hint='grouped_bar_chart',
                )

            elif request.intent == 'nps':
                var = request.variables[0] if request.variables else None
                if not var:
                    raise ValueError("No variable specified for NPS")

                nps_data = self.nps(var)

                summary = f"NPS for '{var}' is {nps_data['nps']:.0f} ({nps_data['nps_category']})."
                key_points = [
                    f"NPS: {nps_data['nps']:.0f}",
                    f"Promoters: {nps_data['promoters_pct']:.1f}%",
                    f"Detractors: {nps_data['detractors_pct']:.1f}%",
                    f"Base: {nps_data['base']:,}",
                ]

                return ChatAnalysisResponse(
                    success=True,
                    analysis_type='nps',
                    summary=summary,
                    key_points=key_points,
                    data=nps_data,
                    visualization_hint='gauge_chart',
                )

            elif request.intent == 'summary' or request.intent == 'demographics':
                info = self.get_dataset_info()
                summary = self.get_quick_summary()

                return ChatAnalysisResponse(
                    success=True,
                    analysis_type='summary',
                    summary=f"Dataset '{info['name']}' has {info['n_cases']:,} responses and {info['n_variables']} variables.",
                    key_points=[
                        f"Total responses: {info['n_cases']:,}",
                        f"Single choice vars: {info['variable_counts']['single']}",
                        f"Multi choice vars: {info['variable_counts']['multi']}",
                        f"Demographics detected: {', '.join(info['demographics'][:5]) if info['demographics'] else 'None'}",
                    ],
                    data=info,
                    visualization_hint=None,
                )

            else:
                raise ValueError(f"Unknown intent: {request.intent}")

        except Exception as e:
            return ChatAnalysisResponse(
                success=False,
                analysis_type=request.intent,
                summary=f"Error: {str(e)}",
                key_points=[],
                data={},
            )

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_to_excel(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Export analysis results to Excel.

        Returns filepath.
        """
        from quantipymrx.export.excel import export_analysis_to_excel

        if self._auto_analysis is None:
            self.auto_analyze()

        if filepath is None:
            self._export_dir.mkdir(exist_ok=True)
            filepath = self._export_dir / f"{self.dataset.name}_analysis.xlsx"

        export_analysis_to_excel(self._auto_analysis, filepath)
        return str(filepath)

    def export_to_powerpoint(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Export analysis results to PowerPoint.

        Returns filepath.
        """
        from quantipymrx.export.powerpoint import export_analysis_to_pptx

        if self._auto_analysis is None:
            self.auto_analyze()

        if filepath is None:
            self._export_dir.mkdir(exist_ok=True)
            filepath = self._export_dir / f"{self.dataset.name}_analysis.pptx"

        export_analysis_to_pptx(self._auto_analysis, filepath)
        return str(filepath)

    def export_to_json(
        self,
        filepath: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Export analysis results to JSON.

        Returns filepath.
        """
        from quantipymrx.export.json_export import save_json

        if self._auto_analysis is None:
            self.auto_analyze()

        if filepath is None:
            self._export_dir.mkdir(exist_ok=True)
            filepath = self._export_dir / f"{self.dataset.name}_analysis.json"

        data = analysis_to_json(self._auto_analysis)
        save_json(data, filepath)
        return str(filepath)

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _cache_key(self, analysis_type: str, variables: str, weight: Optional[str]) -> str:
        """Generate cache key for analysis result."""
        key_str = f"{analysis_type}_{variables}_{weight or 'unweighted'}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        self._auto_analysis = None
