"""
QuantipyMRX - Service Models

Pydantic models for API requests and responses.
Designed for FastAPI integration with Talk2Data.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class AnalysisType(str, Enum):
    """Types of analysis available."""
    FREQUENCY = "frequency"
    CROSSTAB = "crosstab"
    NPS = "nps"
    TOP_BOX = "top_box"
    CORRELATION = "correlation"
    ANOVA = "anova"
    AUTO = "auto"


class ExportFormat(str, Enum):
    """Export format options."""
    JSON = "json"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    CSV = "csv"


class StatType(str, Enum):
    """Statistic types for crosstabs."""
    COUNTS = "counts"
    COL_PCT = "col_pct"
    ROW_PCT = "row_pct"
    MEAN = "mean"


# =============================================================================
# REQUEST MODELS
# =============================================================================

class FrequencyRequest(BaseModel):
    """Request for frequency analysis."""
    variable: str = Field(..., description="Variable name")
    weight: Optional[str] = Field(None, description="Weight variable")
    show_labels: bool = Field(True, description="Show value labels")
    include_statistics: bool = Field(True, description="Include mean/std")

    class Config:
        json_schema_extra = {
            "example": {
                "variable": "q1_satisfaction",
                "weight": "weight_var",
                "show_labels": True,
                "include_statistics": True,
            }
        }


class CrosstabRequest(BaseModel):
    """Request for crosstab analysis."""
    x: str = Field(..., description="Row variable")
    y: str = Field(..., description="Column variable (banner)")
    weight: Optional[str] = Field(None, description="Weight variable")
    stats: List[StatType] = Field(
        default=[StatType.COL_PCT],
        description="Statistics to include"
    )
    test_significance: bool = Field(True, description="Perform significance tests")
    sig_level: float = Field(0.05, description="Significance level")

    class Config:
        json_schema_extra = {
            "example": {
                "x": "q1_satisfaction",
                "y": "gender",
                "weight": "weight_var",
                "stats": ["col_pct", "counts"],
                "test_significance": True,
                "sig_level": 0.05,
            }
        }


class BatchCrosstabRequest(BaseModel):
    """Request for batch crosstab analysis."""
    x_variables: List[str] = Field(..., description="Row variables")
    y_variable: str = Field(..., description="Banner variable")
    weight: Optional[str] = Field(None, description="Weight variable")
    test_significance: bool = Field(True, description="Perform significance tests")


class NPSRequest(BaseModel):
    """Request for NPS calculation."""
    variable: str = Field(..., description="NPS scale variable (0-10)")
    weight: Optional[str] = Field(None, description="Weight variable")
    scale: Literal['0-10', '1-10'] = Field('0-10', description="Scale type")


class TopBoxRequest(BaseModel):
    """Request for Top Box / Bottom Box calculation."""
    variable: str = Field(..., description="Scale variable")
    n_boxes: int = Field(2, description="Number of top/bottom categories")
    weight: Optional[str] = Field(None, description="Weight variable")
    box_type: Literal['top', 'bottom', 'both'] = Field('both', description="Box type")


class CorrelationRequest(BaseModel):
    """Request for correlation analysis."""
    variables: List[str] = Field(..., description="Variables to correlate")
    method: Literal['pearson', 'spearman'] = Field('pearson', description="Correlation method")


class ANOVARequest(BaseModel):
    """Request for ANOVA analysis."""
    variable: str = Field(..., description="Numeric variable to compare")
    group_variable: str = Field(..., description="Grouping variable")
    weight: Optional[str] = Field(None, description="Weight variable")


class AutoAnalysisRequest(BaseModel):
    """Request for automatic analysis."""
    include_frequencies: bool = Field(True, description="Generate frequencies")
    include_crosstabs: bool = Field(True, description="Generate crosstabs")
    max_variables: int = Field(50, description="Max variables to analyze")
    weight: Optional[str] = Field(None, description="Weight variable")


class AnalysisRequest(BaseModel):
    """Generic analysis request."""
    analysis_type: AnalysisType = Field(..., description="Type of analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "frequency",
                "parameters": {
                    "variable": "q1_satisfaction",
                    "weight": "weight_var",
                }
            }
        }


class ExportRequest(BaseModel):
    """Request for exporting analysis results."""
    format: ExportFormat = Field(..., description="Export format")
    analysis_id: Optional[str] = Field(None, description="Analysis ID to export")
    include_all: bool = Field(False, description="Export all available analyses")
    filename: Optional[str] = Field(None, description="Custom filename")


class FilterRequest(BaseModel):
    """Request for data filtering."""
    conditions: List[Dict[str, Any]] = Field(..., description="Filter conditions")
    operator: Literal['and', 'or'] = Field('and', description="Combine operator")

    class Config:
        json_schema_extra = {
            "example": {
                "conditions": [
                    {"variable": "age", "operator": ">=", "value": 18},
                    {"variable": "gender", "operator": "==", "value": 1}
                ],
                "operator": "and"
            }
        }


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = Field(..., description="Operation success")
    message: Optional[str] = Field(None, description="Status message")
    timestamp: datetime = Field(default_factory=datetime.now)
    errors: Optional[List[str]] = Field(None, description="Error messages")


class AnalysisResponse(BaseResponse):
    """Response for analysis requests."""
    analysis_id: str = Field(..., description="Unique analysis ID")
    analysis_type: str = Field(..., description="Type of analysis performed")
    data: Dict[str, Any] = Field(..., description="Analysis results")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "analysis_id": "freq_q1_20240101_120000",
                "analysis_type": "frequency",
                "data": {
                    "variable": "q1",
                    "label": "Satisfaction",
                    "base": 1000,
                    "data": [
                        {"code": 1, "label": "Very Satisfied", "count": 250, "percent": 25.0}
                    ]
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }


class DatasetInfoResponse(BaseResponse):
    """Response for dataset information."""
    name: str
    n_cases: int
    n_variables: int
    variables: List[Dict[str, Any]]
    demographics: List[str]
    scales: List[str]


class ExportResponse(BaseResponse):
    """Response for export requests."""
    format: str
    filename: str
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_size: Optional[int] = None


class InsightsResponse(BaseResponse):
    """Response for AI insights."""
    insights: List[str]
    key_findings: List[Dict[str, Any]]
    recommendations: List[str]


# =============================================================================
# CHAT/AI INTEGRATION MODELS
# =============================================================================

class ChatAnalysisRequest(BaseModel):
    """
    Request from AI chat to perform analysis.

    This model is designed for Claude/AI to send structured requests.
    """
    intent: Literal[
        'frequency', 'crosstab', 'compare', 'trend',
        'nps', 'satisfaction', 'demographics', 'summary'
    ] = Field(..., description="Analysis intent")
    variables: List[str] = Field(default_factory=list, description="Variables involved")
    banner: Optional[str] = Field(None, description="Banner/grouping variable")
    filters: Optional[List[Dict[str, Any]]] = Field(None, description="Filter conditions")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")

    class Config:
        json_schema_extra = {
            "example": {
                "intent": "crosstab",
                "variables": ["satisfaction"],
                "banner": "gender",
                "filters": [{"variable": "age", "operator": ">=", "value": 18}],
                "options": {"include_significance": True}
            }
        }


class ChatAnalysisResponse(BaseModel):
    """
    Response formatted for AI chat consumption.

    Includes both data and natural language description.
    """
    success: bool
    analysis_type: str
    summary: str = Field(..., description="Natural language summary for AI")
    key_points: List[str] = Field(..., description="Key findings as bullet points")
    data: Dict[str, Any] = Field(..., description="Structured data")
    visualization_hint: Optional[str] = Field(None, description="Suggested visualization type")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "analysis_type": "crosstab",
                "summary": "Satisfaction varies significantly by gender. Males show higher satisfaction (45% Top 2 Box) compared to females (38%).",
                "key_points": [
                    "Overall satisfaction: 42% Top 2 Box",
                    "Males: 45% satisfied vs Females: 38%",
                    "Difference is statistically significant (p<0.05)"
                ],
                "data": {},
                "visualization_hint": "grouped_bar_chart"
            }
        }
