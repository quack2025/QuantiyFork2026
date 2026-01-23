"""
QuantipyMRX - Metadata Models

Pydantic models for type-safe metadata handling.
Supports SPSS variable types and multi-language labels.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class VariableType(str, Enum):
    """SPSS/MRX variable types."""
    SINGLE = "single"              # Single-choice categorical
    DELIMITED_SET = "delimited set"  # Multi-choice (stored as "1;2;3;")
    INT = "int"                    # Integer numeric
    FLOAT = "float"                # Decimal numeric
    STRING = "string"              # Open-ended text
    DATE = "date"                  # Date/datetime
    ARRAY = "array"                # Grid/matrix question


class TextKey(str, Enum):
    """Supported text languages."""
    EN_GB = "en-GB"
    EN_US = "en-US"
    ES_ES = "es-ES"
    DE_DE = "de-DE"
    FR_FR = "fr-FR"
    PT_BR = "pt-BR"
    IT_IT = "it-IT"
    NL_NL = "nl-NL"
    JA_JP = "ja-JP"
    ZH_CN = "zh-CN"


class ValueLabel(BaseModel):
    """A single value code with its text label(s)."""
    value: int
    text: Dict[str, str] = Field(default_factory=dict)

    @field_validator('value')
    @classmethod
    def value_must_be_valid(cls, v: int) -> int:
        """Value codes should typically be non-negative."""
        return v

    def get_text(self, text_key: str = "en-GB") -> str:
        """Get label for specific language, fallback to first available."""
        if text_key in self.text:
            return self.text[text_key]
        if self.text:
            return next(iter(self.text.values()))
        return str(self.value)


class VariableMeta(BaseModel):
    """Metadata for a single variable."""
    name: str
    type: VariableType
    text: Dict[str, str] = Field(default_factory=dict)
    values: Optional[List[ValueLabel]] = None
    parent: Optional[str] = None
    missing_values: Optional[List[int]] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True

    @property
    def label(self) -> str:
        """Get default label (en-GB or first available)."""
        return self.get_label()

    def get_label(self, text_key: str = "en-GB") -> str:
        """Get variable label for specific language."""
        if text_key in self.text:
            return self.text[text_key]
        if self.text:
            return next(iter(self.text.values()))
        return self.name

    @property
    def codes(self) -> List[int]:
        """Get list of valid value codes."""
        if self.values:
            return [v.value for v in self.values]
        return []

    def get_value_label(self, code: int, text_key: str = "en-GB") -> str:
        """Get label for a specific value code."""
        if self.values:
            for v in self.values:
                if v.value == code:
                    return v.get_text(text_key)
        return str(code)

    @property
    def is_categorical(self) -> bool:
        """Check if variable is categorical (has value labels)."""
        return self.type in [VariableType.SINGLE, VariableType.DELIMITED_SET]

    @property
    def is_numeric(self) -> bool:
        """Check if variable is numeric."""
        return self.type in [VariableType.INT, VariableType.FLOAT]

    @property
    def is_multi(self) -> bool:
        """Check if variable is multiple choice."""
        return self.type == VariableType.DELIMITED_SET


class ArrayMeta(BaseModel):
    """Metadata for an array/grid question."""
    name: str
    type: VariableType = VariableType.ARRAY
    text: Dict[str, str] = Field(default_factory=dict)
    items: List[str] = Field(default_factory=list)  # Column names of items
    subtype: Optional[VariableType] = None  # Type of items (single, delimited set)
    values: Optional[List[ValueLabel]] = None  # Shared values for all items
    properties: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class DataSetInfo(BaseModel):
    """General information about the dataset."""
    name: str = ""
    source_file: Optional[str] = None
    source_format: Optional[str] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    case_count: int = 0
    weight_variable: Optional[str] = None
    text: Dict[str, str] = Field(default_factory=dict)
    properties: Dict[str, Any] = Field(default_factory=dict)


class DataSetMeta(BaseModel):
    """Complete metadata structure for a dataset."""
    info: DataSetInfo = Field(default_factory=DataSetInfo)
    columns: Dict[str, VariableMeta] = Field(default_factory=dict)
    masks: Dict[str, ArrayMeta] = Field(default_factory=dict)  # Arrays/grids
    sets: Dict[str, List[str]] = Field(default_factory=dict)   # Variable groups

    def get_variable(self, name: str) -> Optional[VariableMeta]:
        """Get variable metadata by name."""
        return self.columns.get(name)

    def get_array(self, name: str) -> Optional[ArrayMeta]:
        """Get array metadata by name."""
        return self.masks.get(name)

    @property
    def variable_names(self) -> List[str]:
        """List all variable names."""
        return list(self.columns.keys())

    @property
    def array_names(self) -> List[str]:
        """List all array names."""
        return list(self.masks.keys())

    def variables_by_type(self, var_type: VariableType) -> List[str]:
        """Get variable names filtered by type."""
        return [
            name for name, meta in self.columns.items()
            if meta.type == var_type
        ]

    @property
    def single_variables(self) -> List[str]:
        """Get all single-choice variables."""
        return self.variables_by_type(VariableType.SINGLE)

    @property
    def multi_variables(self) -> List[str]:
        """Get all multiple-choice variables."""
        return self.variables_by_type(VariableType.DELIMITED_SET)

    @property
    def numeric_variables(self) -> List[str]:
        """Get all numeric variables (int + float)."""
        return (
            self.variables_by_type(VariableType.INT) +
            self.variables_by_type(VariableType.FLOAT)
        )

    @property
    def text_variables(self) -> List[str]:
        """Get all text/string variables."""
        return self.variables_by_type(VariableType.STRING)
