"""
QuantipyMRX - DataSet

Core class for managing survey data with metadata.
Provides intuitive API for data access and analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

from quantipymrx.core.metadata import (
    DataSetMeta,
    DataSetInfo,
    VariableMeta,
    VariableType,
    ValueLabel,
)

if TYPE_CHECKING:
    from quantipymrx.analysis.frequency import FrequencyResult
    from quantipymrx.analysis.crosstab import CrosstabResult


class DataSet:
    """
    A survey dataset with data and metadata.

    The DataSet is the central object in QuantipyMRX. It holds both the
    survey responses (as a pandas DataFrame) and the metadata (variable
    labels, value labels, types, etc.).

    Parameters
    ----------
    name : str
        Name identifier for the dataset.
    text_key : str, default "en-GB"
        Default language for labels.

    Attributes
    ----------
    data : pd.DataFrame
        The survey response data.
    meta : DataSetMeta
        Complete metadata structure.
    name : str
        Dataset name.
    text_key : str
        Default language key.

    Example
    -------
    >>> ds = DataSet.from_spss("survey.sav")
    >>> print(ds.shape)
    (1000, 50)
    >>> print(ds.variables)
    ['q1', 'q2', 'gender', 'age', ...]
    >>> freq = ds.frequency("q1")
    """

    def __init__(self, name: str = "", text_key: str = "en-GB"):
        self.name = name
        self.text_key = text_key
        self._data: pd.DataFrame = pd.DataFrame()
        self._meta: DataSetMeta = DataSetMeta()
        self._weight: Optional[str] = None
        self._filter_mask: Optional[pd.Series] = None

    # -------------------------------------------------------------------------
    # Class methods for loading data
    # -------------------------------------------------------------------------

    @classmethod
    def from_spss(
        cls,
        filepath: Union[str, Path],
        name: Optional[str] = None,
        text_key: str = "en-GB",
        **kwargs,
    ) -> "DataSet":
        """
        Create DataSet from an SPSS .sav file.

        Parameters
        ----------
        filepath : str or Path
            Path to the .sav file.
        name : str, optional
            Dataset name. If None, uses filename.
        text_key : str, default "en-GB"
            Language key for labels.
        **kwargs
            Additional arguments passed to read_spss.

        Returns
        -------
        DataSet
            Loaded dataset.

        Example
        -------
        >>> ds = DataSet.from_spss("survey.sav")
        >>> ds = DataSet.from_spss("survey.sav", text_key="es-ES")
        """
        from quantipymrx.io.spss import read_spss

        filepath = Path(filepath)
        if name is None:
            name = filepath.stem

        data, meta = read_spss(filepath, text_key=text_key, **kwargs)

        ds = cls(name=name, text_key=text_key)
        ds._data = data
        ds._meta = meta
        ds._meta.info.name = name

        return ds

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        name: str = "dataset",
        text_key: str = "en-GB",
    ) -> "DataSet":
        """
        Create DataSet from a pandas DataFrame.

        Basic metadata will be inferred from the DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The data.
        name : str, default "dataset"
            Dataset name.
        text_key : str, default "en-GB"
            Language key.

        Returns
        -------
        DataSet
            Dataset with inferred metadata.
        """
        ds = cls(name=name, text_key=text_key)
        ds._data = data.copy()
        ds._meta = ds._infer_metadata(data, text_key)
        return ds

    def _infer_metadata(self, data: pd.DataFrame, text_key: str) -> DataSetMeta:
        """Infer basic metadata from DataFrame."""
        columns: Dict[str, VariableMeta] = {}

        for col in data.columns:
            dtype = data[col].dtype

            if pd.api.types.is_categorical_dtype(dtype):
                var_type = VariableType.SINGLE
                values = [
                    ValueLabel(value=i, text={text_key: str(cat)})
                    for i, cat in enumerate(data[col].cat.categories, 1)
                ]
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                var_type = VariableType.DATE
                values = None
            elif pd.api.types.is_float_dtype(dtype):
                var_type = VariableType.FLOAT
                values = None
            elif pd.api.types.is_integer_dtype(dtype):
                var_type = VariableType.INT
                values = None
            else:
                var_type = VariableType.STRING
                values = None

            columns[col] = VariableMeta(
                name=col,
                type=var_type,
                text={text_key: col},
                values=values,
            )

        return DataSetMeta(
            info=DataSetInfo(name=self.name, case_count=len(data)),
            columns=columns,
            sets={"data file": list(data.columns)},
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Access the underlying DataFrame."""
        return self._data

    @property
    def meta(self) -> DataSetMeta:
        """Access the metadata."""
        return self._meta

    @property
    def shape(self) -> tuple:
        """Return (n_rows, n_columns)."""
        return self._data.shape

    @property
    def n_cases(self) -> int:
        """Number of cases/rows."""
        return len(self._data)

    @property
    def n_variables(self) -> int:
        """Number of variables/columns."""
        return len(self._meta.columns)

    @property
    def variables(self) -> List[str]:
        """List all variable names."""
        return list(self._meta.columns.keys())

    @property
    def weight(self) -> Optional[str]:
        """Current weight variable."""
        return self._weight

    @weight.setter
    def weight(self, var_name: Optional[str]):
        """Set weight variable."""
        if var_name is not None and var_name not in self._data.columns:
            raise ValueError(f"Weight variable '{var_name}' not found in data")
        self._weight = var_name

    # -------------------------------------------------------------------------
    # Variable access and info
    # -------------------------------------------------------------------------

    def __getitem__(self, key: Union[str, List[str]]) -> pd.Series:
        """Access variable data by name."""
        return self._data[key]

    def __contains__(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self._meta.columns

    def var_exists(self, name: str) -> bool:
        """Check if a variable exists."""
        return name in self._meta.columns

    def get_variable_meta(self, name: str) -> Optional[VariableMeta]:
        """Get metadata for a variable."""
        return self._meta.get_variable(name)

    def label(self, name: str) -> str:
        """Get variable label."""
        meta = self.get_variable_meta(name)
        if meta:
            return meta.get_label(self.text_key)
        return name

    def codes(self, name: str) -> List[int]:
        """Get value codes for a categorical variable."""
        meta = self.get_variable_meta(name)
        if meta and meta.values:
            return meta.codes
        return []

    def value_labels(self, name: str) -> Dict[int, str]:
        """Get value labels as dict {code: label}."""
        meta = self.get_variable_meta(name)
        if meta and meta.values:
            return {
                v.value: v.get_text(self.text_key)
                for v in meta.values
            }
        return {}

    def var_type(self, name: str) -> Optional[VariableType]:
        """Get variable type."""
        meta = self.get_variable_meta(name)
        return meta.type if meta else None

    def is_categorical(self, name: str) -> bool:
        """Check if variable is categorical."""
        meta = self.get_variable_meta(name)
        return meta.is_categorical if meta else False

    def is_numeric(self, name: str) -> bool:
        """Check if variable is numeric."""
        meta = self.get_variable_meta(name)
        return meta.is_numeric if meta else False

    def is_multi(self, name: str) -> bool:
        """Check if variable is multiple choice (delimited set)."""
        meta = self.get_variable_meta(name)
        return meta.is_multi if meta else False

    # -------------------------------------------------------------------------
    # Variable lists by type
    # -------------------------------------------------------------------------

    def singles(self) -> List[str]:
        """Get all single-choice variables."""
        return self._meta.single_variables

    def multis(self) -> List[str]:
        """Get all multiple-choice variables."""
        return self._meta.multi_variables

    def numerics(self) -> List[str]:
        """Get all numeric variables."""
        return self._meta.numeric_variables

    def texts(self) -> List[str]:
        """Get all text variables."""
        return self._meta.text_variables

    # -------------------------------------------------------------------------
    # Data operations
    # -------------------------------------------------------------------------

    def filter(self, condition: Union[str, pd.Series]) -> "DataSet":
        """
        Apply a filter to the dataset.

        Parameters
        ----------
        condition : str or pd.Series
            Filter condition. If string, evaluated as query.
            If Series, used as boolean mask.

        Returns
        -------
        DataSet
            Filtered copy of the dataset.

        Example
        -------
        >>> filtered = ds.filter("gender == 1")
        >>> filtered = ds.filter(ds["age"] > 30)
        """
        ds = self.copy()

        if isinstance(condition, str):
            mask = ds._data.eval(condition)
        else:
            mask = condition

        ds._data = ds._data[mask].reset_index(drop=True)
        ds._meta.info.case_count = len(ds._data)

        return ds

    def copy(self) -> "DataSet":
        """Create a deep copy of the dataset."""
        ds = DataSet(name=self.name, text_key=self.text_key)
        ds._data = self._data.copy()
        ds._meta = self._meta.model_copy(deep=True)
        ds._weight = self._weight
        return ds

    def describe(self) -> pd.DataFrame:
        """Get summary statistics for numeric variables."""
        return self._data[self.numerics()].describe()

    # -------------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------------

    def frequency(
        self,
        variable: str,
        weight: Optional[str] = None,
        **kwargs,
    ) -> "FrequencyResult":
        """
        Calculate frequency distribution for a variable.

        Parameters
        ----------
        variable : str
            Variable name.
        weight : str, optional
            Weight variable. Uses dataset default if not specified.
        **kwargs
            Additional options passed to frequency function.

        Returns
        -------
        FrequencyResult
            Frequency table with counts and percentages.

        Example
        -------
        >>> freq = ds.frequency("q1")
        >>> print(freq.table)
        >>> freq.to_excel("q1_freq.xlsx")
        """
        from quantipymrx.analysis.frequency import frequency

        return frequency(
            self,
            variable,
            weight=weight or self._weight,
            **kwargs,
        )

    def crosstab(
        self,
        x: str,
        y: str,
        weight: Optional[str] = None,
        **kwargs,
    ) -> "CrosstabResult":
        """
        Calculate crosstab between two variables.

        Parameters
        ----------
        x : str
            Row variable.
        y : str
            Column variable.
        weight : str, optional
            Weight variable.
        **kwargs
            Additional options.

        Returns
        -------
        CrosstabResult
            Cross-tabulation with counts and percentages.

        Example
        -------
        >>> xtab = ds.crosstab("q1", "gender")
        >>> print(xtab.table)
        """
        from quantipymrx.analysis.crosstab import crosstab

        return crosstab(
            self,
            x=x,
            y=y,
            weight=weight or self._weight,
            **kwargs,
        )

    def frequencies_all(
        self,
        variables: Optional[List[str]] = None,
        weight: Optional[str] = None,
    ) -> Dict[str, "FrequencyResult"]:
        """
        Calculate frequencies for multiple variables.

        Parameters
        ----------
        variables : list of str, optional
            Variables to analyze. If None, uses all categorical.
        weight : str, optional
            Weight variable.

        Returns
        -------
        dict
            {variable_name: FrequencyResult}
        """
        if variables is None:
            variables = self.singles() + self.multis()

        return {
            var: self.frequency(var, weight=weight)
            for var in variables
            if var in self
        }

    # -------------------------------------------------------------------------
    # Export methods
    # -------------------------------------------------------------------------

    def to_spss(self, filepath: Union[str, Path], **kwargs) -> None:
        """Export to SPSS .sav file."""
        from quantipymrx.io.spss import write_spss
        write_spss(filepath, self._data, self._meta, self.text_key, **kwargs)

    def to_csv(self, filepath: Union[str, Path], **kwargs) -> None:
        """Export data to CSV."""
        self._data.to_csv(filepath, index=False, **kwargs)

    def to_excel(
        self,
        filepath: Union[str, Path],
        include_labels: bool = True,
        **kwargs,
    ) -> None:
        """Export to Excel with optional labels."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            self._data.to_excel(writer, sheet_name='Data', index=False, **kwargs)

            if include_labels:
                # Create labels sheet
                labels = pd.DataFrame([
                    {
                        'Variable': name,
                        'Label': meta.get_label(self.text_key),
                        'Type': meta.type,
                        'Values': str(meta.codes) if meta.values else ''
                    }
                    for name, meta in self._meta.columns.items()
                ])
                labels.to_excel(writer, sheet_name='Labels', index=False)

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DataSet(name='{self.name}', "
            f"cases={self.n_cases}, "
            f"variables={self.n_variables})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def info(self) -> str:
        """Get detailed dataset information."""
        lines = [
            f"DataSet: {self.name}",
            f"{'=' * 40}",
            f"Cases: {self.n_cases:,}",
            f"Variables: {self.n_variables}",
            f"  - Single choice: {len(self.singles())}",
            f"  - Multiple choice: {len(self.multis())}",
            f"  - Numeric: {len(self.numerics())}",
            f"  - Text: {len(self.texts())}",
            f"Weight: {self._weight or 'None'}",
            f"Text key: {self.text_key}",
        ]
        return "\n".join(lines)
