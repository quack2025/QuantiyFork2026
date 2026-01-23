"""
Tests for SPSS reading/writing functionality.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from quantipymrx.core.metadata import VariableType, ValueLabel, VariableMeta
from quantipymrx.core.dataset import DataSet


# Test data path (uses existing Quantipy test data)
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests"
SAMPLE_SAV = TEST_DATA_DIR / "Example Data (A).sav"


class TestSPSSReader:
    """Tests for SPSS file reading."""

    @pytest.fixture
    def sample_dataset(self):
        """Load sample SPSS file."""
        if not SAMPLE_SAV.exists():
            pytest.skip(f"Test file not found: {SAMPLE_SAV}")
        return DataSet.from_spss(SAMPLE_SAV)

    def test_load_spss_file(self, sample_dataset):
        """Test basic SPSS file loading."""
        assert sample_dataset is not None
        assert sample_dataset.n_cases > 0
        assert sample_dataset.n_variables > 0

    def test_metadata_extracted(self, sample_dataset):
        """Test that metadata is extracted."""
        assert sample_dataset.meta is not None
        assert len(sample_dataset.variables) > 0

    def test_variable_labels(self, sample_dataset):
        """Test variable labels are extracted."""
        # Check at least some variables have labels
        labeled = [
            v for v in sample_dataset.variables
            if sample_dataset.label(v) != v
        ]
        assert len(labeled) > 0, "Expected some variables to have labels"

    def test_value_labels(self, sample_dataset):
        """Test value labels for categorical variables."""
        singles = sample_dataset.singles()
        if singles:
            var = singles[0]
            labels = sample_dataset.value_labels(var)
            # Categorical variables should have value labels
            assert len(labels) > 0, f"Expected value labels for {var}"

    def test_variable_types_detected(self, sample_dataset):
        """Test variable types are correctly detected."""
        # Should have at least some categorical variables
        assert len(sample_dataset.singles()) > 0 or len(sample_dataset.multis()) > 0

    def test_data_shape_matches_meta(self, sample_dataset):
        """Test data dimensions match metadata."""
        n_cases = sample_dataset.n_cases
        n_vars = len(sample_dataset.data.columns)
        assert sample_dataset.meta.info.case_count == n_cases


class TestSPSSReaderEdgeCases:
    """Test edge cases for SPSS reading."""

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            DataSet.from_spss("nonexistent_file.sav")

    def test_custom_text_key(self):
        """Test loading with custom text key."""
        if not SAMPLE_SAV.exists():
            pytest.skip(f"Test file not found: {SAMPLE_SAV}")

        ds = DataSet.from_spss(SAMPLE_SAV, text_key="es-ES")
        assert ds.text_key == "es-ES"


class TestDataSetFromDataFrame:
    """Tests for creating DataSet from DataFrame."""

    @pytest.fixture
    def simple_df(self):
        """Create simple test DataFrame."""
        return pd.DataFrame({
            'id': range(1, 101),
            'gender': np.random.choice([1, 2], 100),
            'age': np.random.randint(18, 65, 100),
            'score': np.random.random(100) * 100,
            'comment': ['Comment ' + str(i) for i in range(100)],
        })

    def test_create_from_dataframe(self, simple_df):
        """Test creating DataSet from DataFrame."""
        ds = DataSet.from_dataframe(simple_df, name="test")
        assert ds.n_cases == 100
        assert ds.n_variables == 5

    def test_infer_numeric_types(self, simple_df):
        """Test numeric types are inferred."""
        ds = DataSet.from_dataframe(simple_df)

        # Integer column
        assert ds.var_type('id') in [VariableType.INT, VariableType.FLOAT]
        # Float column
        assert ds.var_type('score') == VariableType.FLOAT

    def test_infer_string_type(self, simple_df):
        """Test string type is inferred."""
        ds = DataSet.from_dataframe(simple_df)
        assert ds.var_type('comment') == VariableType.STRING


class TestMetadataModels:
    """Tests for metadata Pydantic models."""

    def test_value_label_creation(self):
        """Test ValueLabel model."""
        vl = ValueLabel(value=1, text={"en-GB": "Yes", "es-ES": "Sí"})
        assert vl.value == 1
        assert vl.get_text("en-GB") == "Yes"
        assert vl.get_text("es-ES") == "Sí"
        assert vl.get_text("fr-FR") == "Yes"  # Fallback to first

    def test_variable_meta_codes(self):
        """Test VariableMeta codes property."""
        values = [
            ValueLabel(value=1, text={"en-GB": "Yes"}),
            ValueLabel(value=2, text={"en-GB": "No"}),
        ]
        vm = VariableMeta(
            name="q1",
            type=VariableType.SINGLE,
            text={"en-GB": "Question 1"},
            values=values,
        )
        assert vm.codes == [1, 2]
        assert vm.is_categorical
        assert not vm.is_numeric

    def test_variable_meta_numeric(self):
        """Test numeric VariableMeta."""
        vm = VariableMeta(
            name="age",
            type=VariableType.INT,
            text={"en-GB": "Age"},
        )
        assert not vm.is_categorical
        assert vm.is_numeric
        assert vm.codes == []
