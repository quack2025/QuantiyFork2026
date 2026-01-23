"""
Tests for frequency analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest

from quantipymrx.core.dataset import DataSet
from quantipymrx.core.metadata import VariableType, ValueLabel, VariableMeta
from quantipymrx.analysis.frequency import frequency, frequencies_batch, FrequencyResult


class TestFrequency:
    """Tests for frequency calculations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n = 1000

        data = pd.DataFrame({
            'gender': np.random.choice([1, 2], n, p=[0.48, 0.52]),
            'age_group': np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2]),
            'satisfaction': np.random.choice([1, 2, 3, 4, 5], n),
            'income': np.random.normal(50000, 15000, n),
            'weight': np.random.uniform(0.5, 2.0, n),
        })

        ds = DataSet.from_dataframe(data, name="test")

        # Add proper metadata
        ds._meta.columns['gender'] = VariableMeta(
            name='gender',
            type=VariableType.SINGLE,
            text={'en-GB': 'Gender'},
            values=[
                ValueLabel(value=1, text={'en-GB': 'Male'}),
                ValueLabel(value=2, text={'en-GB': 'Female'}),
            ]
        )

        ds._meta.columns['age_group'] = VariableMeta(
            name='age_group',
            type=VariableType.SINGLE,
            text={'en-GB': 'Age Group'},
            values=[
                ValueLabel(value=1, text={'en-GB': '18-24'}),
                ValueLabel(value=2, text={'en-GB': '25-34'}),
                ValueLabel(value=3, text={'en-GB': '35-44'}),
                ValueLabel(value=4, text={'en-GB': '45+'}),
            ]
        )

        ds._meta.columns['satisfaction'] = VariableMeta(
            name='satisfaction',
            type=VariableType.SINGLE,
            text={'en-GB': 'Satisfaction'},
            values=[
                ValueLabel(value=1, text={'en-GB': 'Very Dissatisfied'}),
                ValueLabel(value=2, text={'en-GB': 'Dissatisfied'}),
                ValueLabel(value=3, text={'en-GB': 'Neutral'}),
                ValueLabel(value=4, text={'en-GB': 'Satisfied'}),
                ValueLabel(value=5, text={'en-GB': 'Very Satisfied'}),
            ]
        )

        ds._meta.columns['income'] = VariableMeta(
            name='income',
            type=VariableType.FLOAT,
            text={'en-GB': 'Annual Income'},
        )

        return ds

    def test_basic_frequency(self, sample_dataset):
        """Test basic frequency calculation."""
        result = frequency(sample_dataset, 'gender')

        assert isinstance(result, FrequencyResult)
        assert result.variable == 'gender'
        assert result.base == 1000
        assert len(result.table) == 2  # Male, Female

    def test_frequency_percentages(self, sample_dataset):
        """Test frequency percentages sum to 100."""
        result = frequency(sample_dataset, 'gender')

        total_pct = result.percentages.sum()
        assert abs(total_pct - 100) < 0.1  # Allow small rounding error

    def test_frequency_counts(self, sample_dataset):
        """Test frequency counts sum to base."""
        result = frequency(sample_dataset, 'age_group')

        total_count = result.counts.sum()
        assert total_count == result.base

    def test_weighted_frequency(self, sample_dataset):
        """Test weighted frequency calculation."""
        result = frequency(sample_dataset, 'gender', weight='weight')

        assert result.is_weighted
        assert result.weighted_base is not None
        assert result.weighted_base != result.base  # Should be different

    def test_frequency_with_labels(self, sample_dataset):
        """Test frequency shows value labels."""
        result = frequency(sample_dataset, 'gender', show_labels=True)

        # Index should have labels
        assert 'Male' in result.table.index or 'Female' in result.table.index

    def test_frequency_numeric_stats(self, sample_dataset):
        """Test frequency calculates mean/std for numeric."""
        result = frequency(sample_dataset, 'income')

        assert result.mean is not None
        assert result.std is not None
        assert result.mean > 0  # Income should be positive

    def test_frequency_categorical_mean(self, sample_dataset):
        """Test mean calculation for categorical (mean of codes)."""
        result = frequency(sample_dataset, 'satisfaction')

        assert result.mean is not None
        assert 1 <= result.mean <= 5  # Mean should be within scale range

    def test_frequency_variable_not_found(self, sample_dataset):
        """Test error when variable doesn't exist."""
        with pytest.raises(ValueError):
            frequency(sample_dataset, 'nonexistent_var')

    def test_frequency_to_dict(self, sample_dataset):
        """Test conversion to dictionary."""
        result = frequency(sample_dataset, 'gender')
        d = result.to_dict()

        assert 'variable' in d
        assert 'base' in d
        assert 'data' in d
        assert d['variable'] == 'gender'


class TestFrequencyBatch:
    """Tests for batch frequency calculations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        np.random.seed(42)
        data = pd.DataFrame({
            'q1': np.random.choice([1, 2, 3], 100),
            'q2': np.random.choice([1, 2, 3, 4], 100),
            'q3': np.random.choice([1, 2], 100),
        })

        ds = DataSet.from_dataframe(data)

        # Set as single type
        for var in ['q1', 'q2', 'q3']:
            ds._meta.columns[var].type = VariableType.SINGLE

        return ds

    def test_frequencies_batch(self, sample_dataset):
        """Test batch frequency calculation."""
        results = frequencies_batch(sample_dataset, ['q1', 'q2', 'q3'])

        assert len(results) == 3
        assert all(isinstance(r, FrequencyResult) for r in results.values())

    def test_frequencies_all(self, sample_dataset):
        """Test frequencies_all on dataset."""
        results = sample_dataset.frequencies_all()

        assert len(results) >= 3


class TestDelimitedSetFrequency:
    """Tests for multiple-choice (delimited set) frequency."""

    @pytest.fixture
    def multi_dataset(self):
        """Create dataset with delimited set variable."""
        data = pd.DataFrame({
            'brands': ['1;2;', '1;3;', '2;3;4;', '1;', '2;4;', np.nan, '1;2;3;4;', '3;', '4;', '1;2;'],
        })

        ds = DataSet.from_dataframe(data)
        ds._meta.columns['brands'] = VariableMeta(
            name='brands',
            type=VariableType.DELIMITED_SET,
            text={'en-GB': 'Brands Purchased'},
            values=[
                ValueLabel(value=1, text={'en-GB': 'Brand A'}),
                ValueLabel(value=2, text={'en-GB': 'Brand B'}),
                ValueLabel(value=3, text={'en-GB': 'Brand C'}),
                ValueLabel(value=4, text={'en-GB': 'Brand D'}),
            ]
        )

        return ds

    def test_delimited_set_frequency(self, multi_dataset):
        """Test frequency for delimited set."""
        result = frequency(multi_dataset, 'brands')

        assert result.base == 9  # 10 rows - 1 NaN
        assert len(result.table) == 4  # 4 brands

    def test_delimited_set_percentages(self, multi_dataset):
        """Test percentages can exceed 100% for multi-choice."""
        result = frequency(multi_dataset, 'brands')

        # Sum of percentages can exceed 100 for multiple choice
        total_pct = result.percentages.sum()
        # Each respondent can select multiple, so total > 100 is valid
        assert total_pct > 0
