"""
Tests for crosstab analysis functionality.
"""

import numpy as np
import pandas as pd
import pytest

from quantipymrx.core.dataset import DataSet
from quantipymrx.core.metadata import VariableType, ValueLabel, VariableMeta
from quantipymrx.analysis.crosstab import crosstab, crosstab_batch, CrosstabResult


class TestCrosstab:
    """Tests for crosstab calculations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n = 500

        # Create data with some correlation for significance testing
        gender = np.random.choice([1, 2], n, p=[0.5, 0.5])
        # Make satisfaction correlated with gender for significance
        satisfaction = np.where(
            gender == 1,
            np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.15, 0.25, 0.3, 0.2]),
            np.random.choice([1, 2, 3, 4, 5], n, p=[0.15, 0.2, 0.3, 0.25, 0.1])
        )

        data = pd.DataFrame({
            'gender': gender,
            'age_group': np.random.choice([1, 2, 3], n),
            'satisfaction': satisfaction,
            'region': np.random.choice([1, 2, 3, 4], n),
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
                ValueLabel(value=1, text={'en-GB': '18-34'}),
                ValueLabel(value=2, text={'en-GB': '35-54'}),
                ValueLabel(value=3, text={'en-GB': '55+'}),
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

        ds._meta.columns['region'] = VariableMeta(
            name='region',
            type=VariableType.SINGLE,
            text={'en-GB': 'Region'},
            values=[
                ValueLabel(value=1, text={'en-GB': 'North'}),
                ValueLabel(value=2, text={'en-GB': 'South'}),
                ValueLabel(value=3, text={'en-GB': 'East'}),
                ValueLabel(value=4, text={'en-GB': 'West'}),
            ]
        )

        return ds

    def test_basic_crosstab(self, sample_dataset):
        """Test basic crosstab calculation."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender')

        assert isinstance(result, CrosstabResult)
        assert result.x == 'satisfaction'
        assert result.y == 'gender'
        assert result.total_base == 500

    def test_crosstab_dimensions(self, sample_dataset):
        """Test crosstab has correct dimensions."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender')

        # 5 satisfaction levels x (2 genders + Total)
        assert result.counts.shape[0] == 5
        assert result.counts.shape[1] == 3  # Male, Female, Total

    def test_crosstab_col_percentages(self, sample_dataset):
        """Test column percentages sum to 100."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender')

        for col in result.col_pct.columns:
            col_sum = result.col_pct[col].sum()
            assert abs(col_sum - 100) < 0.5, f"Column {col} sums to {col_sum}"

    def test_crosstab_counts_match_base(self, sample_dataset):
        """Test counts match total base."""
        result = crosstab(sample_dataset, x='age_group', y='gender')

        # Total column should sum to total base
        if 'Total' in result.counts.columns:
            total = result.counts['Total'].sum()
            assert total == result.total_base

    def test_weighted_crosstab(self, sample_dataset):
        """Test weighted crosstab calculation."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender', weight='weight')

        assert result.is_weighted
        # Weighted totals should be different from unweighted
        unweighted = crosstab(sample_dataset, x='satisfaction', y='gender')
        assert result.counts.sum().sum() != unweighted.counts.sum().sum()

    def test_crosstab_with_labels(self, sample_dataset):
        """Test crosstab shows value labels."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender', show_labels=True)

        # Check index has labels
        assert 'Satisfied' in result.counts.index or 'Neutral' in result.counts.index

        # Check columns have labels
        assert 'Male' in result.counts.columns or 'Female' in result.counts.columns

    def test_chi_square_test(self, sample_dataset):
        """Test chi-square calculation."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender', test_significance=True)

        assert result.chi2 is not None
        assert result.chi2_pvalue is not None
        assert 0 <= result.chi2_pvalue <= 1

    def test_significance_letters(self, sample_dataset):
        """Test significance lettering."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender', test_significance=True)

        if result.significance is not None:
            # Significance matrix should have same shape as counts (minus Total)
            assert result.significance.shape[0] == result.counts.shape[0]

    def test_crosstab_variable_not_found(self, sample_dataset):
        """Test error when variable doesn't exist."""
        with pytest.raises(ValueError):
            crosstab(sample_dataset, x='nonexistent', y='gender')

    def test_crosstab_to_dict(self, sample_dataset):
        """Test conversion to dictionary."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender')
        d = result.to_dict()

        assert 'x' in d
        assert 'y' in d
        assert 'counts' in d
        assert 'col_pct' in d
        assert d['total_base'] == 500

    def test_crosstab_no_totals(self, sample_dataset):
        """Test crosstab without totals column."""
        result = crosstab(sample_dataset, x='satisfaction', y='gender', include_totals=False)

        assert 'Total' not in result.counts.columns


class TestCrosstabBatch:
    """Tests for batch crosstab calculations."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        np.random.seed(42)
        data = pd.DataFrame({
            'q1': np.random.choice([1, 2, 3], 200),
            'q2': np.random.choice([1, 2, 3, 4], 200),
            'q3': np.random.choice([1, 2], 200),
            'banner': np.random.choice([1, 2], 200),
        })

        ds = DataSet.from_dataframe(data)

        for var in ['q1', 'q2', 'q3', 'banner']:
            ds._meta.columns[var].type = VariableType.SINGLE

        return ds

    def test_crosstab_batch(self, sample_dataset):
        """Test batch crosstab calculation."""
        results = crosstab_batch(
            sample_dataset,
            x_vars=['q1', 'q2', 'q3'],
            y_var='banner'
        )

        assert len(results) == 3
        assert all(isinstance(r, CrosstabResult) for r in results.values())
        assert all(r.y == 'banner' for r in results.values())


class TestCrosstabStatistics:
    """Tests for statistical calculations in crosstab."""

    @pytest.fixture
    def significant_data(self):
        """Create data with known significant difference."""
        np.random.seed(42)

        # Create strong difference between groups
        group_a = np.random.choice([1, 2], 200, p=[0.8, 0.2])  # 80% choose 1
        group_b = np.random.choice([1, 2], 200, p=[0.2, 0.8])  # 80% choose 2

        data = pd.DataFrame({
            'response': np.concatenate([group_a, group_b]),
            'group': np.concatenate([np.ones(200), np.ones(200) * 2]),
        })

        ds = DataSet.from_dataframe(data)

        ds._meta.columns['response'] = VariableMeta(
            name='response',
            type=VariableType.SINGLE,
            values=[
                ValueLabel(value=1, text={'en-GB': 'Yes'}),
                ValueLabel(value=2, text={'en-GB': 'No'}),
            ]
        )

        ds._meta.columns['group'] = VariableMeta(
            name='group',
            type=VariableType.SINGLE,
            values=[
                ValueLabel(value=1, text={'en-GB': 'Group A'}),
                ValueLabel(value=2, text={'en-GB': 'Group B'}),
            ]
        )

        return ds

    def test_detects_significant_difference(self, significant_data):
        """Test chi-square detects significant difference."""
        result = crosstab(significant_data, x='response', y='group')

        # Should be highly significant
        assert result.chi2_pvalue is not None
        assert result.chi2_pvalue < 0.001  # Very significant

    def test_significance_letters_present(self, significant_data):
        """Test significance letters are generated."""
        result = crosstab(significant_data, x='response', y='group', sig_level=0.05)

        if result.significance is not None:
            # At least one cell should have a significance letter
            has_letters = result.significance.apply(lambda x: x.str.len() > 0).any().any()
            assert has_letters, "Expected significance letters for significant difference"
