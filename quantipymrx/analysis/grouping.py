# -*- coding: utf-8 -*-
"""
QuantipyMRX - Grouping Engine

Combines related variables into analysis-ready formats:
- Top of Mind: Multiple mentions → single frequency table
- Awareness: Brand awareness grid → awareness summary
- Grids: Scale questions → summary statistics

This module transforms detected groups from auto_detect into
actionable analysis outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.analysis.auto_detect import DetectedGroup, ProcessingSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class GroupedFrequency:
    """
    Result of combining multiple variables into a single frequency table.

    Used for Top of Mind, Awareness, and similar question types.
    """
    group_name: str
    display_name: str
    question_type: str
    variables: List[str]

    # Frequency data
    frequency_table: pd.DataFrame  # Code, Label, Count, Percent
    total_base: int
    total_responses: int  # Sum of all mentions

    # Calculated metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "group_name": self.group_name,
            "display_name": self.display_name,
            "question_type": self.question_type,
            "variables": self.variables,
            "total_base": self.total_base,
            "total_responses": self.total_responses,
            "frequency_table": self.frequency_table.to_dict(orient="records"),
            "metrics": self.metrics,
        }


@dataclass
class GridSummary:
    """
    Result of summarizing a grid question.

    Contains statistics for each item in the grid.
    """
    group_name: str
    display_name: str
    question_type: str
    variables: List[str]
    item_labels: List[str]

    # Summary statistics per item
    summary_table: pd.DataFrame  # Item, Base, Mean, StdDev, Top2Box, etc.

    # Overall statistics
    overall_mean: Optional[float] = None
    scale_min: int = 1
    scale_max: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "group_name": self.group_name,
            "display_name": self.display_name,
            "question_type": self.question_type,
            "variables": self.variables,
            "item_labels": self.item_labels,
            "summary_table": self.summary_table.to_dict(orient="records"),
            "overall_mean": self.overall_mean,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
        }


@dataclass
class TopOfMindSummary:
    """
    Result of Top of Mind / Mention analysis.

    Shows first mention, any mention, and total mentions.
    """
    group_name: str
    display_name: str
    variables: List[str]

    # Summary tables
    first_mention: pd.DataFrame  # Brand, Count, Percent
    any_mention: pd.DataFrame    # Brand, Count, Percent
    total_mentions: pd.DataFrame # Brand, Total Mentions

    # Base information
    total_base: int
    total_mentioners: int  # Respondents who mentioned at least one

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "group_name": self.group_name,
            "display_name": self.display_name,
            "variables": self.variables,
            "first_mention": self.first_mention.to_dict(orient="records"),
            "any_mention": self.any_mention.to_dict(orient="records"),
            "total_mentions": self.total_mentions.to_dict(orient="records"),
            "total_base": self.total_base,
            "total_mentioners": self.total_mentioners,
        }


@dataclass
class AwarenessSummary:
    """
    Result of brand awareness analysis.

    Shows awareness rates for each brand/item.
    """
    group_name: str
    display_name: str
    variables: List[str]
    item_labels: List[str]

    # Awareness table
    awareness_table: pd.DataFrame  # Item, Base, Aware Count, Awareness %

    # Overall metrics
    total_base: int
    average_awareness: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "group_name": self.group_name,
            "display_name": self.display_name,
            "variables": self.variables,
            "item_labels": self.item_labels,
            "awareness_table": self.awareness_table.to_dict(orient="records"),
            "total_base": self.total_base,
            "average_awareness": self.average_awareness,
        }


# =============================================================================
# Grouping Engine
# =============================================================================

class GroupingEngine:
    """
    Engine for combining variable groups into analysis outputs.

    Takes detected groups from auto_detect and produces
    analysis-ready summaries.
    """

    def __init__(self, dataset: "DataSet"):
        """
        Initialize the GroupingEngine.

        Parameters
        ----------
        dataset : DataSet
            The dataset to analyze.
        """
        self.dataset = dataset
        self.data = dataset.data
        self.meta = dataset.meta
        self.text_key = dataset.text_key

    def process_group(
        self,
        group: "DetectedGroup",
        weight: Optional[str] = None,
    ) -> Union[GroupedFrequency, GridSummary, TopOfMindSummary, AwarenessSummary]:
        """
        Process a detected group and return appropriate summary.

        Parameters
        ----------
        group : DetectedGroup
            The group to process (from auto_detect).
        weight : str, optional
            Weight variable name.

        Returns
        -------
        GroupedFrequency, GridSummary, TopOfMindSummary, or AwarenessSummary
            The appropriate summary type for the group.
        """
        from quantipymrx.analysis.auto_detect import QuestionType

        qtype = group.question_type

        if qtype == QuestionType.TOP_OF_MIND:
            return self.top_of_mind(group, weight)
        elif qtype == QuestionType.AWARENESS:
            return self.awareness(group, weight)
        elif qtype in [QuestionType.SCALE, QuestionType.NPS]:
            return self.grid_summary(group, weight)
        elif qtype == QuestionType.GRID_SINGLE:
            return self.grid_summary(group, weight)
        else:
            return self.grouped_frequency(group, weight)

    def grouped_frequency(
        self,
        group: "DetectedGroup",
        weight: Optional[str] = None,
    ) -> GroupedFrequency:
        """
        Create a combined frequency table from multiple variables.

        Useful for multiple choice questions stored as separate columns.

        Parameters
        ----------
        group : DetectedGroup
            The group to process.
        weight : str, optional
            Weight variable name.

        Returns
        -------
        GroupedFrequency
        """
        variables = group.variables

        # Get weights
        if weight and weight in self.data.columns:
            weights = self.data[weight].fillna(1)
        else:
            weights = pd.Series(1, index=self.data.index)

        # Combine all responses
        all_responses = []

        for var in variables:
            if var not in self.data.columns:
                continue

            var_meta = self.meta.get_variable(var)
            var_data = self.data[var].dropna()

            for idx in var_data.index:
                value = var_data[idx]
                w = weights.get(idx, 1)
                all_responses.append({
                    'variable': var,
                    'value': value,
                    'weight': w,
                })

        if not all_responses:
            # Return empty result
            return GroupedFrequency(
                group_name=group.name,
                display_name=group.display_name,
                question_type=group.question_type.value,
                variables=variables,
                frequency_table=pd.DataFrame(),
                total_base=0,
                total_responses=0,
            )

        # Create DataFrame and aggregate
        df = pd.DataFrame(all_responses)
        total_base = int(len(self.data))
        total_responses = int(df['weight'].sum())

        # Group by value
        freq_df = df.groupby('value').agg({
            'weight': 'sum'
        }).reset_index()

        freq_df.columns = ['code', 'count']
        freq_df['percent'] = (freq_df['count'] / total_base * 100).round(1)

        # Add labels
        first_var = variables[0]
        var_meta = self.meta.get_variable(first_var)
        if var_meta and var_meta.values:
            code_to_label = {v.value: v.get_text(self.text_key) for v in var_meta.values}
            freq_df['label'] = freq_df['code'].map(code_to_label)
        else:
            freq_df['label'] = freq_df['code'].astype(str)

        # Reorder columns
        freq_df = freq_df[['code', 'label', 'count', 'percent']]
        freq_df = freq_df.sort_values('count', ascending=False)

        return GroupedFrequency(
            group_name=group.name,
            display_name=group.display_name,
            question_type=group.question_type.value,
            variables=variables,
            frequency_table=freq_df,
            total_base=total_base,
            total_responses=total_responses,
        )

    def top_of_mind(
        self,
        group: "DetectedGroup",
        weight: Optional[str] = None,
    ) -> TopOfMindSummary:
        """
        Analyze Top of Mind / Multiple Mention question.

        Calculates:
        - First mention (only the first variable)
        - Any mention (mentioned in any variable)
        - Total mentions (sum across all variables)

        Parameters
        ----------
        group : DetectedGroup
            The group to process.
        weight : str, optional
            Weight variable name.

        Returns
        -------
        TopOfMindSummary
        """
        variables = group.variables

        # Get weights
        if weight and weight in self.data.columns:
            weights = self.data[weight].fillna(1)
        else:
            weights = pd.Series(1, index=self.data.index)

        total_base = int(weights.sum())

        # Get value labels from first variable
        first_var_meta = self.meta.get_variable(variables[0])
        if first_var_meta and first_var_meta.values:
            all_codes = [v.value for v in first_var_meta.values]
            code_to_label = {v.value: v.get_text(self.text_key) for v in first_var_meta.values}
        else:
            # Infer codes from data
            all_codes = set()
            for var in variables:
                if var in self.data.columns:
                    all_codes.update(self.data[var].dropna().unique())
            all_codes = sorted(all_codes)
            code_to_label = {c: str(c) for c in all_codes}

        # === First Mention ===
        first_var = variables[0]
        if first_var in self.data.columns:
            first_counts = {}
            for code in all_codes:
                mask = self.data[first_var] == code
                first_counts[code] = weights[mask].sum()

            first_df = pd.DataFrame([
                {
                    'code': code,
                    'label': code_to_label.get(code, str(code)),
                    'count': first_counts.get(code, 0),
                    'percent': round(first_counts.get(code, 0) / total_base * 100, 1) if total_base > 0 else 0,
                }
                for code in all_codes
            ])
            first_df = first_df.sort_values('count', ascending=False)
        else:
            first_df = pd.DataFrame(columns=['code', 'label', 'count', 'percent'])

        # === Any Mention ===
        any_counts = {}
        for code in all_codes:
            # Check if code appears in any of the variables
            mentioned = pd.Series(False, index=self.data.index)
            for var in variables:
                if var in self.data.columns:
                    mentioned = mentioned | (self.data[var] == code)
            any_counts[code] = weights[mentioned].sum()

        any_df = pd.DataFrame([
            {
                'code': code,
                'label': code_to_label.get(code, str(code)),
                'count': any_counts.get(code, 0),
                'percent': round(any_counts.get(code, 0) / total_base * 100, 1) if total_base > 0 else 0,
            }
            for code in all_codes
        ])
        any_df = any_df.sort_values('count', ascending=False)

        # === Total Mentions ===
        total_counts = {}
        for code in all_codes:
            total = 0
            for var in variables:
                if var in self.data.columns:
                    mask = self.data[var] == code
                    total += weights[mask].sum()
            total_counts[code] = total

        total_df = pd.DataFrame([
            {
                'code': code,
                'label': code_to_label.get(code, str(code)),
                'total_mentions': total_counts.get(code, 0),
            }
            for code in all_codes
        ])
        total_df = total_df.sort_values('total_mentions', ascending=False)

        # Count mentioners (anyone who mentioned at least one item)
        has_mention = pd.Series(False, index=self.data.index)
        for var in variables:
            if var in self.data.columns:
                has_mention = has_mention | self.data[var].notna()
        total_mentioners = int(weights[has_mention].sum())

        return TopOfMindSummary(
            group_name=group.name,
            display_name=group.display_name,
            variables=variables,
            first_mention=first_df,
            any_mention=any_df,
            total_mentions=total_df,
            total_base=total_base,
            total_mentioners=total_mentioners,
        )

    def awareness(
        self,
        group: "DetectedGroup",
        weight: Optional[str] = None,
    ) -> AwarenessSummary:
        """
        Analyze awareness grid.

        Each variable represents a brand/item, values are typically Yes/No.

        Parameters
        ----------
        group : DetectedGroup
            The group to process.
        weight : str, optional
            Weight variable name.

        Returns
        -------
        AwarenessSummary
        """
        variables = group.variables

        # Get weights
        if weight and weight in self.data.columns:
            weights = self.data[weight].fillna(1)
        else:
            weights = pd.Series(1, index=self.data.index)

        total_base = int(weights.sum())

        # Build awareness table
        rows = []
        item_labels = []

        for var in variables:
            if var not in self.data.columns:
                continue

            var_meta = self.meta.get_variable(var)
            item_label = var_meta.get_label(self.text_key) if var_meta else var
            item_labels.append(item_label)

            # Calculate awareness (typically code 1 = Yes/Aware)
            var_data = self.data[var]

            # Find the "positive" code (usually 1 or the first code)
            positive_codes = [1]  # Default
            if var_meta and var_meta.values:
                # Check for binary Yes/No
                for v in var_meta.values:
                    label_lower = v.get_text(self.text_key).lower()
                    if any(p in label_lower for p in ['yes', 'si', 'sí', 'sim', 'oui', 'ja']):
                        positive_codes = [v.value]
                        break

            # Count aware
            mask = var_data.isin(positive_codes)
            aware_count = weights[mask].sum()
            base_for_var = weights[var_data.notna()].sum()

            awareness_pct = (aware_count / base_for_var * 100) if base_for_var > 0 else 0

            rows.append({
                'variable': var,
                'item_label': item_label,
                'base': int(base_for_var),
                'aware_count': int(aware_count),
                'awareness_pct': round(awareness_pct, 1),
            })

        awareness_df = pd.DataFrame(rows)
        awareness_df = awareness_df.sort_values('awareness_pct', ascending=False)

        # Calculate average awareness
        if len(rows) > 0:
            avg_awareness = awareness_df['awareness_pct'].mean()
        else:
            avg_awareness = 0.0

        return AwarenessSummary(
            group_name=group.name,
            display_name=group.display_name,
            variables=variables,
            item_labels=item_labels,
            awareness_table=awareness_df,
            total_base=total_base,
            average_awareness=round(avg_awareness, 1),
        )

    def grid_summary(
        self,
        group: "DetectedGroup",
        weight: Optional[str] = None,
    ) -> GridSummary:
        """
        Summarize a grid/scale question.

        Calculates mean, std dev, top 2 box, bottom 2 box for each item.

        Parameters
        ----------
        group : DetectedGroup
            The group to process.
        weight : str, optional
            Weight variable name.

        Returns
        -------
        GridSummary
        """
        variables = group.variables

        # Get weights
        if weight and weight in self.data.columns:
            weights = self.data[weight].fillna(1)
        else:
            weights = pd.Series(1, index=self.data.index)

        # Determine scale range from first variable
        first_var_meta = self.meta.get_variable(variables[0])
        if first_var_meta and first_var_meta.values:
            codes = [v.value for v in first_var_meta.values]
            scale_min = min(codes)
            scale_max = max(codes)
        else:
            scale_min = 1
            scale_max = 5

        # Calculate statistics for each item
        rows = []
        item_labels = []

        for var in variables:
            if var not in self.data.columns:
                continue

            var_meta = self.meta.get_variable(var)
            item_label = var_meta.get_label(self.text_key) if var_meta else var
            item_labels.append(item_label)

            var_data = self.data[var]

            # Convert Categorical to numeric if needed
            if hasattr(var_data, 'cat'):
                # It's a Categorical - convert to numeric codes
                var_data = pd.to_numeric(var_data, errors='coerce')
            elif var_data.dtype == object:
                # Try to convert string to numeric
                var_data = pd.to_numeric(var_data, errors='coerce')

            valid_mask = var_data.notna()
            valid_data = var_data[valid_mask].astype(float)
            valid_weights = weights[valid_mask]

            base = valid_weights.sum()

            if base > 0:
                # Weighted mean
                mean_val = np.average(valid_data, weights=valid_weights)

                # Weighted std dev
                variance = np.average((valid_data - mean_val) ** 2, weights=valid_weights)
                std_val = np.sqrt(variance)

                # Top 2 Box (top 2 scale points)
                top_2_mask = valid_data >= (scale_max - 1)
                top_2_box = valid_weights[top_2_mask].sum() / base * 100

                # Bottom 2 Box (bottom 2 scale points)
                bot_2_mask = valid_data <= (scale_min + 1)
                bot_2_box = valid_weights[bot_2_mask].sum() / base * 100

                # For NPS: calculate score
                if scale_min == 0 and scale_max == 10:
                    promoters = valid_weights[valid_data >= 9].sum()
                    detractors = valid_weights[valid_data <= 6].sum()
                    nps_score = (promoters - detractors) / base * 100
                else:
                    nps_score = None
            else:
                mean_val = std_val = top_2_box = bot_2_box = nps_score = None

            row = {
                'variable': var,
                'item_label': item_label,
                'base': int(base),
                'mean': round(mean_val, 2) if mean_val is not None else None,
                'std_dev': round(std_val, 2) if std_val is not None else None,
                'top_2_box': round(top_2_box, 1) if top_2_box is not None else None,
                'bottom_2_box': round(bot_2_box, 1) if bot_2_box is not None else None,
            }

            if nps_score is not None:
                row['nps_score'] = round(nps_score, 1)

            rows.append(row)

        summary_df = pd.DataFrame(rows)

        # Calculate overall mean
        if len(rows) > 0 and summary_df['mean'].notna().any():
            overall_mean = summary_df['mean'].mean()
        else:
            overall_mean = None

        return GridSummary(
            group_name=group.name,
            display_name=group.display_name,
            question_type=group.question_type.value,
            variables=variables,
            item_labels=item_labels,
            summary_table=summary_df,
            overall_mean=round(overall_mean, 2) if overall_mean else None,
            scale_min=scale_min,
            scale_max=scale_max,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def process_all_groups(
    dataset: "DataSet",
    spec: "ProcessingSpec",
    weight: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process all detected groups from a ProcessingSpec.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    spec : ProcessingSpec
        The processing specification from auto_detect.
    weight : str, optional
        Weight variable name. If None, uses spec.detected_weight.

    Returns
    -------
    dict
        Dictionary mapping group names to their summaries.
    """
    engine = GroupingEngine(dataset)
    weight = weight or spec.detected_weight

    results = {}
    for group in spec.variable_groups:
        try:
            results[group.name] = engine.process_group(group, weight)
        except Exception as e:
            logger.warning(f"Failed to process group {group.name}: {e}")

    return results


def combine_mentions(
    dataset: "DataSet",
    variables: List[str],
    name: str = "combined",
    weight: Optional[str] = None,
) -> TopOfMindSummary:
    """
    Combine multiple mention variables into a Top of Mind summary.

    Convenience function for creating TOM analysis without auto-detect.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variables : list of str
        Variable names to combine.
    name : str
        Name for the combined group.
    weight : str, optional
        Weight variable name.

    Returns
    -------
    TopOfMindSummary
    """
    from quantipymrx.analysis.auto_detect import DetectedGroup, QuestionType

    # Create a mock group
    group = DetectedGroup(
        name=name,
        display_name=name,
        question_type=QuestionType.TOP_OF_MIND,
        variables=variables,
        confidence=1.0,
    )

    engine = GroupingEngine(dataset)
    return engine.top_of_mind(group, weight)


def combine_awareness(
    dataset: "DataSet",
    variables: List[str],
    name: str = "awareness",
    weight: Optional[str] = None,
) -> AwarenessSummary:
    """
    Combine multiple awareness variables into an Awareness summary.

    Parameters
    ----------
    dataset : DataSet
        The dataset.
    variables : list of str
        Variable names to combine.
    name : str
        Name for the combined group.
    weight : str, optional
        Weight variable name.

    Returns
    -------
    AwarenessSummary
    """
    from quantipymrx.analysis.auto_detect import DetectedGroup, QuestionType

    group = DetectedGroup(
        name=name,
        display_name=name,
        question_type=QuestionType.AWARENESS,
        variables=variables,
        confidence=1.0,
    )

    engine = GroupingEngine(dataset)
    return engine.awareness(group, weight)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Result classes
    "GroupedFrequency",
    "GridSummary",
    "TopOfMindSummary",
    "AwarenessSummary",
    # Engine
    "GroupingEngine",
    # Functions
    "process_all_groups",
    "combine_mentions",
    "combine_awareness",
]
