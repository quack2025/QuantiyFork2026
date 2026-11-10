# -*- coding: utf-8 -*-
"""
QuantipyMRX - Auto-Detection Module for Market Research

Automatically detects patterns in survey data:
- Question types (grids, scales, top-of-mind, awareness)
- Banner candidates (demographics)
- Variable groupings
- Suggested calculations (nets, recodes, NPS, Top 2 Box)

This module is designed to help analysts quickly set up their
data processing without manual configuration.
"""

from __future__ import annotations

import re
import logging
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from quantipymrx.core.dataset import DataSet
    from quantipymrx.core.metadata import VariableMeta

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class QuestionType(str, Enum):
    """Types of market research questions."""
    SINGLE = "single"                    # Single choice
    MULTIPLE = "multiple"                # Multiple choice
    GRID_SINGLE = "grid_single"          # Grid with single choice per row
    GRID_MULTIPLE = "grid_multiple"      # Grid with multiple choice per row
    SCALE = "scale"                      # Rating scale (1-5, 1-10, etc.)
    NPS = "nps"                          # Net Promoter Score (0-10)
    TOP_OF_MIND = "top_of_mind"          # First/any mention
    AWARENESS = "awareness"              # Brand awareness (aided/unaided)
    RANKING = "ranking"                  # Ranking question
    NUMERIC = "numeric"                  # Numeric/continuous
    OPEN_END = "open_end"                # Open-ended text
    DATE = "date"                        # Date field
    DEMOGRAPHIC = "demographic"          # Demographics (age, gender, etc.)
    WEIGHT = "weight"                    # Weight variable
    ID = "id"                            # Respondent ID
    UNKNOWN = "unknown"


class BannerType(str, Enum):
    """Types of banner variables."""
    GENDER = "gender"
    AGE = "age"
    AGE_GROUP = "age_group"
    REGION = "region"
    INCOME = "income"
    EDUCATION = "education"
    OCCUPATION = "occupation"
    TOTAL = "total"
    CUSTOM = "custom"


@dataclass
class DetectedVariable:
    """Result of detecting a single variable."""
    name: str
    label: str
    question_type: QuestionType
    confidence: float  # 0.0 to 1.0
    group_name: Optional[str] = None  # If part of a group
    group_index: Optional[int] = None  # Position in group (1st mention, 2nd, etc.)
    suggested_calculations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedGroup:
    """A group of related variables (grid, top-of-mind, etc.)."""
    name: str
    display_name: str
    question_type: QuestionType
    variables: List[str]
    confidence: float
    suggested_calculations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedBanner:
    """A suggested banner variable."""
    variable: str
    label: str
    banner_type: BannerType
    confidence: float
    n_categories: int
    categories: List[str] = field(default_factory=list)


@dataclass
class ProcessingSpec:
    """Complete processing specification for a dataset."""
    dataset_name: str
    case_count: int
    detected_weight: Optional[str]
    banners: List[DetectedBanner]
    variable_groups: List[DetectedGroup]
    individual_variables: List[DetectedVariable]
    suggested_analyses: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset_name": self.dataset_name,
            "case_count": self.case_count,
            "detected_weight": self.detected_weight,
            "banners": [
                {
                    "variable": b.variable,
                    "label": b.label,
                    "type": b.banner_type.value,
                    "confidence": b.confidence,
                    "n_categories": b.n_categories,
                    "categories": b.categories,
                }
                for b in self.banners
            ],
            "variable_groups": [
                {
                    "name": g.name,
                    "display_name": g.display_name,
                    "type": g.question_type.value,
                    "variables": g.variables,
                    "confidence": g.confidence,
                    "calculations": g.suggested_calculations,
                }
                for g in self.variable_groups
            ],
            "individual_variables": [
                {
                    "name": v.name,
                    "label": v.label,
                    "type": v.question_type.value,
                    "confidence": v.confidence,
                    "calculations": v.suggested_calculations,
                }
                for v in self.individual_variables
            ],
            "suggested_analyses": self.suggested_analyses,
        }


# =============================================================================
# Pattern Detection
# =============================================================================

# Common patterns for variable name detection
PATTERNS = {
    # Grid patterns: Q1_1, Q1_2 or Q1a, Q1b or Q1.1, Q1.2
    'grid_numeric': re.compile(r'^(.+?)[-_.](\d+)$'),
    'grid_alpha': re.compile(r'^(.+?)([a-z])$', re.IGNORECASE),

    # Top of Mind: TOM_1, TOM_2 or BRAND_1ST, BRAND_2ND
    'tom_numeric': re.compile(r'^(.+?)[-_]?(\d+)(?:st|nd|rd|th)?$', re.IGNORECASE),
    'tom_mention': re.compile(r'^(.+?)[-_]?(first|second|third|1st|2nd|3rd|any)[-_]?(?:mention)?$', re.IGNORECASE),

    # Demographics
    'gender': re.compile(r'(?:^|[-_])(?:gender|sex|sexo|genero)(?:$|[-_])', re.IGNORECASE),
    'age': re.compile(r'(?:^|[-_])(?:age|edad|alter)(?:$|[-_])', re.IGNORECASE),
    'age_group': re.compile(r'(?:^|[-_])(?:age[-_]?group|edad[-_]?grupo|rango[-_]?edad|clasificacion[-_]?edad)(?:$|[-_])', re.IGNORECASE),
    'region': re.compile(r'(?:^|[-_])(?:region|zona|area|ciudad|city|state|estado|pais|country)(?:$|[-_])', re.IGNORECASE),
    'income': re.compile(r'(?:^|[-_])(?:income|ingreso|nse|nivel[-_]?socio)(?:$|[-_])', re.IGNORECASE),
    'education': re.compile(r'(?:^|[-_])(?:education|educacion|estudios|escolaridad)(?:$|[-_])', re.IGNORECASE),

    # Weight variable
    'weight': re.compile(r'(?:^|[-_])(?:weight|peso|ponderador|pond)(?:$|[-_\d])', re.IGNORECASE),

    # ID variable
    'id': re.compile(r'(?:^|[-_])(?:id|folio|respondent|encuestado|case)(?:$|[-_])', re.IGNORECASE),

    # NPS
    'nps': re.compile(r'(?:^|[-_])(?:nps|recommend|recomendar|promoter)(?:$|[-_])', re.IGNORECASE),

    # Satisfaction
    'satisfaction': re.compile(r'(?:^|[-_])(?:sat|satisf|csat)(?:$|[-_])', re.IGNORECASE),

    # Awareness
    'awareness': re.compile(r'(?:^|[-_])(?:aware|conoce|conocimiento|brand[-_]?aware)(?:$|[-_])', re.IGNORECASE),
}

# =============================================================================
# Banner detection by whole words (2026-11-10)
# =============================================================================
# `PATTERNS` above only see a word joined by `-`/`_`, so a label with spaces
# ("¿En cuál ciudad vive?") never matches, and "Género" (with an accent) does
# not match "genero". These are used ONLY to suggest banners; they never change
# a question's type.

def _fold_words(text: Optional[str]) -> List[str]:
    """Words of a name or label: no accents, lower case (`P1_EDAD` -> p1, edad)."""
    if not text:
        return []
    decomposed = unicodedata.normalize("NFKD", str(text))
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    return re.findall(r"[a-z0-9]+", plain)


def _has_phrase(words: List[str], phrase: str) -> bool:
    target = _fold_words(phrase)
    n = len(target)
    return any(words[i:i + n] == target for i in range(len(words) - n + 1))


def _any_phrase(words: List[str], phrases: List[str]) -> bool:
    return any(_has_phrase(words, p) for p in phrases)


_BANNER_TERMS: List[Tuple["BannerType", List[str]]] = [
    (BannerType.GENDER, ["sexo", "genero", "gender", "sex"]),
    (BannerType.AGE_GROUP, ["rango de edad", "rangos de edad", "grupo de edad", "grupo etario", "age group", "age band", "faixa etaria"]),
    (BannerType.AGE, ["edad", "age", "idade"]),
    (BannerType.REGION, ["ciudad", "region", "provincia", "city", "cidade", "regiao"]),
    # «ingresos» alone is ambiguous (an insurance against loss of income): only as
    # a phrase that asks about the respondent's income.
    (BannerType.INCOME, ["estrato", "nse", "nivel socioeconomico", "nivel socio economico", "clase social", "household income", "income level",
                         "nivel de ingresos", "nivel de ingreso", "ingreso familiar", "ingresos familiares", "ingreso mensual", "ingresos mensuales",
                         "ingresos del hogar", "ingreso del hogar", "renda familiar", "renda mensal", "classe social"]),
    (BannerType.EDUCATION, ["escolaridad", "nivel educativo", "nivel de estudios", "education", "escolaridade"]),
    (BannerType.OCCUPATION, ["ocupacion", "occupation", "ocupacao"]),
]
_OTHER_PERSON = {
    "hijo", "hija", "hijos", "hijas", "nino", "nina", "ninos", "bebe", "pareja", "conyuge",
    "child", "children", "daughter", "partner", "filho", "filha", "filhos", "mascota", "pet",
    "vehiculo", "carro", "auto", "car",
}
_CHECKBOX_VALUES = {
    "seleccionado", "no seleccionado", "selected", "not selected", "checked", "unchecked", "marcado", "no marcado",
    "si", "no", "yes", "sim", "nao", "0", "1",
}
_SEX_VALUES = {"hombre", "mujer", "masculino", "femenino", "male", "female", "man", "woman", "homem", "mulher"}
# Value labels shaped like age ranges: "18-24", "25 a 34", "55+", "55 o más",
# "Menos de 25", "65 and over". Matched on the folded label (no accents).
_AGE_RANGE_SHAPES = [
    re.compile(r"^(\d{2}) ?[-–] ?(\d{2})(?: anos| years)?$"),
    re.compile(r"^(\d{2}) (?:a|al|to|y|e) (\d{2})(?: anos| years)?$"),
    re.compile(r"^(\d{2}) ?(?:\+|o mas|y mas|anos o mas|and over|or more|ou mais|anos ou mais)$"),
    re.compile(r"^(?:menos de|menor de|under|less than|ate) (\d{2})(?: anos| years)?$"),
]


def _age_ranges_in_values(categories: List[str]) -> bool:
    """At least 3 value labels are age ranges (two-digit ages 12-99, no money)."""
    hits = 0
    for c in categories:
        if any(sym in c for sym in ("$", "€", "£", "%")):
            return False
        decomposed = unicodedata.normalize("NFKD", str(c))
        text = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
        text = re.sub(r"\s+", " ", text).strip().rstrip(".")
        for shape in _AGE_RANGE_SHAPES:
            m = shape.match(text)
            if m and all(12 <= int(g) <= 99 for g in m.groups() if g):
                hits += 1
                break
    return hits >= 3


def _profile_banner_type(var_name: str, label: str, categories: List[str]) -> Optional["BannerType"]:
    """The profile kind of a question by whole words of its name and label, or None."""
    words = _fold_words(var_name) + _fold_words(label)
    value_words = {w for c in categories for w in _fold_words(c)}
    about_someone_else = any(w in _OTHER_PERSON for w in words)
    # An option of a multiple-response question ("<option>: ¿Cuáles de…?",
    # values selected / not selected) is not a profile question, whatever
    # words its label has.
    folded_values = {" ".join(_fold_words(c)) for c in categories}
    if len(categories) <= 2 and folded_values & _CHECKBOX_VALUES and not (value_words & _SEX_VALUES):
        return None
    # An item of a series ("P9_6" labelled "<product>: <attribute>"): the
    # profile word is the item's text, not what the question asks about the
    # respondent.
    if re.search(r"[_.]\d+$", var_name) and ":" in (label or ""):
        return None
    for banner_type, terms in _BANNER_TERMS:
        if about_someone_else and banner_type not in (BannerType.REGION, BannerType.INCOME):
            continue
        if not _any_phrase(words, terms):
            continue
        # «Género musical»: a gender question has sex values.
        if banner_type == BannerType.GENDER and value_words and not (value_words & _SEX_VALUES):
            continue
        return banner_type
    # «¿Es usted…?» is how sex is usually asked, but so is «¿Es usted cliente?».
    if (_has_phrase(_fold_words(label), "es usted") or _has_phrase(_fold_words(label), "voce e")) and value_words & _SEX_VALUES:
        return BannerType.GENDER
    if not about_someone_else and _age_ranges_in_values(categories):
        return BannerType.AGE_GROUP
    return None


# =============================================================================
# Value Label Analysis Patterns (Multi-language support)
# =============================================================================

# Binary Yes/No patterns in multiple languages
BINARY_YES_PATTERNS = [
    # English
    r'^yes$', r'^y$', r'^si$', r'^sí$',
    # Spanish
    r'^si$', r'^sí$',
    # Portuguese
    r'^sim$',
    # French
    r'^oui$',
    # German
    r'^ja$',
    # Italian
    r'^sì$',
    # Generic
    r'^1$', r'^true$',
]

BINARY_NO_PATTERNS = [
    # English
    r'^no$', r'^n$',
    # Spanish
    r'^no$',
    # Portuguese
    r'^não$', r'^nao$',
    # French
    r'^non$',
    # German
    r'^nein$',
    # Italian
    r'^no$',
    # Generic
    r'^0$', r'^false$',
]

# Awareness-specific value labels
AWARENESS_POSITIVE_PATTERNS = [
    # English
    r'know', r'aware', r'heard', r'recognize', r'familiar',
    # Spanish
    r'conoce', r'conocí', r'escuchado', r'reconoce', r'familiar',
    r'sabe', r'recuerda', r'menciona', r'ha visto', r'ha oído',
    # Portuguese
    r'conhece', r'ouviu', r'reconhece',
    # Generic selection
    r'selected', r'seleccionado', r'marcado', r'checked',
]

AWARENESS_NEGATIVE_PATTERNS = [
    # English
    r'not know', r'don\'t know', r'never heard', r'not aware', r'unfamiliar',
    # Spanish
    r'no conoce', r'no sabe', r'no recuerda', r'nunca', r'no ha',
    # Portuguese
    r'não conhece', r'nunca ouviu',
    # Generic
    r'not selected', r'no seleccionado', r'no marcado',
]

# Question label patterns for awareness detection
AWARENESS_QUESTION_PATTERNS = [
    # English
    r'which.*know', r'which.*aware', r'which.*heard', r'do you know',
    r'have you heard', r'are you familiar', r'which.*recognize',
    # Spanish
    r'cu[aá]l.*conoce', r'cu[aá]l.*sabe', r'conoce.*marca',
    r'ha escuchado', r'ha o[ií]do', r'reconoce', r'recuerda.*marca',
    r'qu[eé].*marca.*conoce', r'de las siguientes.*conoce',
    # Portuguese
    r'qual.*conhece', r'ouviu falar',
]


def _normalize_label(label: str) -> str:
    """Normalize a label for pattern matching."""
    if not label:
        return ""
    # Lowercase, remove extra spaces, normalize accents for comparison
    return label.lower().strip()


def _is_binary_values(values: List[Any], text_key: str = "en-GB") -> bool:
    """
    Check if value labels represent binary Yes/No responses.

    Parameters
    ----------
    values : list
        List of ValueLabel objects.
    text_key : str
        Language key for getting label text.

    Returns
    -------
    bool
        True if values appear to be binary Yes/No.
    """
    if not values or len(values) != 2:
        return False

    labels = [_normalize_label(v.get_text(text_key)) for v in values]

    # Check each label against yes/no patterns
    has_yes = False
    has_no = False

    for label in labels:
        for pattern in BINARY_YES_PATTERNS:
            if re.match(pattern, label, re.IGNORECASE):
                has_yes = True
                break
        for pattern in BINARY_NO_PATTERNS:
            if re.match(pattern, label, re.IGNORECASE):
                has_no = True
                break

    return has_yes and has_no


def _is_awareness_values(values: List[Any], text_key: str = "en-GB") -> bool:
    """
    Check if value labels indicate awareness question.

    Parameters
    ----------
    values : list
        List of ValueLabel objects.
    text_key : str
        Language key for getting label text.

    Returns
    -------
    bool
        True if values appear to be awareness-type responses.
    """
    if not values:
        return False

    # First check if it's binary (most awareness questions are)
    if _is_binary_values(values, text_key):
        return True

    # Check for awareness-specific labels
    labels = [_normalize_label(v.get_text(text_key)) for v in values]

    for label in labels:
        for pattern in AWARENESS_POSITIVE_PATTERNS:
            if re.search(pattern, label, re.IGNORECASE):
                return True
        for pattern in AWARENESS_NEGATIVE_PATTERNS:
            if re.search(pattern, label, re.IGNORECASE):
                return True

    return False


def _is_awareness_question_label(label: str) -> bool:
    """
    Check if a question label indicates an awareness question.

    Parameters
    ----------
    label : str
        The question/variable label.

    Returns
    -------
    bool
        True if label suggests awareness question.
    """
    if not label:
        return False

    label_lower = _normalize_label(label)

    for pattern in AWARENESS_QUESTION_PATTERNS:
        if re.search(pattern, label_lower, re.IGNORECASE):
            return True

    return False


def _detect_value_label_type(values: List[Any], text_key: str = "en-GB") -> Tuple[str, float]:
    """
    Analyze value labels to determine question type.

    Parameters
    ----------
    values : list
        List of ValueLabel objects.
    text_key : str
        Language key.

    Returns
    -------
    tuple
        (detected_type, confidence) where type is one of:
        'binary', 'awareness', 'scale', 'categorical', 'unknown'
    """
    if not values:
        return ('unknown', 0.0)

    n_values = len(values)

    # Check for binary Yes/No
    if _is_binary_values(values, text_key):
        return ('binary', 0.95)

    # Check for awareness patterns
    if _is_awareness_values(values, text_key):
        return ('awareness', 0.90)

    # Check for scale patterns (consecutive numbers)
    codes = sorted([v.value for v in values])
    if n_values >= 3:
        expected_range = list(range(codes[0], codes[-1] + 1))
        if codes == expected_range:
            # Common scale lengths
            if n_values == 11 and codes[0] == 0:  # NPS 0-10
                return ('nps', 0.95)
            elif n_values in [4, 5, 6, 7, 10]:  # Common scales
                return ('scale', 0.85)

    # Otherwise categorical
    return ('categorical', 0.70)


class PatternDetector:
    """
    Detects patterns in survey data to identify question types and groupings.
    """

    def __init__(self, dataset: "DataSet"):
        self.dataset = dataset
        self.data = dataset.data
        self.meta = dataset.meta
        self.text_key = dataset.text_key
        self._detected_groups: List[DetectedGroup] = []
        self._detected_variables: Dict[str, DetectedVariable] = {}
        self._grouped_vars: Set[str] = set()

    def detect_all(self) -> ProcessingSpec:
        """
        Run all detection algorithms and return a complete ProcessingSpec.
        """
        logger.info(f"Starting pattern detection for dataset: {self.dataset.name}")

        # Step 1: Detect variable groups (grids, TOM, etc.)
        self._detect_groups()

        # Step 2: Detect individual variable types
        self._detect_individual_variables()

        # Step 3: Detect banner candidates
        banners = self._detect_banners()

        # Step 4: Detect weight variable
        weight = self._detect_weight()

        # Step 5: Generate suggested analyses
        suggested = self._generate_suggested_analyses(banners)

        # Build the spec
        spec = ProcessingSpec(
            dataset_name=self.dataset.name,
            case_count=self.dataset.n_cases,
            detected_weight=weight,
            banners=banners,
            variable_groups=self._detected_groups,
            individual_variables=list(self._detected_variables.values()),
            suggested_analyses=suggested,
        )

        logger.info(
            f"Detection complete: {len(self._detected_groups)} groups, "
            f"{len(banners)} banners, {len(self._detected_variables)} variables"
        )

        return spec

    def _detect_groups(self):
        """Detect variable groups (grids, top-of-mind, etc.)."""
        variables = list(self.meta.columns.keys())

        # Group by prefix patterns
        prefix_groups: Dict[str, List[Tuple[str, int]]] = {}

        for var_name in variables:
            # Try numeric suffix pattern (Q1_1, Q1_2)
            match = PATTERNS['grid_numeric'].match(var_name)
            if match:
                prefix = match.group(1)
                index = int(match.group(2))
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append((var_name, index))
                continue

            # Try alpha suffix pattern (Q1a, Q1b)
            match = PATTERNS['grid_alpha'].match(var_name)
            if match:
                prefix = match.group(1)
                index = ord(match.group(2).lower()) - ord('a') + 1
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append((var_name, index))

        # Analyze each prefix group
        for prefix, var_list in prefix_groups.items():
            if len(var_list) < 2:
                continue  # Not a group if only 1 variable

            # Sort by index
            var_list.sort(key=lambda x: x[1])
            var_names = [v[0] for v in var_list]

            # Determine group type
            group_type, confidence, calculations = self._classify_group(
                prefix, var_names
            )

            if group_type != QuestionType.UNKNOWN:
                # Get display name from first variable's label
                first_var_meta = self.meta.get_variable(var_names[0])
                display_name = first_var_meta.get_label(self.text_key) if first_var_meta else prefix

                group = DetectedGroup(
                    name=prefix,
                    display_name=display_name,
                    question_type=group_type,
                    variables=var_names,
                    confidence=confidence,
                    suggested_calculations=calculations,
                )
                self._detected_groups.append(group)
                self._grouped_vars.update(var_names)

                logger.debug(f"Detected group '{prefix}': {group_type.value} ({len(var_names)} vars)")

    def _classify_group(
        self, prefix: str, var_names: List[str]
    ) -> Tuple[QuestionType, float, List[str]]:
        """
        Classify a group of variables by type.

        Uses multiple signals:
        1. Variable name patterns (prefix, suffix)
        2. Question label content
        3. Value label analysis (binary, scale, awareness)
        4. Number of value options

        Returns (question_type, confidence, suggested_calculations)
        """
        # Get metadata for analysis
        first_var = var_names[0]
        meta = self.meta.get_variable(first_var)

        if not meta:
            return QuestionType.UNKNOWN, 0.0, []

        # Check if all variables have the same value labels
        all_same_values = True
        first_values = meta.values if meta else None

        for var_name in var_names[1:]:
            var_meta = self.meta.get_variable(var_name)
            if var_meta:
                if var_meta.values != first_values:
                    all_same_values = False
                    break

        # Get label and prefix for pattern matching
        label_lower = (meta.get_label(self.text_key) or "").lower()
        prefix_lower = prefix.lower()

        # === STEP 1: Analyze value labels first (most reliable) ===
        if all_same_values and meta.values:
            value_type, value_confidence = _detect_value_label_type(
                meta.values, self.text_key
            )

            # Binary values (Yes/No) -> likely awareness
            if value_type == 'binary':
                # Check if question label confirms awareness
                if _is_awareness_question_label(meta.get_label(self.text_key)):
                    return (
                        QuestionType.AWARENESS,
                        0.95,
                        ["awareness_rate", "total_aware"]
                    )
                # Even without awareness label, binary grid is often awareness
                return (
                    QuestionType.AWARENESS,
                    0.85,
                    ["awareness_rate", "total_aware"]
                )

            # NPS detected by value labels
            if value_type == 'nps':
                return (
                    QuestionType.NPS,
                    0.95,
                    ["nps_score", "promoters", "passives", "detractors"]
                )

            # Scale detected by value labels
            if value_type == 'scale':
                return (
                    QuestionType.SCALE,
                    0.90,
                    ["mean", "top_2_box", "bottom_2_box"]
                )

        # === STEP 2: Check question label patterns ===

        # Awareness patterns in question label
        if _is_awareness_question_label(meta.get_label(self.text_key)):
            return (
                QuestionType.AWARENESS,
                0.90,
                ["awareness_rate", "total_aware"]
            )

        # Top of Mind / Mention patterns
        is_tom_pattern = any([
            'mention' in label_lower,
            'mencion' in label_lower,
            'mención' in label_lower,
            'first' in label_lower,
            'primera' in label_lower,
            'segundo' in label_lower,
            'tercero' in label_lower,
            'tom' in prefix_lower,
        ])

        if is_tom_pattern:
            return (
                QuestionType.TOP_OF_MIND,
                0.85,
                ["first_mention", "any_mention", "total_mentions"]
            )

        # Ranking patterns
        is_ranking = any([
            'rank' in label_lower,
            'orden' in label_lower,
            'preferencia' in label_lower,
            'posicion' in label_lower,
            'posición' in label_lower,
        ])

        if is_ranking:
            return (
                QuestionType.RANKING,
                0.80,
                ["first_choice", "top_3"]
            )

        # === STEP 3: Fallback to code analysis ===
        if all_same_values and meta.values:
            codes = [v.value for v in meta.values]

            # Check for scale by code pattern
            if self._is_scale(codes):
                if len(codes) == 11 and min(codes) == 0:
                    return (
                        QuestionType.NPS,
                        0.85,
                        ["nps_score", "promoters", "passives", "detractors"]
                    )
                else:
                    return (
                        QuestionType.SCALE,
                        0.80,
                        ["mean", "top_2_box", "bottom_2_box"]
                    )

        # === STEP 4: Default classification ===
        if all_same_values:
            # If only 2-3 values and not detected as awareness, might still be
            if meta.values and len(meta.values) <= 3:
                return (
                    QuestionType.AWARENESS,
                    0.70,
                    ["awareness_rate"]
                )
            return (
                QuestionType.GRID_SINGLE,
                0.70,
                ["frequency", "mean"]
            )
        else:
            return (
                QuestionType.GRID_MULTIPLE,
                0.65,
                ["any_selected", "all_selected"]
            )

    def _is_scale(self, codes: List[int]) -> bool:
        """Check if codes represent a rating scale."""
        if len(codes) < 3:
            return False

        codes_sorted = sorted(codes)

        # Check for common scale ranges
        scale_ranges = [
            (1, 5),   # 1-5 scale
            (1, 7),   # 1-7 scale
            (1, 10),  # 1-10 scale
            (0, 10),  # NPS scale
            (1, 4),   # 1-4 scale
            (1, 6),   # 1-6 scale
        ]

        for start, end in scale_ranges:
            expected = list(range(start, end + 1))
            if codes_sorted == expected:
                return True

        # Check for consecutive integers
        if codes_sorted == list(range(codes_sorted[0], codes_sorted[-1] + 1)):
            return True

        return False

    def _detect_individual_variables(self):
        """Detect types for individual (non-grouped) variables."""
        for var_name, var_meta in self.meta.columns.items():
            if var_name in self._grouped_vars:
                continue  # Skip grouped variables

            question_type, confidence, calculations = self._classify_variable(
                var_name, var_meta
            )

            detected = DetectedVariable(
                name=var_name,
                label=var_meta.get_label(self.text_key),
                question_type=question_type,
                confidence=confidence,
                suggested_calculations=calculations,
            )
            self._detected_variables[var_name] = detected

    def _classify_variable(
        self, var_name: str, var_meta: "VariableMeta"
    ) -> Tuple[QuestionType, float, List[str]]:
        """
        Classify a single variable by type.

        Uses multiple signals:
        1. Variable name patterns (id, weight, demographics)
        2. Question label content
        3. Value label analysis (binary, scale, awareness)
        4. Variable type from metadata
        """
        label = var_meta.get_label(self.text_key) or ""
        label_lower = label.lower()
        name_lower = var_name.lower()

        # === PRIORITY 1: System variables (ID, Weight) ===
        if PATTERNS['id'].search(var_name) or PATTERNS['id'].search(label):
            return QuestionType.ID, 0.95, []

        if PATTERNS['weight'].search(var_name) or PATTERNS['weight'].search(label):
            return QuestionType.WEIGHT, 0.95, []

        # === PRIORITY 2: Demographics ===
        if PATTERNS['gender'].search(var_name) or PATTERNS['gender'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.95, []

        if PATTERNS['age_group'].search(var_name) or PATTERNS['age_group'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.90, []

        if PATTERNS['age'].search(var_name) or PATTERNS['age'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.85, ["mean", "median"]

        if PATTERNS['region'].search(var_name) or PATTERNS['region'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.85, []

        if PATTERNS['income'].search(var_name) or PATTERNS['income'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.85, []

        if PATTERNS['education'].search(var_name) or PATTERNS['education'].search(label):
            return QuestionType.DEMOGRAPHIC, 0.85, []

        # === PRIORITY 3: Special question types by name/label ===
        if PATTERNS['nps'].search(var_name) or PATTERNS['nps'].search(label):
            return QuestionType.NPS, 0.90, ["nps_score", "promoters", "passives", "detractors"]

        if PATTERNS['satisfaction'].search(var_name) or PATTERNS['satisfaction'].search(label):
            if var_meta.values:
                return QuestionType.SCALE, 0.85, ["mean", "top_2_box", "bottom_2_box"]

        # === PRIORITY 4: Analyze value labels (most reliable for classification) ===
        if var_meta.values:
            value_type, value_confidence = _detect_value_label_type(
                var_meta.values, self.text_key
            )

            # Binary Yes/No -> check if awareness
            if value_type == 'binary':
                if _is_awareness_question_label(label):
                    return QuestionType.AWARENESS, 0.95, ["awareness_rate"]
                # Binary without awareness label - still likely awareness or filter
                if PATTERNS['awareness'].search(var_name) or PATTERNS['awareness'].search(label):
                    return QuestionType.AWARENESS, 0.90, ["awareness_rate"]
                # Generic binary
                return QuestionType.SINGLE, 0.80, []

            # NPS by value labels
            if value_type == 'nps':
                return QuestionType.NPS, 0.95, ["nps_score", "promoters", "passives", "detractors"]

            # Scale by value labels
            if value_type == 'scale':
                return QuestionType.SCALE, 0.85, ["mean", "top_2_box", "bottom_2_box"]

        # === PRIORITY 5: Check by question label for awareness ===
        if _is_awareness_question_label(label):
            return QuestionType.AWARENESS, 0.85, ["awareness_rate"]

        if PATTERNS['awareness'].search(var_name) or PATTERNS['awareness'].search(label):
            return QuestionType.AWARENESS, 0.80, ["awareness_rate"]

        # === PRIORITY 6: Check by variable type from metadata ===
        from quantipymrx.core.metadata import VariableType

        if var_meta.type == VariableType.STRING:
            return QuestionType.OPEN_END, 0.85, []

        if var_meta.type == VariableType.DATE:
            return QuestionType.DATE, 0.95, []

        if var_meta.type in [VariableType.INT, VariableType.FLOAT]:
            if not var_meta.values:
                return QuestionType.NUMERIC, 0.80, ["mean", "median", "std"]

        if var_meta.type == VariableType.DELIMITED_SET:
            return QuestionType.MULTIPLE, 0.90, ["any_selected", "total_selected"]

        # === PRIORITY 7: Fallback - analyze codes ===
        if var_meta.values:
            codes = [v.value for v in var_meta.values]
            if self._is_scale(codes):
                if len(codes) == 11 and min(codes) == 0:
                    return QuestionType.NPS, 0.75, ["nps_score"]
                else:
                    return QuestionType.SCALE, 0.75, ["mean", "top_2_box"]
            else:
                return QuestionType.SINGLE, 0.75, []

        return QuestionType.UNKNOWN, 0.5, []

    def _detect_banners(self) -> List[DetectedBanner]:
        """Detect banner candidates (demographics)."""
        banners = []

        for var_name, detected in self._detected_variables.items():
            if detected.question_type != QuestionType.DEMOGRAPHIC:
                continue

            var_meta = self.meta.get_variable(var_name)
            if not var_meta or not var_meta.values:
                continue

            # Determine banner type
            banner_type = self._get_banner_type(var_name, var_meta)

            # A numeric age (one column with 18, 19, 20…) is never a banner:
            # only an age in ranges is. Its value labels are usually just the
            # non-answers (99 = "No responde"), which is why it got here.
            if banner_type == BannerType.AGE and self._looks_continuous(var_name):
                continue

            # Get categories
            categories = [v.get_text(self.text_key) for v in var_meta.values]

            banner = DetectedBanner(
                variable=var_name,
                label=var_meta.get_label(self.text_key),
                banner_type=banner_type,
                confidence=detected.confidence,
                n_categories=len(var_meta.values),
                categories=categories,
            )
            banners.append(banner)

        # Second pass (2026-11-10): profile questions the classifier does not
        # see as DEMOGRAPHIC because `PATTERNS` only match a whole word when
        # it is joined by `-`/`_` ("SEXO", "edad_grupo"), never inside a label
        # with spaces ("¿En cuál ciudad vive?"), and without accents.
        # `PATTERNS` and `_classify_variable` are NOT touched: they also decide
        # every question's type (NPS, scale…), so this pass only adds banners.
        from quantipymrx.core.metadata import VariableType
        already = {b.variable for b in banners}
        # Only kinds the first pass did not find: a second «gender» question (a
        # gender-identity item) must not push income or education out of the
        # banners a file already had.
        covered = {b.banner_type for b in banners}
        for var_name, detected in self._detected_variables.items():
            if var_name in already or detected.question_type in (
                QuestionType.ID, QuestionType.WEIGHT, QuestionType.DEMOGRAPHIC,
            ):
                continue
            var_meta = self.meta.get_variable(var_name)
            if not var_meta or not var_meta.values or len(var_meta.values) < 2:
                continue
            if var_meta.type not in (VariableType.SINGLE, VariableType.INT, VariableType.FLOAT):
                continue
            categories = [v.get_text(self.text_key) for v in var_meta.values]
            banner_type = _profile_banner_type(
                var_name, var_meta.get_label(self.text_key), categories,
            )
            if banner_type is None or banner_type in covered:
                continue
            if banner_type == BannerType.AGE and self._looks_continuous(var_name):
                continue
            banners.append(DetectedBanner(
                variable=var_name,
                label=var_meta.get_label(self.text_key),
                banner_type=banner_type,
                confidence=0.75,
                n_categories=len(var_meta.values),
                categories=categories,
            ))
            already.add(var_name)
            covered.add(banner_type)

        # Sort by confidence and type priority
        type_priority = {
            BannerType.GENDER: 1,
            BannerType.AGE_GROUP: 2,
            BannerType.AGE: 3,
            BannerType.REGION: 4,
            BannerType.INCOME: 5,
            BannerType.EDUCATION: 6,
            BannerType.CUSTOM: 10,
        }

        banners.sort(key=lambda b: (type_priority.get(b.banner_type, 10), -b.confidence))

        return banners

    def _looks_continuous(self, var_name: str) -> bool:
        """A column with many distinct values (an age in years, a score)."""
        if self.data is None or var_name not in self.data.columns:
            return False
        try:
            return int(self.data[var_name].dropna().nunique()) > 15
        except Exception:  # pragma: no cover - defensive
            return False

    def _get_banner_type(self, var_name: str, var_meta: "VariableMeta") -> BannerType:
        """Determine the banner type for a variable."""
        label = var_meta.get_label(self.text_key) or ""

        if PATTERNS['gender'].search(var_name) or PATTERNS['gender'].search(label):
            return BannerType.GENDER

        if PATTERNS['age_group'].search(var_name) or PATTERNS['age_group'].search(label):
            return BannerType.AGE_GROUP

        if PATTERNS['age'].search(var_name) or PATTERNS['age'].search(label):
            return BannerType.AGE

        if PATTERNS['region'].search(var_name) or PATTERNS['region'].search(label):
            return BannerType.REGION

        if PATTERNS['income'].search(var_name) or PATTERNS['income'].search(label):
            return BannerType.INCOME

        if PATTERNS['education'].search(var_name) or PATTERNS['education'].search(label):
            return BannerType.EDUCATION

        return BannerType.CUSTOM

    def _detect_weight(self) -> Optional[str]:
        """Detect weight variable."""
        for var_name, detected in self._detected_variables.items():
            if detected.question_type == QuestionType.WEIGHT:
                return var_name

        # Also check by name pattern in all variables
        for var_name in self.meta.columns.keys():
            if PATTERNS['weight'].search(var_name):
                return var_name

        return None

    def _generate_suggested_analyses(
        self, banners: List[DetectedBanner]
    ) -> List[Dict[str, Any]]:
        """Generate suggested analyses based on detected patterns."""
        suggestions = []

        # Get banner variables for crosstabs
        banner_vars = [b.variable for b in banners[:3]]  # Top 3 banners

        # Suggest analyses for groups
        for group in self._detected_groups:
            if group.question_type == QuestionType.TOP_OF_MIND:
                suggestions.append({
                    "type": "top_of_mind",
                    "name": f"Top of Mind - {group.display_name}",
                    "variables": group.variables,
                    "banners": banner_vars,
                    "calculations": ["first_mention", "any_mention"],
                })

            elif group.question_type == QuestionType.NPS:
                suggestions.append({
                    "type": "nps_analysis",
                    "name": f"NPS - {group.display_name}",
                    "variables": group.variables,
                    "banners": banner_vars,
                    "calculations": ["nps_score", "segment_breakdown"],
                })

            elif group.question_type in [QuestionType.SCALE, QuestionType.GRID_SINGLE]:
                suggestions.append({
                    "type": "satisfaction_grid",
                    "name": f"Satisfaction - {group.display_name}",
                    "variables": group.variables,
                    "banners": banner_vars,
                    "calculations": ["mean", "top_2_box"],
                })

        # Suggest analyses for individual variables
        for var_name, detected in self._detected_variables.items():
            if detected.question_type == QuestionType.NPS:
                suggestions.append({
                    "type": "nps_analysis",
                    "name": f"NPS - {detected.label}",
                    "variables": [var_name],
                    "banners": banner_vars,
                    "calculations": ["nps_score", "segment_breakdown"],
                })

            elif detected.question_type == QuestionType.SCALE:
                suggestions.append({
                    "type": "scale_analysis",
                    "name": f"Scale - {detected.label}",
                    "variables": [var_name],
                    "banners": banner_vars,
                    "calculations": ["mean", "top_2_box", "distribution"],
                })

        return suggestions


# =============================================================================
# Convenience Functions
# =============================================================================

def auto_detect(dataset: "DataSet") -> ProcessingSpec:
    """
    Automatically detect patterns in a dataset.

    This is the main entry point for auto-detection.

    Parameters
    ----------
    dataset : DataSet
        The dataset to analyze.

    Returns
    -------
    ProcessingSpec
        Complete processing specification with detected patterns.

    Example
    -------
    >>> from quantipymrx import DataSet
    >>> from quantipymrx.analysis.auto_detect import auto_detect
    >>>
    >>> ds = DataSet.from_spss("survey.sav", text_key="es-ES")
    >>> spec = auto_detect(ds)
    >>>
    >>> print(f"Detected {len(spec.banners)} banners")
    >>> print(f"Detected {len(spec.variable_groups)} variable groups")
    >>>
    >>> # Export to JSON for frontend
    >>> import json
    >>> json_spec = json.dumps(spec.to_dict(), indent=2)
    """
    detector = PatternDetector(dataset)
    return detector.detect_all()


def suggest_banners(dataset: "DataSet", max_banners: int = 5) -> List[DetectedBanner]:
    """
    Suggest banner variables for cross-tabulation.

    Parameters
    ----------
    dataset : DataSet
        The dataset to analyze.
    max_banners : int, default 5
        Maximum number of banners to suggest.

    Returns
    -------
    list of DetectedBanner
        Suggested banner variables.
    """
    detector = PatternDetector(dataset)
    detector._detect_individual_variables()
    banners = detector._detect_banners()
    return banners[:max_banners]


def detect_groups(dataset: "DataSet") -> List[DetectedGroup]:
    """
    Detect variable groups (grids, top-of-mind, etc.).

    Parameters
    ----------
    dataset : DataSet
        The dataset to analyze.

    Returns
    -------
    list of DetectedGroup
        Detected variable groups.
    """
    detector = PatternDetector(dataset)
    detector._detect_groups()
    return detector._detected_groups


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "QuestionType",
    "BannerType",
    # Data classes
    "DetectedVariable",
    "DetectedGroup",
    "DetectedBanner",
    "ProcessingSpec",
    # Main class
    "PatternDetector",
    # Functions
    "auto_detect",
    "suggest_banners",
    "detect_groups",
]
