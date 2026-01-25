# -*- coding: utf-8 -*-
"""
Internationalization and Encoding Support Module

Provides comprehensive support for multiple languages and character encodings
commonly used in market research data from different regions.

This module is a key differentiator for talk2data, enabling seamless
handling of survey data in Spanish, Portuguese, German, French, Chinese,
Japanese, Arabic, and many other languages.

Supported Languages and Text Keys:
- es-ES, es-MX, es-AR: Spanish (Spain, Mexico, Argentina)
- pt-BR, pt-PT: Portuguese (Brazil, Portugal)
- de-DE, de-AT, de-CH: German (Germany, Austria, Switzerland)
- fr-FR, fr-CA: French (France, Canada)
- it-IT: Italian
- nl-NL, nl-BE: Dutch (Netherlands, Belgium)
- pl-PL: Polish
- ru-RU: Russian
- zh-CN, zh-TW: Chinese (Simplified, Traditional)
- ja-JP: Japanese
- ko-KR: Korean
- ar-SA, ar-AE: Arabic (Saudi Arabia, UAE)
- hi-IN: Hindi
- th-TH: Thai
- vi-VN: Vietnamese
- id-ID: Indonesian
- ms-MY: Malay
- tr-TR: Turkish
- he-IL: Hebrew
- en-GB, en-US, en-AU: English (UK, US, Australia)
- da-DK: Danish
- sv-SE: Swedish
- nb-NO, nn-NO: Norwegian
- fi-FI: Finnish
- is-IS: Icelandic

Usage:
    from quantipymrx._compat.i18n import (
        fix_encoding,
        detect_encoding,
        normalize_text,
        get_text_key_info,
        SUPPORTED_TEXT_KEYS
    )

    # Fix encoding issues
    fixed_text = fix_encoding("Menor de 24 a\\xf1os")
    # Result: "Menor de 24 anos"

    # Detect encoding
    encoding = detect_encoding(b"Some bytes")

    # Normalize text for consistent comparisons
    normalized = normalize_text("  Cafe  ")
    # Result: "cafe"
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# Try to import chardet for encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

# Try to import ftfy for text fixing
try:
    from ftfy import fix_text as ftfy_fix_text
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False


# =============================================================================
# Supported Text Keys and Language Configuration
# =============================================================================

SUPPORTED_TEXT_KEYS: Dict[str, Dict[str, Any]] = {
    # English
    'en-GB': {'name': 'English (UK)', 'encoding': 'utf-8', 'region': 'Europe'},
    'en-US': {'name': 'English (US)', 'encoding': 'utf-8', 'region': 'North America'},
    'en-AU': {'name': 'English (Australia)', 'encoding': 'utf-8', 'region': 'Oceania'},
    'en': {'name': 'English', 'encoding': 'utf-8', 'region': 'Global'},

    # Spanish
    'es-ES': {'name': 'Spanish (Spain)', 'encoding': 'utf-8', 'region': 'Europe'},
    'es-MX': {'name': 'Spanish (Mexico)', 'encoding': 'utf-8', 'region': 'North America'},
    'es-AR': {'name': 'Spanish (Argentina)', 'encoding': 'utf-8', 'region': 'South America'},
    'es-CO': {'name': 'Spanish (Colombia)', 'encoding': 'utf-8', 'region': 'South America'},
    'es-CL': {'name': 'Spanish (Chile)', 'encoding': 'utf-8', 'region': 'South America'},
    'es-PE': {'name': 'Spanish (Peru)', 'encoding': 'utf-8', 'region': 'South America'},

    # Portuguese
    'pt-BR': {'name': 'Portuguese (Brazil)', 'encoding': 'utf-8', 'region': 'South America'},
    'pt-PT': {'name': 'Portuguese (Portugal)', 'encoding': 'utf-8', 'region': 'Europe'},

    # German
    'de-DE': {'name': 'German (Germany)', 'encoding': 'utf-8', 'region': 'Europe'},
    'de-AT': {'name': 'German (Austria)', 'encoding': 'utf-8', 'region': 'Europe'},
    'de-CH': {'name': 'German (Switzerland)', 'encoding': 'utf-8', 'region': 'Europe'},

    # French
    'fr-FR': {'name': 'French (France)', 'encoding': 'utf-8', 'region': 'Europe'},
    'fr-CA': {'name': 'French (Canada)', 'encoding': 'utf-8', 'region': 'North America'},
    'fr-BE': {'name': 'French (Belgium)', 'encoding': 'utf-8', 'region': 'Europe'},
    'fr-CH': {'name': 'French (Switzerland)', 'encoding': 'utf-8', 'region': 'Europe'},

    # Italian
    'it-IT': {'name': 'Italian', 'encoding': 'utf-8', 'region': 'Europe'},

    # Dutch
    'nl-NL': {'name': 'Dutch (Netherlands)', 'encoding': 'utf-8', 'region': 'Europe'},
    'nl-BE': {'name': 'Dutch (Belgium)', 'encoding': 'utf-8', 'region': 'Europe'},

    # Polish
    'pl-PL': {'name': 'Polish', 'encoding': 'utf-8', 'region': 'Europe'},

    # Russian
    'ru-RU': {'name': 'Russian', 'encoding': 'utf-8', 'region': 'Europe/Asia'},

    # Chinese
    'zh-CN': {'name': 'Chinese (Simplified)', 'encoding': 'utf-8', 'region': 'Asia'},
    'zh-TW': {'name': 'Chinese (Traditional)', 'encoding': 'utf-8', 'region': 'Asia'},
    'zh-HK': {'name': 'Chinese (Hong Kong)', 'encoding': 'utf-8', 'region': 'Asia'},

    # Japanese
    'ja-JP': {'name': 'Japanese', 'encoding': 'utf-8', 'region': 'Asia'},

    # Korean
    'ko-KR': {'name': 'Korean', 'encoding': 'utf-8', 'region': 'Asia'},

    # Arabic
    'ar-SA': {'name': 'Arabic (Saudi Arabia)', 'encoding': 'utf-8', 'region': 'Middle East'},
    'ar-AE': {'name': 'Arabic (UAE)', 'encoding': 'utf-8', 'region': 'Middle East'},
    'ar-EG': {'name': 'Arabic (Egypt)', 'encoding': 'utf-8', 'region': 'Middle East'},

    # Hindi
    'hi-IN': {'name': 'Hindi', 'encoding': 'utf-8', 'region': 'Asia'},

    # Thai
    'th-TH': {'name': 'Thai', 'encoding': 'utf-8', 'region': 'Asia'},

    # Vietnamese
    'vi-VN': {'name': 'Vietnamese', 'encoding': 'utf-8', 'region': 'Asia'},

    # Indonesian/Malay
    'id-ID': {'name': 'Indonesian', 'encoding': 'utf-8', 'region': 'Asia'},
    'ms-MY': {'name': 'Malay', 'encoding': 'utf-8', 'region': 'Asia'},

    # Turkish
    'tr-TR': {'name': 'Turkish', 'encoding': 'utf-8', 'region': 'Europe/Asia'},

    # Hebrew
    'he-IL': {'name': 'Hebrew', 'encoding': 'utf-8', 'region': 'Middle East'},

    # Scandinavian
    'da-DK': {'name': 'Danish', 'encoding': 'utf-8', 'region': 'Europe'},
    'sv-SE': {'name': 'Swedish', 'encoding': 'utf-8', 'region': 'Europe'},
    'nb-NO': {'name': 'Norwegian (Bokmal)', 'encoding': 'utf-8', 'region': 'Europe'},
    'nn-NO': {'name': 'Norwegian (Nynorsk)', 'encoding': 'utf-8', 'region': 'Europe'},
    'fi-FI': {'name': 'Finnish', 'encoding': 'utf-8', 'region': 'Europe'},
    'is-IS': {'name': 'Icelandic', 'encoding': 'utf-8', 'region': 'Europe'},

    # Other European
    'el-GR': {'name': 'Greek', 'encoding': 'utf-8', 'region': 'Europe'},
    'cs-CZ': {'name': 'Czech', 'encoding': 'utf-8', 'region': 'Europe'},
    'sk-SK': {'name': 'Slovak', 'encoding': 'utf-8', 'region': 'Europe'},
    'hu-HU': {'name': 'Hungarian', 'encoding': 'utf-8', 'region': 'Europe'},
    'ro-RO': {'name': 'Romanian', 'encoding': 'utf-8', 'region': 'Europe'},
    'bg-BG': {'name': 'Bulgarian', 'encoding': 'utf-8', 'region': 'Europe'},
    'uk-UA': {'name': 'Ukrainian', 'encoding': 'utf-8', 'region': 'Europe'},
    'hr-HR': {'name': 'Croatian', 'encoding': 'utf-8', 'region': 'Europe'},
    'sr-RS': {'name': 'Serbian', 'encoding': 'utf-8', 'region': 'Europe'},
    'sl-SI': {'name': 'Slovenian', 'encoding': 'utf-8', 'region': 'Europe'},
}

# Common encoding aliases
ENCODING_ALIASES: Dict[str, str] = {
    'latin1': 'latin-1',
    'latin-1': 'iso-8859-1',
    'iso8859-1': 'iso-8859-1',
    'cp1252': 'windows-1252',
    'windows1252': 'windows-1252',
    'utf8': 'utf-8',
    'utf-8-sig': 'utf-8',
}

# Common mojibake patterns - using Unicode escapes to avoid encoding issues
# These patterns represent UTF-8 text incorrectly decoded as Latin-1
MOJIBAKE_PATTERNS: List[Tuple[str, str]] = [
    # Spanish accented vowels (UTF-8 as Latin-1)
    ('\xc3\xa1', '\xe1'),  # a with acute -> a
    ('\xc3\xa9', '\xe9'),  # e with acute -> e
    ('\xc3\xad', '\xed'),  # i with acute -> i
    ('\xc3\xb3', '\xf3'),  # o with acute -> o
    ('\xc3\xba', '\xfa'),  # u with acute -> u
    ('\xc3\xb1', '\xf1'),  # n with tilde -> n

    # Spanish uppercase
    ('\xc3\x81', '\xc1'),  # A with acute
    ('\xc3\x89', '\xc9'),  # E with acute
    ('\xc3\x8d', '\xcd'),  # I with acute
    ('\xc3\x93', '\xd3'),  # O with acute
    ('\xc3\x9a', '\xda'),  # U with acute
    ('\xc3\x91', '\xd1'),  # N with tilde

    # Spanish punctuation
    ('\xc2\xbf', '\xbf'),  # inverted question mark
    ('\xc2\xa1', '\xa1'),  # inverted exclamation

    # Portuguese
    ('\xc3\xa3', '\xe3'),  # a with tilde
    ('\xc3\xb5', '\xf5'),  # o with tilde
    ('\xc3\xa7', '\xe7'),  # c cedilla
    ('\xc3\xaa', '\xea'),  # e circumflex
    ('\xc3\xb4', '\xf4'),  # o circumflex
    ('\xc3\xa2', '\xe2'),  # a circumflex
    ('\xc3\x80', '\xc0'),  # A grave

    # German
    ('\xc3\xa4', '\xe4'),  # a umlaut
    ('\xc3\xb6', '\xf6'),  # o umlaut
    ('\xc3\xbc', '\xfc'),  # u umlaut
    ('\xc3\x9f', '\xdf'),  # eszett
    ('\xc3\x84', '\xc4'),  # A umlaut
    ('\xc3\x96', '\xd6'),  # O umlaut
    ('\xc3\x9c', '\xdc'),  # U umlaut

    # French
    ('\xc3\xa8', '\xe8'),  # e grave
    ('\xc3\xb9', '\xf9'),  # u grave
    ('\xc3\xae', '\xee'),  # i circumflex
    ('\xc3\xaf', '\xef'),  # i diaeresis
    ('\xc3\xab', '\xeb'),  # e diaeresis
    ('\xc5\x93', '\x9c'),  # oe ligature (windows-1252)

    # Common symbols
    ('\xe2\x80\x94', '\x97'),  # em dash
    ('\xe2\x80\x93', '\x96'),  # en dash
    ('\xe2\x80\x99', "'"),      # right single quote -> apostrophe
    ('\xe2\x80\x9c', '"'),      # left double quote
    ('\xe2\x80\x9d', '"'),      # right double quote
    ('\xe2\x80\xa6', '...'),    # ellipsis
    ('\xe2\x80\xa2', '\xb7'),   # bullet
    ('\xc2\xb0', '\xb0'),       # degree sign
    ('\xc2\xa9', '\xa9'),       # copyright
    ('\xc2\xae', '\xae'),       # registered
    ('\xe2\x82\xac', '\x80'),   # euro (windows-1252)
    ('\xc2\xa3', '\xa3'),       # pound
    ('\xc2\xa5', '\xa5'),       # yen

    # Replacement character cleanup
    ('\xef\xbf\xbd', ''),       # Unicode replacement character as UTF-8 bytes
    ('\ufffd', ''),             # Unicode replacement character
]


# =============================================================================
# Encoding Detection and Fixing
# =============================================================================

def detect_encoding(data: bytes) -> str:
    """
    Detect the encoding of byte data.

    Uses chardet if available, otherwise falls back to common encodings.

    Parameters
    ----------
    data : bytes
        The byte data to analyze

    Returns
    -------
    str
        The detected encoding name
    """
    if HAS_CHARDET:
        result = chardet.detect(data)
        encoding = result.get('encoding', 'utf-8')
        confidence = result.get('confidence', 0)
        if confidence < 0.7:
            # Low confidence, try common encodings
            return _try_common_encodings(data) or encoding
        return encoding or 'utf-8'
    else:
        return _try_common_encodings(data) or 'utf-8'


def _try_common_encodings(data: bytes) -> Optional[str]:
    """Try to decode with common encodings."""
    common_encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
    for encoding in common_encodings:
        try:
            data.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def fix_encoding(
    text: str,
    source_encoding: Optional[str] = None,
    use_ftfy: bool = True
) -> str:
    """
    Fix encoding issues in text (mojibake).

    This function handles common encoding problems found in survey data,
    particularly when data has been incorrectly decoded or has gone through
    multiple encoding conversions.

    Parameters
    ----------
    text : str
        The text to fix
    source_encoding : str, optional
        If known, the original encoding of the text
    use_ftfy : bool, default True
        Whether to use the ftfy library if available

    Returns
    -------
    str
        The fixed text

    Examples
    --------
    >>> fix_encoding("Menor de 24 anos")
    'Menor de 24 anos'
    """
    if not text or not isinstance(text, str):
        return text

    # Try ftfy first if available - it's the most robust solution
    if use_ftfy and HAS_FTFY:
        try:
            fixed = ftfy_fix_text(text)
            if fixed != text:
                return fixed
        except Exception:
            pass

    # Apply mojibake patterns
    result = text
    for pattern, replacement in MOJIBAKE_PATTERNS:
        result = result.replace(pattern, replacement)

    # Try to fix double-encoded UTF-8 (common when UTF-8 is read as Latin-1)
    try:
        # Detect if this looks like UTF-8 read as Latin-1
        decoded = result.encode('latin-1').decode('utf-8')
        if decoded != result:
            return decoded
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Try the reverse: Latin-1 characters stored as UTF-8 codes
    try:
        # Check for common latin-1 encoded as UTF-8 pattern
        if '\xc3' in result or '\xc2' in result:
            decoded = result.encode('latin-1').decode('utf-8')
            return decoded
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return result


def fix_dataframe_encoding(
    df,
    columns: Optional[List[str]] = None,
    inplace: bool = False
):
    """
    Fix encoding issues in all string columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to fix
    columns : list of str, optional
        Specific columns to fix. If None, all object columns are processed.
    inplace : bool, default False
        Whether to modify the DataFrame in place

    Returns
    -------
    pandas.DataFrame
        DataFrame with fixed encoding
    """
    import pandas as pd

    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: fix_encoding(x) if isinstance(x, str) else x
            )

    return df


def fix_meta_encoding(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix encoding issues in metadata dictionary.

    Recursively processes all text values in the metadata structure.

    Parameters
    ----------
    meta : dict
        The metadata dictionary to fix

    Returns
    -------
    dict
        Metadata with fixed encoding
    """
    def fix_recursive(obj):
        if isinstance(obj, str):
            return fix_encoding(obj)
        elif isinstance(obj, dict):
            return {k: fix_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [fix_recursive(item) for item in obj]
        return obj

    return fix_recursive(meta)


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_text(
    text: str,
    lowercase: bool = True,
    strip: bool = True,
    remove_accents: bool = False,
    remove_punctuation: bool = False
) -> str:
    """
    Normalize text for consistent comparisons.

    Parameters
    ----------
    text : str
        The text to normalize
    lowercase : bool, default True
        Convert to lowercase
    strip : bool, default True
        Remove leading/trailing whitespace
    remove_accents : bool, default False
        Remove accented characters (e -> e)
    remove_punctuation : bool, default False
        Remove punctuation marks

    Returns
    -------
    str
        Normalized text

    Examples
    --------
    >>> normalize_text("  Cafe  ")
    'cafe'

    >>> normalize_text("Cafe", remove_accents=True)
    'cafe'
    """
    if not text or not isinstance(text, str):
        return text

    result = text

    if strip:
        result = result.strip()

    if lowercase:
        result = result.lower()

    if remove_accents:
        # Decompose unicode and remove combining characters
        result = unicodedata.normalize('NFKD', result)
        result = ''.join(c for c in result if not unicodedata.combining(c))

    if remove_punctuation:
        result = re.sub(r'[^\w\s]', '', result)

    return result


# =============================================================================
# Text Key Utilities
# =============================================================================

def get_text_key_info(text_key: str) -> Dict[str, Any]:
    """
    Get information about a text key.

    Parameters
    ----------
    text_key : str
        The text key (e.g., 'es-ES', 'en-GB')

    Returns
    -------
    dict
        Information about the text key including name, encoding, and region
    """
    # Normalize the text key
    normalized = text_key.strip()

    if normalized in SUPPORTED_TEXT_KEYS:
        return SUPPORTED_TEXT_KEYS[normalized].copy()

    # Try without region suffix
    base_lang = normalized.split('-')[0]
    for key, info in SUPPORTED_TEXT_KEYS.items():
        if key.startswith(base_lang + '-'):
            return info.copy()

    # Unknown text key
    return {
        'name': f'Unknown ({text_key})',
        'encoding': 'utf-8',
        'region': 'Unknown'
    }


def is_rtl_language(text_key: str) -> bool:
    """
    Check if a text key represents a right-to-left language.

    Parameters
    ----------
    text_key : str
        The text key

    Returns
    -------
    bool
        True if the language is RTL
    """
    rtl_prefixes = ['ar', 'he', 'fa', 'ur']
    base_lang = text_key.split('-')[0].lower()
    return base_lang in rtl_prefixes


def get_language_family(text_key: str) -> str:
    """
    Get the language family for a text key.

    Parameters
    ----------
    text_key : str
        The text key

    Returns
    -------
    str
        The language family name
    """
    families = {
        'en': 'Germanic',
        'de': 'Germanic',
        'nl': 'Germanic',
        'sv': 'Germanic',
        'da': 'Germanic',
        'nb': 'Germanic',
        'nn': 'Germanic',
        'is': 'Germanic',
        'es': 'Romance',
        'pt': 'Romance',
        'fr': 'Romance',
        'it': 'Romance',
        'ro': 'Romance',
        'pl': 'Slavic',
        'ru': 'Slavic',
        'uk': 'Slavic',
        'cs': 'Slavic',
        'sk': 'Slavic',
        'bg': 'Slavic',
        'hr': 'Slavic',
        'sr': 'Slavic',
        'sl': 'Slavic',
        'zh': 'Sino-Tibetan',
        'ja': 'Japonic',
        'ko': 'Koreanic',
        'ar': 'Semitic',
        'he': 'Semitic',
        'hi': 'Indo-Aryan',
        'th': 'Tai-Kadai',
        'vi': 'Austroasiatic',
        'id': 'Austronesian',
        'ms': 'Austronesian',
        'tr': 'Turkic',
        'fi': 'Uralic',
        'hu': 'Uralic',
        'el': 'Hellenic',
    }
    base_lang = text_key.split('-')[0].lower()
    return families.get(base_lang, 'Other')


def suggest_text_key(locale: str) -> str:
    """
    Suggest an appropriate text key based on a locale string.

    Parameters
    ----------
    locale : str
        A locale string (e.g., 'es_ES.UTF-8', 'en_US', 'Spanish')

    Returns
    -------
    str
        Suggested text key
    """
    locale_lower = locale.lower().strip()

    # Direct match
    normalized = locale_lower.replace('_', '-').split('.')[0]
    if normalized in [k.lower() for k in SUPPORTED_TEXT_KEYS]:
        for key in SUPPORTED_TEXT_KEYS:
            if key.lower() == normalized:
                return key

    # Language name match
    language_map = {
        'spanish': 'es-ES',
        'portuguese': 'pt-BR',
        'german': 'de-DE',
        'french': 'fr-FR',
        'italian': 'it-IT',
        'english': 'en-GB',
        'chinese': 'zh-CN',
        'japanese': 'ja-JP',
        'korean': 'ko-KR',
        'arabic': 'ar-SA',
        'russian': 'ru-RU',
    }

    for name, key in language_map.items():
        if name in locale_lower:
            return key

    # Default
    return 'en-GB'


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    'SUPPORTED_TEXT_KEYS',
    'ENCODING_ALIASES',
    'MOJIBAKE_PATTERNS',

    # Detection and fixing
    'detect_encoding',
    'fix_encoding',
    'fix_dataframe_encoding',
    'fix_meta_encoding',

    # Normalization
    'normalize_text',

    # Text key utilities
    'get_text_key_info',
    'is_rtl_language',
    'get_language_family',
    'suggest_text_key',

    # Feature flags
    'HAS_CHARDET',
    'HAS_FTFY',
]
