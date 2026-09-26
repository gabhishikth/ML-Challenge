"""Canonical preprocessing shared by blocking, features, training and prediction."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from unidecode import unidecode

_PUNCT_RE = re.compile(r"[^0-9a-zA-Z\s]+")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+[a-z]?")
_POSTAL_RE = re.compile(r"\b\d{4,10}\b", re.I)
_UNIT_RE = re.compile(r"\b(?:apt|apartment|unit|suite|ste|flat|floor|fl)\s*([a-z0-9-]+)", re.I)

_NAME_SUFFIXES = {
    "private limited": " ",
    "pvt ltd": " ",
    "pvt limited": " ",
    "private ltd": " ",
    "limited": " ",
    "ltd": " ",
    "llp": " ",
    "incorporated": " ",
    "inc": " ",
    "corporation": " ",
    "corp": " ",
    "company": " ",
    "co": " ",
}

_ADDRESS_REPLACEMENTS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "apartment": "apt",
    "suite": "ste", "floor": "fl", "opposite": "opp", "near": "near",
}

def _clean_unicode(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Keep original script separately; this ASCII form is the baseline retrieval form.
    return text

def normalize_text(value: Any, remove_punctuation: bool = True) -> str:
    text = _clean_unicode(value).lower()
    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def normalize_name(value: Any) -> str:
    # Transliteration is used only as a normalization signal; no translation/API lookup.
    text = normalize_text(unidecode(value))
    # Conservative: remove legal suffixes only when they occur at the end.
    changed = True
    while changed and text:
        changed = False
        for suffix in sorted(_NAME_SUFFIXES, key=len, reverse=True):
            pattern = rf"(?:\s|^){re.escape(suffix)}$"
            new_text = re.sub(pattern, "", text).strip()
            if new_text != text:
                text = new_text
                changed = True
                break
    return text

def normalize_address(value: Any) -> str:
    text = normalize_text(unidecode(value))
    return " ".join(_ADDRESS_REPLACEMENTS.get(tok, tok) for tok in text.split())

def normalize_country(value: Any) -> str:
    # Open-set: never restrict to a fixed country list.
    return normalize_text(value)

def extract_numeric_tokens(value: Any) -> tuple[str, ...]:
    return tuple(_NUM_RE.findall(normalize_text(value, remove_punctuation=False)))

def extract_postal_code(value: Any) -> str:
    matches = _POSTAL_RE.findall(normalize_text(value, remove_punctuation=False))
    return matches[-1] if matches else ""

def extract_house_number(value: Any) -> str:
    nums = extract_numeric_tokens(value)
    return nums[0] if nums else ""

def extract_unit_number(value: Any) -> str:
    match = _UNIT_RE.search(normalize_text(value, remove_punctuation=False))
    return match.group(1).lower() if match else ""

def add_normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["name_normalized"] = result["business_name"].map(normalize_name)
    result["address_normalized"] = result["business_address"].map(normalize_address)
    result["country_normalized"] = result["country"].map(normalize_country)
    result["name_numeric_tokens"] = result["name_normalized"].map(extract_numeric_tokens)
    result["address_numeric_tokens"] = result["address_normalized"].map(extract_numeric_tokens)
    result["postal_code"] = result["address_normalized"].map(extract_postal_code)
    result["house_number"] = result["address_normalized"].map(extract_house_number)
    result["unit_number"] = result["address_normalized"].map(extract_unit_number)
    return result
