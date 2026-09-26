"""Pairwise matching features. Uses real TF-IDF spaces plus fuzzy string features."""

from __future__ import annotations
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import add_normalized_columns

FEATURE_COLUMNS = [
    "name_exact", "name_jaccard", "name_token_sort_ratio", "name_partial_ratio",
    "name_levenshtein_similarity", "name_jaro", "name_char_similarity",
    "name_token_overlap", "name_length_difference", "name_length_ratio",
    "address_exact", "address_jaccard", "address_token_sort_ratio", "address_partial_ratio",
    "address_levenshtein_similarity", "address_jaro", "address_char_similarity",
    "address_token_overlap", "address_length_difference", "address_length_ratio",
    "country_match", "house_number_match", "unit_number_match", "postal_code_match",
    "numeric_token_overlap", "name_tfidf_cosine", "address_tfidf_cosine",
    "combined_tfidf_cosine", "blocking_similarity",
]

def _tokens(s): return [x for x in str(s).split() if x]
def _jaccard(a, b):
    A, B = set(_tokens(a)), set(_tokens(b))
    return len(A & B) / len(A | B) if A | B else 0.0
def _overlap(a, b):
    A, B = set(_tokens(a)), set(_tokens(b))
    return len(A & B) / min(len(A), len(B)) if A and B else 0.0
def _seq(a, b): return SequenceMatcher(None, str(a), str(b)).ratio()
def _len_ratio(a, b):
    m = max(len(str(a)), len(str(b)))
    return min(len(str(a)), len(str(b))) / m if m else 1.0
def _numeric_overlap(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 0.0

def _tfidf_pair(a, b):
    if not a or not b: return 0.0
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), sublinear_tf=True)
    X = vec.fit_transform([a,b])
    return float(cosine_similarity(X[0], X[1])[0,0])

def _pair(left, right, blocking_similarity=0.0):
    nl, nr = left.name_normalized, right.name_normalized
    al, ar = left.address_normalized, right.address_normalized
    nsorted_l, nsorted_r = " ".join(sorted(_tokens(nl))), " ".join(sorted(_tokens(nr)))
    asorted_l, asorted_r = " ".join(sorted(_tokens(al))), " ".join(sorted(_tokens(ar)))
    nchar = fuzz.ratio(nl, nr) / 100.0
    achar = fuzz.ratio(al, ar) / 100.0
    return {
        "source1_entity_id": str(left.entity_id),
        "candidate_entity_id": str(right.entity_id),
        "name_exact": float(bool(nl) and nl == nr),
        "name_jaccard": _jaccard(nl, nr),
        "name_token_sort_ratio": fuzz.ratio(nsorted_l, nsorted_r) / 100.0 if nsorted_l and nsorted_r else 0.0,
        "name_partial_ratio": fuzz.partial_ratio(nl, nr) / 100.0 if nl and nr else 0.0,
        "name_levenshtein_similarity": fuzz.ratio(nl, nr) / 100.0,
        "name_jaro": fuzz.WRatio(nl, nr) / 100.0 if nl and nr else 0.0,
        "name_char_similarity": nchar,
        "name_token_overlap": _overlap(nl, nr),
        "name_length_difference": abs(len(nl)-len(nr)),
        "name_length_ratio": _len_ratio(nl,nr),
        "address_exact": float(bool(al) and al == ar),
        "address_jaccard": _jaccard(al, ar),
        "address_token_sort_ratio": fuzz.ratio(asorted_l, asorted_r) / 100.0 if asorted_l and asorted_r else 0.0,
        "address_partial_ratio": fuzz.partial_ratio(al, ar) / 100.0 if al and ar else 0.0,
        "address_levenshtein_similarity": fuzz.ratio(al, ar) / 100.0,
        "address_jaro": fuzz.WRatio(al, ar) / 100.0 if al and ar else 0.0,
        "address_char_similarity": achar,
        "address_token_overlap": _overlap(al, ar),
        "address_length_difference": abs(len(al)-len(ar)),
        "address_length_ratio": _len_ratio(al,ar),
        "country_match": float(bool(left.country_normalized) and left.country_normalized == right.country_normalized),
        "house_number_match": float(bool(left.house_number) and left.house_number == right.house_number),
        "unit_number_match": float(bool(left.unit_number) and left.unit_number == right.unit_number),
        "postal_code_match": float(bool(left.postal_code) and left.postal_code == right.postal_code),
        "numeric_token_overlap": _numeric_overlap(left.address_numeric_tokens, right.address_numeric_tokens),
        "name_tfidf_cosine": _tfidf_pair(nl,nr),
        "address_tfidf_cosine": _tfidf_pair(al,ar),
        "combined_tfidf_cosine": _tfidf_pair(f"{nl} {al}".strip(), f"{nr} {ar}".strip()),
        "blocking_similarity": float(blocking_similarity),
    }

def build_pair_features(source1, reference, candidate_pairs):
    if candidate_pairs.empty:
        return pd.DataFrame(columns=["source1_entity_id","candidate_entity_id",*FEATURE_COLUMNS])
    left = add_normalized_columns(source1)
    right = add_normalized_columns(reference)
    lm = {str(r.entity_id): r for r in left.itertuples(index=False)}
    rm = {str(r.entity_id): r for r in right.itertuples(index=False)}
    rows = []
    for p in candidate_pairs.itertuples(index=False):
        l, r = lm.get(str(p.source1_entity_id)), rm.get(str(p.candidate_entity_id))
        if l is not None and r is not None:
            rows.append(_pair(l, r, getattr(p, "blocking_similarity", 0.0)))
    return pd.DataFrame(rows, columns=["source1_entity_id","candidate_entity_id",*FEATURE_COLUMNS])
