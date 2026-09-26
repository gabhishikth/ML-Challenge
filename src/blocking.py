"""Hybrid candidate generation: exact/token/numeric indexes + character TF-IDF.

Important design:
- Blocking routes are independent and UNIONed; no strict name->address cascade.
- Country is used as a partition when known.
- Empty/unknown country queries are allowed to search all reference records.
- Character TF-IDF protects recall for typos and noisy variants.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .preprocessing import add_normalized_columns

@dataclass(frozen=True)
class BlockingConfig:
    name_token_limit: int = 80
    address_token_limit: int = 80
    numeric_limit: int = 100
    max_candidates_per_entity: int = 300
    tfidf_top_k: int = 150
    tfidf_min_similarity: float = 0.18
    char_ngram_range: tuple[int, int] = (3, 5)
    min_token_length: int = 3

class HybridBlocker:
    def __init__(self, config: BlockingConfig | None = None):
        self.config = config or BlockingConfig()
        self.reference = None
        self._name_exact = defaultdict(list)
        self._address_exact = defaultdict(list)
        self._name_tokens = defaultdict(list)
        self._address_tokens = defaultdict(list)
        self._house = defaultdict(list)
        self._postal = defaultdict(list)
        self._name_vectorizer = None
        self._address_vectorizer = None
        self._name_matrix = None
        self._address_matrix = None

    def fit(self, reference: pd.DataFrame):
        self.reference = add_normalized_columns(reference).reset_index(drop=True)
        for row in self.reference.itertuples(index=False):
            eid = str(row.entity_id)
            country = str(row.country_normalized)
            if row.name_normalized:
                self._name_exact[(country, row.name_normalized)].append(eid)
                for tok in set(row.name_normalized.split()):
                    if len(tok) >= self.config.min_token_length:
                        self._name_tokens[(country, tok)].append(eid)
            if row.address_normalized:
                self._address_exact[(country, row.address_normalized)].append(eid)
                for tok in set(row.address_normalized.split()):
                    if len(tok) >= self.config.min_token_length:
                        self._address_tokens[(country, tok)].append(eid)
            if row.house_number:
                self._house[(country, row.house_number)].append(eid)
            if row.postal_code:
                self._postal[(country, row.postal_code)].append(eid)

        # Real character TF-IDF. Fit once, not once per pair.
        self._name_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=self.config.char_ngram_range,
                                                min_df=1, sublinear_tf=True)
        self._address_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=self.config.char_ngram_range,
                                                   min_df=1, sublinear_tf=True)
        self._name_matrix = self._name_vectorizer.fit_transform(self.reference["name_normalized"].fillna(""))
        self._address_matrix = self._address_vectorizer.fit_transform(self.reference["address_normalized"].fillna(""))
        return self

    def _lookup(self, index, country, key, limit):
        if country:
            vals = index.get((country, key), [])
            if vals:
                return vals[:limit]
        # Open-set/unknown-country fallback: search globally rather than only "unknown".
        if not country:
            out = []
            for (c, k), vals in index.items():
                if k == key:
                    out.extend(vals)
                    if len(out) >= limit:
                        return out[:limit]
        return []

    def _tfidf_candidates(self, query_text, vectorizer, matrix, top_k, min_sim):
        if not query_text or matrix is None:
            return []
        q = vectorizer.transform([query_text])
        sims = linear_kernel(q, matrix).ravel()
        if len(sims) == 0:
            return []
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(str(self.reference.iloc[i].entity_id), float(sims[i])) for i in idx if sims[i] >= min_sim]

    def _row_candidates(self, row):
        country = str(row.country_normalized)
        candidates = {}
        def add(eid, route, sim=0.0):
            old = candidates.get(eid)
            if old is None:
                candidates[eid] = {"routes": {route}, "blocking_similarity": sim}
            else:
                old["routes"].add(route)
                old["blocking_similarity"] = max(old["blocking_similarity"], sim)

        if row.name_normalized:
            for eid in self._lookup(self._name_exact, country, row.name_normalized, self.config.max_candidates_per_entity):
                add(eid, "name_exact", 1.0)
            for tok in set(row.name_normalized.split()):
                if len(tok) >= self.config.min_token_length:
                    for eid in self._lookup(self._name_tokens, country, tok, self.config.name_token_limit):
                        add(eid, "name_token")
            for eid, sim in self._tfidf_candidates(row.name_normalized, self._name_vectorizer, self._name_matrix,
                                                   self.config.tfidf_top_k, self.config.tfidf_min_similarity):
                add(eid, "name_tfidf", sim)

        if row.address_normalized:
            for eid in self._lookup(self._address_exact, country, row.address_normalized, self.config.max_candidates_per_entity):
                add(eid, "address_exact", 1.0)
            for tok in set(row.address_normalized.split()):
                if len(tok) >= self.config.min_token_length:
                    for eid in self._lookup(self._address_tokens, country, tok, self.config.address_token_limit):
                        add(eid, "address_token")
            for eid, sim in self._tfidf_candidates(row.address_normalized, self._address_vectorizer, self._address_matrix,
                                                   self.config.tfidf_top_k, self.config.tfidf_min_similarity):
                add(eid, "address_tfidf", sim)

        if row.house_number:
            for eid in self._lookup(self._house, country, row.house_number, self.config.numeric_limit):
                add(eid, "house_number")
        if row.postal_code:
            for eid in self._lookup(self._postal, country, row.postal_code, self.config.numeric_limit):
                add(eid, "postal_code")

        # Country is a filter when present. Candidates returned by indexes already obey it.
        # TF-IDF is globally ranked; enforce country here unless query country is empty.
        if country:
            valid = set(self.reference.loc[
                self.reference["country_normalized"].astype(str) == country, "entity_id"
            ].astype(str))
            candidates = {eid: meta for eid, meta in candidates.items() if eid in valid}

        ranked = sorted(
            candidates.items(),
            key=lambda kv: (
                -len(kv[1]["routes"]),
                -kv[1]["blocking_similarity"],
                kv[0],
            ),
        )[: self.config.max_candidates_per_entity]

        return ranked

    def generate(self, source1: pd.DataFrame):
        queries = add_normalized_columns(source1)
        rows = []
        counts = []
        for row in queries.itertuples(index=False):
            ranked = self._row_candidates(row)
            counts.append(len(ranked))
            for eid, meta in ranked:
                rows.append({
                    "source1_entity_id": str(row.entity_id),
                    "candidate_entity_id": eid,
                    "blocking_similarity": float(meta["blocking_similarity"]),
                    "blocking_routes": "|".join(sorted(meta["routes"])),
                })
        pairs = pd.DataFrame(rows, columns=[
            "source1_entity_id", "candidate_entity_id", "blocking_similarity", "blocking_routes"
        ])
        total_possible = len(source1) * len(self.reference)
        stats = {
            "average_candidates": float(np.mean(counts)) if counts else 0.0,
            "median_candidates": float(np.median(counts)) if counts else 0.0,
            "maximum_candidates": float(max(counts)) if counts else 0.0,
            "candidate_reduction_ratio": 1.0 - len(pairs) / total_possible if total_possible else 1.0,
        }
        return pairs, stats

def evaluate_blocking_recall(candidate_pairs: pd.DataFrame, ground_truth: dict[str, set[str]]) -> float:
    grouped = candidate_pairs.groupby("source1_entity_id")["candidate_entity_id"].agg(set).to_dict()
    # Only entities with non-empty truth are relevant to recall ceiling.
    truth_items = [(eid, truth) for eid, truth in ground_truth.items() if truth]
    if not truth_items:
        return 1.0
    covered = sum(truth.issubset(grouped.get(eid, set())) for eid, truth in truth_items)
    return covered / len(truth_items)
