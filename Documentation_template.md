# Methodology — Business Entity Resolution

## 1. Problem
Match every Source-1 business to all matching Source-2/Source-3 records.

## 2. Preprocessing
- Unicode normalization
- Accent removal
- Offline transliteration signal
- Punctuation/whitespace normalization
- Conservative trailing legal-suffix removal
- Address abbreviation normalization
- Country kept as an open-set field
- House number, unit number, postal code and numeric tokens extracted

## 3. Candidate generation
Independent candidate routes:
- exact normalized name
- exact normalized address
- shared name tokens
- shared address tokens
- house number
- postal code
- character TF-IDF name retrieval
- character TF-IDF address retrieval

All routes are UNIONed and deduplicated before the final candidate cap.

## 4. Pairwise features
Name:
- exact match
- token Jaccard
- token-sort similarity
- partial similarity
- edit similarity
- character similarity
- token overlap
- length features
- TF-IDF cosine

Address:
- same family of features

Structured:
- country
- house number
- unit number
- postal code
- numeric-token overlap

Blocking similarity is retained as a model feature.

## 5. Model
LightGBM binary classifier. If LightGBM is unavailable, HistGradientBoostingClassifier is used.

## 6. Training
Source-1 grouped train/validation split prevents records belonging to the same Source-1 entity from crossing the split.

All discovered positives are retained. Hard negatives and random negatives are sampled to make the classifier learn precision-sensitive distinctions.

## 7. Threshold
Threshold is selected using macro F0.5 on validation data.

## 8. Inference
Test Source-1 records are processed in chunks. Candidate pairs are generated first, then only candidate pairs are scored.

## 9. Output
- `matching_results.tsv`
- `candidate_pairs.tsv`

Every final match is produced from the final candidate set.

## 10. Constraints
No external business identity lookup, geocoding, government registry lookup, or external business-data augmentation is used.
