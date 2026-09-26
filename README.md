# Amazon ML Challenge 2026 — Hybrid Business Entity Resolution

## Final pipeline

This version combines the two reviewed approaches:

1. Conservative, shared preprocessing
2. Open-set country handling
3. Exact name/address indexes
4. Name/address token blocking
5. House-number and postal-code blocking
6. Character TF-IDF name retrieval
7. Character TF-IDF address retrieval
8. UNION of independent blocking routes
9. Candidate cap (default 300)
10. Rich pairwise similarity features
11. LightGBM matcher (sklearn fallback)
12. Source-1 grouped validation
13. Macro F0.5 threshold tuning
14. Chunked test prediction

The challenge requires all test Source-1 entities to appear in the output and requires final matches to be a subset of `candidate_pairs.tsv`.

## Folder structure

```text
amazon-ml-challenge/
├── dataset/
│   ├── train/
│   └── test/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── evaluate.py
│   ├── evaluate_model.py
│   ├── train.py
│   └── predict.py
├── models/
├── output/
├── experiments/
├── notebooks/
├── requirements.txt
├── README.md
└── Documentation_template.md
```

## Setup

```bash
cd ~/Desktop/amazon-ml-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. First run: 1,000 Source-1 training sample

```bash
python3 -m src.train \
  --dataset-root dataset \
  --sample-source1 1000 \
  --model-path models/entity_resolution_model.joblib
```

Look for:
- `blocking_recall`
- `average_candidates`
- `candidate_reduction_ratio`
- validation `macro_f0_5`
- selected threshold

Do not move to full prediction until the baseline is working.

## 2. Generate test submission

```bash
python3 -m src.predict \
  --dataset-root dataset \
  --model-path models/entity_resolution_model.joblib \
  --output-dir output/full_submission \
  --chunk-size 5000
```

For a larger machine/RAM budget, try:

```bash
--chunk-size 10000
```

## 3. Validate

Copy the challenge-provided `utils/validate_submission.py` into the project before running:

```bash
python3 utils/validate_submission.py \
  --matching output/full_submission/matching_results.tsv \
  --candidate output/full_submission/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Important design decisions

### Blocking is UNION, not sequential filtering

We do not require a record to pass name blocking before address blocking. A true pair can have a noisy name but a strong address, or vice versa.

### Character TF-IDF is real TF-IDF

The vectorizer is fitted over the reference corpus and uses character n-grams `(3,5)`. It is not raw word-count cosine.

### Numeric signals

House number, postal code, unit number and numeric-token overlap are matching features and also provide candidate-generation routes.

### Negative sampling

Training keeps all discovered positives and combines hard negatives with random negatives. Hard negatives are important because F0.5 is precision-heavy.

### Multilingual handling

No external translation or business lookup is used. Unicode normalization and offline transliteration are baseline signals. Multilingual embeddings are intentionally a later experiment, not part of this baseline.

## Tuning order

1. Blocking recall
2. Candidate count/reduction
3. F0.5 threshold
4. Hard-negative sampling
5. Candidate cap
6. Character n-gram range
7. Additional multilingual representation

Do not change everything at once.

## Challenge compliance

Use only the supplied challenge data. Do not add external business databases, geocoding, identity APIs, web lookups or external business-data augmentation.
